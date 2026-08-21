# modules/executor.py
import time
import threading
from collections import deque, defaultdict

from modules.ledger import PaperBroker

__all__ = ["TradeExecutor", "PaperBroker"]


class TradeExecutor:
    """
    交易执行器
    - 支持 paper / live（live 预留，未接交易所 SDK 时返回未实现）
    - 限流与护栏：每小时限单、当日回撤熔断、幂等 3 秒窗口
    - reduce_only：当风控 reason 属于平仓类时仅允许减仓
    - 支持从外部注入 price_getter（last）与 ob_getter（best bid/ask）
    """
    def __init__(self, cfg, price_getter, ob_getter=None):
        tcfg = (cfg.get("trading") or {})
        self.enabled = bool(tcfg.get("enabled", False))
        self.mode = str(tcfg.get("mode", "paper")).lower()     # paper | live
        self.venue = str(tcfg.get("venue", "futures")).lower() # futures | spot（live时用）
        self.symbol_map = {str(k).lower(): str(v) for k, v in (tcfg.get("symbol_map") or {}).items()}

        self.base_notional = float(tcfg.get("base_notional_usd", 200))
        self.max_pos = int(tcfg.get("max_pos_per_symbol", 1))
        self.leverage = int(tcfg.get("leverage", 2))
        self.reduce_only_on_exit = bool(tcfg.get("reduce_only_on_exit", True))
        self.max_trades_per_hour = int(tcfg.get("max_trades_per_hour", 6))
        self.max_daily_loss_bp = float(tcfg.get("max_daily_loss_bp", 150))
        self.taker_fee_bp = float(tcfg.get("taker_fee_bp", 0.0))
        self.slippage_bp = float(tcfg.get("slippage_bp", 5))
        self.order_type = str(tcfg.get("order_type", "MARKET")).upper()
        self.testnet = bool(tcfg.get("testnet", True))
        # Default false: p must not size or flip. Exits (reduce-only) still fill.
        self.allow_new_opens = bool(tcfg.get("allow_new_opens", False))

        # 依赖注入
        self.price_getter = price_getter          # fn(symbol)->last_price
        self.ob_getter = ob_getter                # fn(symbol)->(best_bid, best_ask)

        # 节流/幂等
        self.hour_buckets = defaultdict(deque)    # symbol -> timestamps within 1h
        self.idem_guard = {}                      # symbol -> last_exec_ts
        self.lock = threading.Lock()

        # 纸面经纪商
        self.paper = PaperBroker()

        # TODO: live 接入位（Binance/OKX 等）
        self._live_ready = False  # 未接 SDK 前默认 False

    # ---------- 限流与护栏 ----------
    def _rate_limited(self, sym: str) -> bool:
        now = time.time()
        q = self.hour_buckets[sym]
        while q and now - q[0] > 3600:
            q.popleft()
        return len(q) >= self.max_trades_per_hour

    def _bump_rate(self, sym: str):
        self.hour_buckets[sym].append(time.time())

    def _daily_circuit_break(self) -> bool:
        # 使用纸面经纪商的权益与当日回撤估算
        return self.paper.equity_drawdown_bp() >= self.max_daily_loss_bp

    # ---------- 主执行 ----------
    def execute(self, symbol: str, decision: str, reason: str):
        """
        仅在风控判定为 BUY/SELL 时调用
        返回 dict：{status, fill_px?, qty?, info}
        """
        if not self.enabled:
            return {"status": "SKIP", "info": "trading disabled"}

        sym = str(symbol).lower()
        mapped = self.symbol_map.get(sym, sym.upper())  # live venue id only

        with self.lock:
            # 幂等：3 秒窗口不重复
            last_ts = self.idem_guard.get(sym, 0)
            if time.time() - last_ts < 3.0:
                return {"status": "SKIP", "info": "idem_guard"}

            # 限速
            if self._rate_limited(sym):
                return {"status": "SKIP", "info": "rate_limited"}

            # 当日熔断
            if self._daily_circuit_break():
                return {"status": "SKIP", "info": "daily_circuit_break"}

            # 价格与盘口
            last_px = float(self.price_getter(sym) or 0.0)
            if last_px <= 0:
                return {"status": "ERR", "info": "no_price"}

            best_bid, best_ask = (None, None)
            if self.ob_getter:
                try:
                    ob = self.ob_getter(sym)
                    if ob and len(ob) == 2:
                        best_bid, best_ask = float(ob[0] or 0.0), float(ob[1] or 0.0)
                except Exception:
                    best_bid, best_ask = (None, None)

            # 自适应滑点：若无盘口可用，则按最近的 spread 动态放大滑点
            dynamic_slip_bp = float(self.slippage_bp)
            if (not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0 or best_ask < best_bid):
                try:
                    if self.ob_getter:
                        ob2 = self.ob_getter(sym)
                        if ob2 and ob2[0] and ob2[1] and ob2[1] > 0 and ob2[0] > 0 and ob2[1] >= ob2[0]:
                            mid = 0.5 * (float(ob2[0]) + float(ob2[1]))
                            spread_bp = (float(ob2[1]) - float(ob2[0])) / max(mid, 1e-12) * 1e4
                            dynamic_slip_bp = max(dynamic_slip_bp, spread_bp)
                except Exception:
                    pass

            # notional（名义美元）
            notional = float(self.base_notional)
            if self.leverage > 1:
                notional = self.base_notional * float(self.leverage)

            # reduce_only：风控给出的平仓类原因
            reduce_only_reasons = {"take_profit", "stop_loss", "trail_take", "timeout", "breakeven", "fast_take"}
            reduce_only = self.reduce_only_on_exit and (reason in reduce_only_reasons)

            if (not reduce_only) and (not self.allow_new_opens):
                return {"status": "SKIP", "info": "allow_new_opens=false"}

            # === 执行 ===
            # Paper book is keyed by the runner symbol (lowercase). `mapped` is
            # only for a future live venue — paper must not fork a second book.
            if self.mode == "paper":
                resp = self.paper.place(
                    sym, decision, last_px, notional,
                    best_bid=best_bid, best_ask=best_ask,
                    reduce_only=reduce_only,
                    slippage_bp=dynamic_slip_bp,
                    taker_fee_bp=self.taker_fee_bp,
                    reason=reason
                )
                if resp.get("status") == "FILLED":
                    self._bump_rate(sym)
                    self.idem_guard[sym] = time.time()
                    return {
                        "status": "FILLED",
                        "fill_px": resp["price"],
                        "qty": resp["qty"],
                        "info": "paper",
                        "event": resp.get("event"),
                        "events": resp.get("events") or [resp.get("event")],
                        "pnl": resp.get("pnl"),
                    }
                return {"status": resp.get("status", "ERR"), "info": resp.get("info", "paper_error")}

            # ==== live（预留位）：未接交易所 SDK 时返回未实现 ====
            return {"status": "ERR", "info": "live_not_implemented_yet"}
