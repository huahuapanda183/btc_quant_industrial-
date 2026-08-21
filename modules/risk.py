# modules/risk.py
# 风控：ATR% 驱动的 TP/SL + 冷却 + 追踪/保本 + 时间平仓
# 新增：
#  - ATR 自适应追踪止盈（trail_take = max(trail_take_min, trail_k * ATR%)）
#  - 快速止盈（短时达到多倍 ATR 即落袋）
#  - MM 直通：当 reason_text 含 "mm_gate" 且开启直通开关时，绕过冷却/反手护栏/最小价差
import yaml
import pickle
import time
import numpy as np
import os
from typing import Tuple, Union, Dict, Any, Callable, Optional


EXIT_REASONS = {
    "take_profit", "stop_loss", "trail_take", "timeout", "breakeven", "fast_take"
}


class RiskController:
    """
    judge() 返回: (decision, reason)
      decision: "BUY"/"SELL"/"HOLD"
      reason:   "open"/"take_profit"/"stop_loss"/"trail_take"/"breakeven"/"timeout"/
                "cooldown"/"reverse_guard"/"min_move"/"no_change"/"fast_take"

    使用：
      - 在调用 judge 时，将 SignalFusion 的诊断字符串传入 price_data["reason_text"]
        （当包含 "mm_gate" 时视为做市直通信号）
      - 可传 price_data["cooldown_release"]=True 以在冷却期内事件化放行
      - 纸面交易必须注入 position_view（读 PaperBroker）。judge 只做决策，
        不改账本；反手/平仓都等 fill 之后才与账本对齐。
      - Unbound (no position_view): read-only / no-position. Never write a
        local pickle book as if it were fills. One ledger: PaperBroker.
    """
    def __init__(self, symbol: str, position_view: Optional[Callable[[], Dict[str, Any]]] = None):
        self.symbol = symbol.lower()
        self.position_view = position_view
        self.position = "HOLD"
        self.last_price: Union[float, None] = None
        self.last_trade_time: float = 0.0
        self.entry_time: float = 0.0
        self._peak: Union[float, None] = None
        self._trough: Union[float, None] = None

        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}

        rc = cfg.get("risk", {}) or {}

        # ===== 原有参数 =====
        self.cooldown_seconds = int(rc.get("cooldown_seconds", 60))
        self.tp_mult = float(rc.get("tp_mult", 2.0))
        self.sl_mult = float(rc.get("sl_mult", 2.0))
        self.tp_min = float(rc.get("tp_min", 0.003))
        self.sl_min = float(rc.get("sl_min", 0.003))
        self.future_holding = int(rc.get("future_holding", 30))  # 分钟
        self.breakeven_ratio = float(rc.get("breakeven_ratio", 0.5))

        # === 追踪止盈改为自适应 ===
        self.trail_take_min = float(rc.get("trail_take_min", 0.002))  # 最小追踪比例
        self.trail_take_k   = float(rc.get("trail_take_k", 2.0))      # 随 ATR% 放大倍数

        # 反手护栏 + 最小价差
        self.reverse_guard_seconds = int(rc.get("reverse_guard_seconds", 180))
        self.strong_flip_prob = float(rc.get("strong_flip_prob", 0.85))
        self.strong_flip_margin = float(rc.get("strong_flip_margin", 0.12))
        self.min_move_bp_cfg = rc.get("min_move_bp", 0.0015)
        if isinstance(self.min_move_bp_cfg, (int, float)):
            self.min_move_bp_cfg = {"default": float(self.min_move_bp_cfg)}
        elif not isinstance(self.min_move_bp_cfg, dict):
            self.min_move_bp_cfg = {"default": 0.0015}

        # === 快速止盈（短时间达到多倍 ATR 直接落袋）===
        ft_cfg = rc.get("fast_take", {}) or {}
        self.fast_tp_enable = bool(ft_cfg.get("enable", True))
        self.fast_tp_mult   = float(ft_cfg.get("tp_mult", 1.8))   # 达到 1.8*ATR% 即触发
        self.fast_tp_window_min = int(ft_cfg.get("window_min", 20))  # N 分钟内

        # === MM 直通（可绕过冷却/反手护栏/最小价差）===
        self.mm_bypass_cooldown = bool(rc.get("mm_bypass_cooldown", True))
        self.mm_bypass_reverse_guard = bool(rc.get("mm_bypass_reverse_guard", True))
        self.mm_bypass_min_move = bool(rc.get("mm_bypass_min_move", True))

        # 数值稳健性 & 保本武装状态
        self.eps = float(rc.get("eps", 1e-6))
        self._breakeven_armed = False

        tcfg = cfg.get("trading") or {}
        # Default false: freeze new opens / flips. TP/SL/timeout still emit exits.
        self.allow_new_opens = bool(tcfg.get("allow_new_opens", False))

        self.state_path = f"{self.symbol}_risk.pkl"
        self._pending_exit = None  # (decision, reason) until the ledger fill lands
        self._book_side = None
        self._book_entry_time = None
        self.load_state()

    # ========= 状态持久化 =========
    def load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                # Pickle is never a fill book. Bound: ledger is authoritative.
                # Unbound: stay no-position / read-only — do not restore a local book.
                if self.position_view is not None:
                    self.last_trade_time = data.get("last_trade_time", 0.0)
                    self._peak = data.get("peak", None)
                    self._trough = data.get("trough", None)
                    self._breakeven_armed = data.get("breakeven_armed", False)
                else:
                    self.position = "HOLD"
                    self.last_price = None
                    self.entry_time = 0.0
                    self.last_trade_time = data.get("last_trade_time", 0.0)
            else:
                self.position, self.last_price = "HOLD", None
        except Exception:
            self.position, self.last_price = "HOLD", None
            try:
                os.remove(self.state_path)
            except Exception:
                pass

    def save_state(self):
        # Unbound: never persist a local position book. Pickle is not fills.
        if self.position_view is None:
            return
        state = {
            "position": "HOLD",
            "last_price": None,
            "last_trade_time": self.last_trade_time,
            "entry_time": 0.0,
            "peak": self._peak,
            "trough": self._trough,
            "breakeven_armed": self._breakeven_armed
        }
        with open(self.state_path, "wb") as f:
            pickle.dump(state, f)

    def _normalize_side(self, side: Any) -> str:
        s = str(side or "HOLD").upper()
        if s in ("BUY", "LONG"):
            return "BUY"
        if s in ("SELL", "SHORT"):
            return "SELL"
        return "HOLD"

    def _sync_from_ledger(self):
        """Copy the authoritative book into local snapshot fields. No fills."""
        if self.position_view is None:
            return
        try:
            snap = self.position_view() or {}
        except Exception:
            return
        side = self._normalize_side(snap.get("side", "HOLD"))
        entry = float(snap.get("entry") or 0.0) or None
        entry_time = float(snap.get("entry_time") or 0.0)
        if side != self._book_side or entry_time != self._book_entry_time:
            self._book_side = side
            self._book_entry_time = entry_time
            if side == "BUY":
                self._peak = entry
                self._trough = None
                self._breakeven_armed = False
            elif side == "SELL":
                self._trough = entry
                self._peak = None
                self._breakeven_armed = False
            else:
                self._peak = self._trough = None
                self._breakeven_armed = False
                self._pending_exit = None
        self.position = side
        self.last_price = entry if side in ("BUY", "SELL") else None
        self.entry_time = entry_time if side in ("BUY", "SELL") else 0.0

    def in_position(self) -> bool:
        self._sync_from_ledger()
        return self.position in ("BUY", "SELL")

    def on_fill(self, decision: str, reason: str, price: float, exec_resp: Optional[Dict[str, Any]] = None):
        """Update tracking after a PaperBroker fill. Does not write the book."""
        if self.position_view is None:
            # Unbound: no local book. Ignore fill-shaped mutations.
            return
        self.last_trade_time = time.time()
        events = []
        if exec_resp:
            events = list(exec_resp.get("events") or [])
            if exec_resp.get("event"):
                events.append(str(exec_resp.get("event")))
        ev_join = " ".join(str(e) for e in events)
        if reason in EXIT_REASONS or "CLOSE" in ev_join:
            self._pending_exit = None
            self._peak = self._trough = None
            self._breakeven_armed = False
        if reason == "open" or "OPEN" in ev_join or "REVERSE" in ev_join:
            self._pending_exit = None
            self.entry_time = time.time()
            if decision == "BUY":
                self._peak = float(price)
                self._trough = None
            elif decision == "SELL":
                self._trough = float(price)
                self._peak = None
            self._breakeven_armed = False
        self.save_state()

    # ========= 工具 =========
    @staticmethod
    def _atr_pct_from_close(close_arr):
        # 简化：rolling std 近似 ATR_abs；建议上游传真实 ATR_abs
        if close_arr is None or len(close_arr) < 2:
            return 0.003
        arr = np.asarray(close_arr, dtype=np.float64)
        std5 = float(np.std(arr[-5:])) if len(arr) >= 5 else float(np.std(arr))
        c = float(arr[-1])
        return max(1e-5, std5 / max(c, 1e-6))

    def _min_move_band(self, price: float) -> float:
        """按 symbol 取最小价差带（基点），换算成绝对价格带。"""
        bp_map = {k.lower(): float(v) for k, v in self.min_move_bp_cfg.items()}
        bp = bp_map.get(self.symbol, bp_map.get("default", 0.0015))
        return max(0.01, price * bp)

    @staticmethod
    def _is_mm_gate(reason_text: Any) -> bool:
        return isinstance(reason_text, str) and ("mm_gate" in reason_text)

    # ========= 主判定 =========
    def judge(self, signal: str, price_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        price_data:
          - 'p' 或 'price'（必需）
          - 'close' (历史收盘数组) 用于估计 ATR%
          - 'p_hat_prob' 与 'p_min'（可选）用于反手护栏“超强信号”
          - 'reason_text'（可选）传入 SignalFusion 的 diag；含 "mm_gate" 视为 MM 直通
          - 'cooldown_release'（可选）冷却内事件放行
        """
        if self.position_view is None:
            # Fail closed: no second book. Stay flat / read-only.
            self.position = "HOLD"
            self.last_price = None
            self.entry_time = 0.0
            return "HOLD", "unbound_no_ledger"

        self._sync_from_ledger()
        p = float(price_data.get("p", price_data.get("price", 0.0)))
        closes = price_data.get("close", None)
        reason_text = price_data.get("reason_text", "")
        is_mm = self._is_mm_gate(reason_text)

        atr_pct = self._atr_pct_from_close(closes)
        now = time.time()
        if self._pending_exit and self.position in ("BUY", "SELL"):
            # still waiting for the reduce-only fill; re-issue the same exit
            return self._pending_exit

        # 冷却期：不允许新开/反手（MM 可选择绕过；或事件释放）
        in_cooldown = (now - self.last_trade_time) < self.cooldown_seconds
        if in_cooldown and not (is_mm and self.mm_bypass_cooldown):
            cooldown_release = bool(price_data.get("cooldown_release", False))
            moved_out_band = False
            if self.last_price:
                band = self._min_move_band(self.last_price)
                moved_out_band = abs(p - self.last_price) >= band
            if not (cooldown_release or moved_out_band):
                return "HOLD", "cooldown"

        # 期望 TP/SL（相对）
        TP = max(self.tp_mult * atr_pct, self.tp_min)
        SL = max(self.sl_mult * atr_pct, self.sl_min)

        # 自适应追踪止盈：随 ATR% 变化
        trail_dyn = max(self.trail_take_min, self.trail_take_k * atr_pct)

        # ===== 持仓管理 =====
        if self.position == "BUY" and self.last_price:
            ret = (p - self.last_price) / max(self.last_price, 1e-12)
            self._peak = max(self._peak or p, p)

            # 保本武装：浮盈达到一定比例后，允许回撤到 0 即平（reason=breakeven）
            if (not self._breakeven_armed) and (ret >= self.breakeven_ratio * TP):
                self._breakeven_armed = True

            # 快速止盈：N 分钟内达到多倍 ATR 直接走
            if self.fast_tp_enable and self.entry_time and (now - self.entry_time) <= self.fast_tp_window_min * 60:
                if ret >= self.fast_tp_mult * atr_pct:
                    return self._emit_exit("SELL", "fast_take")

            # 常规 TP
            if ret >= TP:
                return self._emit_exit("SELL", "take_profit")

            # 止损 / 保本触发
            if self._breakeven_armed:
                if ret <= 0.0 + self.eps:
                    return self._emit_exit("SELL", "breakeven")
            else:
                if ret <= -SL:
                    return self._emit_exit("SELL", "stop_loss")

            # 追踪止盈
            if self._peak and (self._peak - p) / max(self._peak, 1e-12) >= trail_dyn:
                return self._emit_exit("SELL", "trail_take")

            # 时间平仓
            if self.entry_time and (now - self.entry_time) >= self.future_holding * 60:
                return self._emit_exit("SELL", "timeout")

        elif self.position == "SELL" and self.last_price:
            ret = (self.last_price - p) / max(self.last_price, 1e-12)
            self._trough = min(self._trough or p, p)

            # 保本武装（空头同理）
            if (not self._breakeven_armed) and (ret >= self.breakeven_ratio * TP):
                self._breakeven_armed = True

            # 快速止盈
            if self.fast_tp_enable and self.entry_time and (now - self.entry_time) <= self.fast_tp_window_min * 60:
                if ret >= self.fast_tp_mult * atr_pct:
                    return self._emit_exit("BUY", "fast_take")

            # 常规 TP
            if ret >= TP:
                return self._emit_exit("BUY", "take_profit")

            # 止损 / 保本触发（空头）
            if self._breakeven_armed:
                if ret <= 0.0 + self.eps:
                    return self._emit_exit("BUY", "breakeven")
            else:
                if ret <= -SL:
                    return self._emit_exit("BUY", "stop_loss")

            # 追踪止盈（空头）
            if self._trough and (p - self._trough) / max(self._trough, 1e-12) >= trail_dyn:
                return self._emit_exit("BUY", "trail_take")

            # 时间平仓
            if self.entry_time and (now - self.entry_time) >= self.future_holding * 60:
                return self._emit_exit("BUY", "timeout")

        # ===== 反手护栏：持仓后 N 秒内禁止反手（MM 可选择绕过）=====
        if self.position in ("BUY", "SELL") and signal in ("BUY", "SELL") and signal != self.position:
            if not (is_mm and self.mm_bypass_reverse_guard):
                if (now - self.last_trade_time) < self.reverse_guard_seconds:
                    p_hat_prob = float(price_data.get("p_hat_prob", 0.0))
                    p_min = float(price_data.get("p_min", 0.0))
                    strong_ok = (p_hat_prob >= self.strong_flip_prob) and ((p_hat_prob - p_min) >= self.strong_flip_margin)
                    if not strong_ok:
                        return "HOLD", "reverse_guard"

        # ===== 最小价差带：价格没走出带宽，不换向（MM 可选择绕过）=====
        if self.position in ("BUY", "SELL") and signal in ("BUY", "SELL") and signal != self.position and self.last_price:
            if not (is_mm and self.mm_bypass_min_move):
                band = self._min_move_band(self.last_price)
                if abs(p - self.last_price) < band:
                    return "HOLD", "min_move"

        # ===== 新开仓 / 反手（只决策，账本由 PaperBroker 平后开）=====
        if signal != "HOLD" and signal != self.position:
            if not self.allow_new_opens:
                return "HOLD", "opens_frozen"
            return signal, "open"

        return "HOLD", "no_change"

    def _emit_exit(self, decision: str, reason: str):
        if self.position_view is None:
            return "HOLD", "unbound_no_ledger"
        self._pending_exit = (decision, reason)
        self.save_state()
        return decision, reason

    def _reset_position(self):
        """Tracking reset only. Must not be the paper fill. Unbound is a no-op."""
        if self.position_view is None:
            self.position = "HOLD"
            self.last_price = None
            self.entry_time = 0.0
            return
        self.position = "HOLD"
        self.last_price = None
        self.last_trade_time = time.time()
        self.entry_time = 0.0
        self._peak, self._trough = None, None
        self._breakeven_armed = False
        self._pending_exit = None
        self.save_state()
