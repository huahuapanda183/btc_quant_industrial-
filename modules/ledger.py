# modules/ledger.py
"""Authoritative paper position book.

Paper trading has exactly one fill-authoritative ledger: PaperBroker.
Risk / SignalFusion / PhaseSim may *decide* or *report*, but they must not
keep a second mutable position that can diverge from fills.

Reverse = CLOSE (realize pnl_gross / fees / pnl_net) then OPEN.
Never overwrite qty/side/entry without a CLOSE event.

Risk TP/SL/timeout/fast_take/breakeven go through the same place() pipeline
(reduce-only close). PhaseSim is a reporter bound to this book, not a book.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


AUTHORITATIVE_LEDGER = "modules.ledger.PaperBroker"


def _empty_pos() -> dict:
    return {
        "side": "FLAT",
        "qty": 0.0,
        "entry": 0.0,
        "entry_fee": 0.0,
        "entry_time": 0.0,
    }


class PaperBroker:
    """Single paper ledger: positions, equity, journal."""

    def __init__(self, journal_path: str = "logs/paper_trades.jsonl"):
        self.position = defaultdict(_empty_pos)
        self.equity = 100000.0
        self.starting_equity = 100000.0
        self.daily_eq_hi = self.equity
        self.daily_eq_lo = self.equity
        self.trades: List[tuple] = []
        self.day = time.strftime("%Y-%m-%d")
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _key(self, sym: str) -> str:
        return str(sym).lower()

    def _roll_day(self):
        d = time.strftime("%Y-%m-%d")
        if d != self.day:
            self.day = d
            self.daily_eq_hi = self.equity
            self.daily_eq_lo = self.equity

    def equity_drawdown_bp(self):
        self._roll_day()
        if self.daily_eq_hi <= 0:
            return 0.0
        dd = (self.daily_eq_hi - self.equity) / self.daily_eq_hi
        return max(0.0, dd) * 10000.0

    def snapshot(self, sym: str) -> Dict[str, Any]:
        """Risk-facing view of the book (BUY/SELL/HOLD)."""
        pos = self.position[self._key(sym)]
        raw = pos.get("side", "FLAT")
        if raw == "LONG":
            side = "BUY"
        elif raw == "SHORT":
            side = "SELL"
        else:
            side = "HOLD"
        return {
            "side": side,
            "raw_side": raw,
            "qty": float(pos.get("qty") or 0.0),
            "entry": float(pos.get("entry") or 0.0),
            "entry_fee": float(pos.get("entry_fee") or 0.0),
            "entry_time": float(pos.get("entry_time") or 0.0),
        }

    def get_position(self, sym: str) -> dict:
        return dict(self.position[self._key(sym)])

    def mark_to_market(self, sym: str, price: float) -> float:
        pos = self.position[self._key(sym)]
        if pos["side"] == "LONG":
            return (price - pos["entry"]) * pos["qty"]
        if pos["side"] == "SHORT":
            return (pos["entry"] - price) * pos["qty"]
        return 0.0

    def _update_daily_extrema(self):
        self.daily_eq_hi = max(self.daily_eq_hi, self.equity)
        self.daily_eq_lo = min(self.daily_eq_lo, self.equity)

    def _append_journal(self, rec: dict):
        try:
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _fill_price_with_slippage(
        self,
        side: str,
        last_px: float,
        best_bid: float,
        best_ask: float,
        slippage_bp: float,
    ):
        if best_bid and best_ask and best_bid > 0 and best_ask > 0 and best_ask >= best_bid:
            return best_ask if side == "BUY" else best_bid
        if last_px <= 0:
            return 0.0
        slip = slippage_bp / 10000.0
        return last_px * (1.0 + slip if side == "BUY" else 1.0 - slip)

    def _flat(self, pos: dict):
        pos.update(_empty_pos())

    def _close_full(
        self,
        sym: str,
        pos: dict,
        px: float,
        taker_fee_bp: float,
        reason: str,
        ts: float,
        reduce_only: bool,
    ) -> dict:
        close_qty = float(pos["qty"])
        if pos["side"] == "SHORT":
            event = "CLOSE_SHORT"
            order_side = "BUY"
            pnl_gross = (pos["entry"] - px) * close_qty
        else:
            event = "CLOSE_LONG"
            order_side = "SELL"
            pnl_gross = (px - pos["entry"]) * close_qty
        fee_close = px * close_qty * (taker_fee_bp / 10000.0)
        fee_open = float(pos.get("entry_fee", 0.0))
        pnl_net = pnl_gross - fee_open - fee_close
        # Open fee was already deducted from equity at OPEN.
        equity_delta = pnl_gross - fee_close
        self.equity += equity_delta
        rec = self._record(
            ts=ts,
            sym=sym,
            side=order_side,
            event=event,
            qty=close_qty,
            px=px,
            pnl_net=pnl_net,
            pnl_gross=pnl_gross,
            fee_open=fee_open,
            fee_close=fee_close,
            reduce_only=reduce_only,
            reason=reason,
            equity_delta=equity_delta,
        )
        self._flat(pos)
        return rec

    def _open_side(
        self,
        sym: str,
        pos: dict,
        side: str,
        px: float,
        qty: float,
        taker_fee_bp: float,
        reason: str,
        ts: float,
    ) -> dict:
        book_side = "LONG" if side == "BUY" else "SHORT"
        event = "OPEN_LONG" if side == "BUY" else "OPEN_SHORT"
        fee_open = px * qty * (taker_fee_bp / 10000.0)
        pos["side"] = book_side
        pos["qty"] = qty
        pos["entry"] = px
        pos["entry_fee"] = fee_open
        pos["entry_time"] = ts
        equity_delta = -fee_open
        self.equity += equity_delta
        rec = self._record(
            ts=ts,
            sym=sym,
            side=side,
            event=event,
            qty=qty,
            px=px,
            pnl_net=-fee_open,
            pnl_gross=0.0,
            fee_open=fee_open,
            fee_close=0.0,
            reduce_only=False,
            reason=reason,
            equity_delta=equity_delta,
        )
        return rec

    def _record(
        self,
        ts: float,
        sym: str,
        side: str,
        event: str,
        qty: float,
        px: float,
        pnl_net: float,
        pnl_gross: float,
        fee_open: float,
        fee_close: float,
        reduce_only: bool,
        reason: str,
        equity_delta: float,
    ) -> dict:
        fees = float(fee_open + fee_close)
        rec = {
            "ts": ts,
            "symbol": sym,
            "side": side,
            "event": event,
            "qty": float(qty),
            "price": float(px),
            "pnl": float(pnl_net),
            "pnl_net": float(pnl_net),
            "pnl_gross": float(pnl_gross),
            "fee_open": float(fee_open),
            "fee_close": float(fee_close),
            "fee_total": fees,
            "fees": fees,
            "equity": float(self.equity),
            "equity_delta": float(equity_delta),
            "reduce_only": bool(reduce_only),
            "reason": reason,
        }
        self._append_journal(rec)
        self.trades.append((ts, sym, side, qty, px, event, pnl_net))
        return rec

    def place(
        self,
        sym: str,
        side: str,
        last_px: float,
        notional_quote: float,
        best_bid: float = None,
        best_ask: float = None,
        reduce_only: bool = False,
        slippage_bp: float = 5.0,
        taker_fee_bp: float = 0.0,
        reason: str = "",
    ):
        """Fill an order against the single book.

        Opposite open (reason=open or any non-reduce flip): CLOSE full size,
        journal it, then OPEN the new side. reduce_only exits only CLOSE.
        Same-side open does not overwrite an existing position.
        """
        self._roll_day()
        ts = time.time()
        sym = self._key(sym)
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            return {"status": "ERR", "info": "bad_side"}

        px = float(self._fill_price_with_slippage(side, last_px, best_bid, best_ask, slippage_bp))
        if px <= 0:
            return {"status": "ERR", "info": "bad_price"}

        qty = max(0.0001, float(notional_quote) / px)
        pos = self.position[sym]
        fills: List[dict] = []

        want_long = side == "BUY"
        in_long = pos["side"] == "LONG" and pos["qty"] > 1e-12
        in_short = pos["side"] == "SHORT" and pos["qty"] > 1e-12
        opposite = (want_long and in_short) or ((not want_long) and in_long)
        same = (want_long and in_long) or ((not want_long) and in_short)

        if same:
            if reduce_only:
                # reduce-only same-side is a no-op (cannot increase)
                return {"status": "SKIP", "info": "already_in_position", "event": "SKIP", "events": []}
            return {"status": "SKIP", "info": "already_in_position", "event": "SKIP", "events": []}

        if opposite:
            close_rec = self._close_full(sym, pos, px, taker_fee_bp, reason, ts, reduce_only)
            fills.append(close_rec)
            if reduce_only:
                self._update_daily_extrema()
                return {
                    "status": "FILLED",
                    "price": px,
                    "qty": close_rec["qty"],
                    "equity": self.equity,
                    "event": close_rec["event"],
                    "events": [close_rec["event"]],
                    "pnl": float(close_rec["pnl_net"]),
                    "fills": fills,
                }
            # reverse: open the new side after the close (never overwrite)
            open_rec = self._open_side(sym, pos, side, px, qty, taker_fee_bp, reason, ts)
            fills.append(open_rec)
            self._update_daily_extrema()
            return {
                "status": "FILLED",
                "price": px,
                "qty": qty,
                "equity": self.equity,
                "event": "REVERSE",
                "events": [close_rec["event"], open_rec["event"]],
                "pnl": float(close_rec["pnl_net"] + open_rec["pnl_net"]),
                "fills": fills,
            }

        if reduce_only:
            return {"status": "SKIP", "info": "flat_reduce_only", "event": "SKIP", "events": []}

        open_rec = self._open_side(sym, pos, side, px, qty, taker_fee_bp, reason, ts)
        fills.append(open_rec)
        self._update_daily_extrema()
        return {
            "status": "FILLED",
            "price": px,
            "qty": qty,
            "equity": self.equity,
            "event": open_rec["event"],
            "events": [open_rec["event"]],
            "pnl": float(open_rec["pnl_net"]),
            "fills": fills,
        }
