"""Reverse is close-then-open; one book used by risk exits and paper fills."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from modules.ledger import AUTHORITATIVE_LEDGER, PaperBroker
from modules.risk import RiskController


class LedgerReverseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = Path(self.tmp.name) / "paper_trades.jsonl"
        self.book = PaperBroker(journal_path=str(self.journal))

    def tearDown(self):
        self.tmp.cleanup()

    def _read_journal(self):
        rows = []
        for ln in self.journal.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(ln))
        return rows

    def test_reverse_writes_close_then_open_and_realizes_pnl(self):
        start = self.book.equity
        # LONG 100 notional @ 10000, 2bp taker, 0 slip
        o = self.book.place("btcusdt", "BUY", 10000.0, 100.0, slippage_bp=0.0, taker_fee_bp=2.0, reason="open")
        self.assertEqual(o["event"], "OPEN_LONG")
        qty = o["qty"]
        self.assertAlmostEqual(qty, 0.01)
        fee_open = 10000.0 * qty * 0.0002
        self.assertAlmostEqual(self.book.equity, start - fee_open, places=8)

        # reverse to SHORT @ 11000: must CLOSE_LONG then OPEN_SHORT
        r = self.book.place("btcusdt", "SELL", 11000.0, 100.0, slippage_bp=0.0, taker_fee_bp=2.0, reason="open")
        self.assertEqual(r["event"], "REVERSE")
        self.assertEqual(r["events"], ["CLOSE_LONG", "OPEN_SHORT"])
        self.assertEqual(self.book.snapshot("btcusdt")["side"], "SELL")

        rows = self._read_journal()
        self.assertEqual([x["event"] for x in rows], ["OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT"])
        close = rows[1]
        pnl_gross = (11000.0 - 10000.0) * qty
        fee_close = 11000.0 * qty * 0.0002
        self.assertAlmostEqual(close["pnl_gross"], pnl_gross, places=8)
        self.assertAlmostEqual(close["fee_total"], fee_open + fee_close, places=8)
        self.assertAlmostEqual(close["fees"], close["fee_total"], places=8)
        self.assertAlmostEqual(close["pnl_net"], pnl_gross - fee_open - fee_close, places=8)
        self.assertEqual(close["reason"], "open")
        # journal contract used by report scripts
        for key in ("pnl_net", "pnl_gross", "fee_total", "pnl"):
            self.assertIn(key, close)

        # no silent overwrite: CLOSE qty is the old position, not a new overwrite
        self.assertAlmostEqual(close["qty"], qty, places=8)

        # equity matches journal deltas
        self.assertAlmostEqual(
            self.book.equity - start,
            sum(x["equity_delta"] for x in rows),
            places=8,
        )

    def test_same_side_does_not_overwrite(self):
        self.book.place("btcusdt", "BUY", 100.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0, reason="open")
        entry = self.book.snapshot("btcusdt")["entry"]
        skip = self.book.place("btcusdt", "BUY", 120.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0, reason="open")
        self.assertEqual(skip["status"], "SKIP")
        self.assertAlmostEqual(self.book.snapshot("btcusdt")["entry"], entry)

    def test_risk_exit_does_not_flatten_book_until_fill(self):
        self.book.place("btcusdt", "BUY", 100.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0, reason="open")
        risk = RiskController("btcusdt", position_view=lambda: self.book.snapshot("btcusdt"))
        risk.state_path = str(Path(self.tmp.name) / "risk.pkl")
        risk.fast_tp_enable = False
        d, reason = risk.judge("HOLD", {"p": 200.0, "close": [100.0, 100.1, 100.2]})
        self.assertEqual(d, "SELL")
        self.assertEqual(reason, "take_profit")
        self.assertEqual(self.book.snapshot("btcusdt")["side"], "BUY")  # still open
        filled = self.book.place(
            "btcusdt", "SELL", 200.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0,
            reduce_only=True, reason=reason,
        )
        self.assertEqual(filled["event"], "CLOSE_LONG")
        self.assertEqual(self.book.snapshot("btcusdt")["side"], "HOLD")
        risk.on_fill(d, reason, 200.0, filled)
        self.assertFalse(risk.in_position())

    def test_authoritative_ledger_constant(self):
        self.assertEqual(AUTHORITATIVE_LEDGER, "modules.ledger.PaperBroker")
        self.assertIs(PaperBroker, PaperBroker)

    def test_unbound_risk_cannot_open_or_overwrite_local_position(self):
        """No position_view: stay flat / read-only. Never write a local fill book."""
        risk = RiskController("btcusdt")  # unbound
        risk.state_path = str(Path(self.tmp.name) / "unbound_risk.pkl")
        # leftover pickle that used to be treated as a local book
        stale = {
            "position": "BUY",
            "last_price": 100.0,
            "last_trade_time": 1.0,
            "entry_time": 1.0,
            "peak": 100.0,
            "trough": None,
            "breakeven_armed": False,
        }
        with open(risk.state_path, "wb") as f:
            pickle.dump(stale, f)
        risk.load_state()
        self.assertEqual(risk.position, "HOLD")
        self.assertIsNone(risk.last_price)

        before = Path(risk.state_path).read_bytes()
        d, reason = risk.judge("BUY", {"p": 100.0, "close": [100.0, 100.1]})
        self.assertEqual(d, "HOLD")
        self.assertEqual(reason, "unbound_no_ledger")
        self.assertEqual(risk.position, "HOLD")
        self.assertIsNone(risk.last_price)

        # must not rewrite pickle as if a fill landed (stale BUY on disk is ignored)
        self.assertEqual(Path(risk.state_path).read_bytes(), before)
        risk.save_state()
        self.assertEqual(Path(risk.state_path).read_bytes(), before)

        # on_fill must not open/overwrite a local book either
        risk.on_fill("BUY", "open", 110.0, {"event": "OPEN_LONG", "status": "FILLED"})
        self.assertEqual(risk.position, "HOLD")
        self.assertIsNone(risk.last_price)
        self.assertFalse(risk.in_position())
        self.assertEqual(Path(risk.state_path).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
