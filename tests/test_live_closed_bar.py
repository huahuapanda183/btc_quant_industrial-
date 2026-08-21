"""Live infer must match offline FeatureBuilder: closed 1m only, no live L2 in seq."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from get_train_data import build_offline_dataset
from modules.bars import ClosedBarClock, parse_kline_message, replay_closed_klines
from modules.features import (
    BOOK_FEATURE_IDX,
    FeatureBuilder,
    build_from_kline,
    synth_kline_events,
)
from modules.ledger import PaperBroker
from modules.risk import EXIT_REASONS, RiskController


def _klines_df(n=40, start_ms=1_700_000_000_000, px0=100.0):
    rows = []
    px = float(px0)
    for i in range(n):
        o = px
        c = px + (0.12 if i % 3 else -0.07)
        h = max(o, c) + 0.25
        l = min(o, c) - 0.18
        v = 10.0 + i
        rows.append({
            "time": pd.Timestamp(start_ms + i * 60_000, unit="ms", tz="UTC"),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
        px = c
    return pd.DataFrame(rows)


class ClosedBarOnlyTests(unittest.TestCase):
    def test_forming_bar_skipped_clock_and_builder(self):
        clock = ClosedBarClock()
        fb = FeatureBuilder(seq_len=5, k_levels=3)
        forming = {
            "channel": "kline",
            "open_ts_ms": 1_700_000_000_000,
            "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.4, "v": 12.0,
            "closed": False,
        }
        self.assertIsNone(clock.ingest_kline_msg(forming))
        self.assertEqual(len(fb.buf), 0)

        # Binance-style forming print
        bn = {"k": {"t": 1_700_000_000_000, "o": "100", "h": "101", "l": "99",
                    "c": "100.4", "v": "12", "x": False}}
        parsed = parse_kline_message(bn)
        self.assertFalse(parsed["closed"])
        self.assertIsNone(clock.ingest_kline_msg(bn))
        self.assertEqual(len(fb.buf), 0)

        closed_msg = dict(forming, closed=True)
        bar = clock.ingest_kline_msg(closed_msg)
        self.assertIsNotNone(bar)
        seq = build_from_kline(fb, bar.open, bar.high, bar.low, bar.close, bar.volume)
        self.assertEqual(len(fb.buf), 1)
        self.assertIsNone(seq)  # seq_len=5 not yet full

        # duplicate close must not enter FeatureBuilder again
        self.assertIsNone(clock.ingest_kline_msg(closed_msg))
        self.assertEqual(len(fb.buf), 1)

    def test_runner_skips_forming_then_builds_once(self):
        from main import SymbolRunner

        r = SymbolRunner("btcusdt")
        before = len(r.fb.buf)
        forming = {
            "channel": "kline", "open_ts_ms": 1_700_000_060_000,
            "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.2, "v": 8.0, "closed": False,
        }
        self.assertIsNone(r._maybe_emit_closed_from_kline(forming))
        self.assertEqual(len(r.fb.buf), before)

        closed = dict(forming, closed=True)
        bar = r._maybe_emit_closed_from_kline(closed)
        self.assertIsNotNone(bar)
        r._on_closed_bar(bar)
        self.assertEqual(len(r.fb.buf), before + 1)
        self.assertIsNone(r._maybe_emit_closed_from_kline(closed))
        self.assertEqual(len(r.fb.buf), before + 1)

    def test_trade_aggregation_does_not_emit_forming_minute(self):
        clock = ClosedBarClock()
        t0 = 1_700_000_000_000
        self.assertIsNone(clock.ingest_trade(t0 + 1_000, 100.0, 1.0))
        self.assertIsNone(clock.ingest_trade(t0 + 30_000, 100.5, 1.0))
        closed = clock.ingest_trade(t0 + 60_000, 101.0, 1.0)
        self.assertIsNotNone(closed)
        self.assertEqual(closed.open_ts_ms, t0)
        self.assertEqual(closed.close, 100.5)
        self.assertIsNone(clock.ingest_trade(t0 + 90_000, 101.2, 1.0))


class OfflineParityTests(unittest.TestCase):
    def test_closed_klines_match_offline_builder(self):
        df = _klines_df(45)
        cfg = {
            "seq_len": 30,
            "input_size": 12,
            "label": {"horizon_minutes": 20, "drop_deadzone": False, "dead_zone": 0.0},
            "trading": {"taker_fee_bp": 2.0, "slippage_bp": 3.0},
        }
        ds = build_offline_dataset(cfg, df=df)

        fb_live = FeatureBuilder(seq_len=30, k_levels=3)
        clock = ClosedBarClock()
        live_seqs = []
        for _, row in df.iterrows():
            ts_ms = int(row["time"].timestamp() * 1000)
            self.assertIsNone(clock.ingest_kline(
                ts_ms, row["open"], row["high"], row["low"], row["close"], row["volume"], closed=False
            ))
            bar = clock.ingest_kline(
                ts_ms, row["open"], row["high"], row["low"], row["close"], row["volume"], closed=True
            )
            self.assertIsNotNone(bar)
            seq = build_from_kline(fb_live, bar.open, bar.high, bar.low, bar.close, bar.volume)
            if seq is not None:
                live_seqs.append(seq)

        self.assertGreaterEqual(len(live_seqs), 1)
        fb_off = FeatureBuilder(seq_len=30, k_levels=3)
        off_seqs = []
        for _, row in df.iterrows():
            seq = build_from_kline(fb_off, row["open"], row["high"], row["low"], row["close"], row["volume"])
            if seq is not None:
                off_seqs.append(seq)
        self.assertEqual(len(live_seqs), len(off_seqs))
        for a, b in zip(live_seqs, off_seqs):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

        replayed = replay_closed_klines(
            [(int(r["time"].timestamp() * 1000), r["open"], r["high"], r["low"], r["close"], r["volume"])
             for _, r in df.iterrows()],
            seq_len=30,
        )
        self.assertEqual(len(live_seqs), len(replayed))
        for a, b in zip(live_seqs, replayed):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)

        self.assertEqual(ds["X"].shape[1], 30)
        self.assertEqual(ds["X"].shape[2], 12)
        # labeled last_rows are FeatureBuilder rows from the same kline replay
        if len(ds["last_row"]):
            live_last = np.stack([s[-1] for s in live_seqs], axis=0)
            for row in ds["last_row"]:
                dist = np.max(np.abs(live_last - row), axis=1)
                self.assertLess(float(np.min(dist)), 1e-5)

    def test_get_train_data_uses_shared_kline_path(self):
        src = Path("get_train_data.py").read_text(encoding="utf-8")
        self.assertIn("build_from_kline", src)
        self.assertIn("synth_kline_events", src)


class NoLiveBookInModelSeqTests(unittest.TestCase):
    def test_live_book_fields_do_not_appear_in_model_seq(self):
        o, h, l, c, v = 100.0, 101.0, 99.0, 100.5, 10.0
        live_book = {
            "b": [["50.0", "1.0"]],
            "a": [["150.0", "9999.0"]],  # wild spread + ask-heavy imb
        }
        fb_kline = FeatureBuilder(seq_len=1, k_levels=3)
        seq = build_from_kline(fb_kline, o, h, l, c, v)
        self.assertIsNotNone(seq)

        fb_synth = FeatureBuilder(seq_len=1, k_levels=3)
        trade_evt, depth_evt = synth_kline_events(o, h, l, c, v)
        seq_synth = fb_synth.build(trade_evt, depth_evt)
        np.testing.assert_allclose(seq, seq_synth)

        fb_leaked = FeatureBuilder(seq_len=1, k_levels=3)
        seq_leaked = fb_leaked.build({"p": str(c), "q": str(max(v * 0.1, 1.0)), "m": False}, live_book)
        for name, idx in BOOK_FEATURE_IDX.items():
            self.assertAlmostEqual(float(seq[0, idx]), float(seq_synth[0, idx]), places=6, msg=name)
            self.assertFalse(
                np.isclose(float(seq[0, idx]), float(seq_leaked[0, idx]), atol=1e-8),
                msg=f"{name} matched live book — leak",
            )
        # close still comes from the kline, not the book mid
        self.assertAlmostEqual(float(seq[0, 0]), c, places=5)

    def test_main_does_not_attach_live_book_to_featurebuilder(self):
        src = Path("main.py").read_text(encoding="utf-8")
        self.assertNotIn("self.fb.build(trade, self._last_depth)", src)
        self.assertNotIn("fb.build(trade,", src)
        self.assertIn("update_closed_kline", src)
        self.assertIn("ClosedBarClock", src)
        self.assertIn("allow_new_opens", src)


class AllowNewOpensFreezeTests(unittest.TestCase):
    def test_config_default_false(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.assertFalse(bool((cfg.get("trading") or {}).get("allow_new_opens", False)))
        self.assertEqual(float((cfg.get("label") or {}).get("horizon_minutes")), 20.0)

    def test_executor_skips_open_but_fills_exit(self):
        from modules.executor import TradeExecutor

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ex = TradeExecutor(
            {"trading": {
                "enabled": True, "mode": "paper", "allow_new_opens": False,
                "base_notional_usd": 100, "taker_fee_bp": 0.0, "slippage_bp": 0.0,
            }},
            price_getter=lambda s: 100.0,
        )
        ex.paper = PaperBroker(journal_path=str(Path(tmp.name) / "paper.jsonl"))
        skipped = ex.execute("btcusdt", "BUY", "open")
        self.assertEqual(skipped["status"], "SKIP")
        self.assertIn("allow_new_opens", skipped["info"])
        self.assertEqual(ex.paper.snapshot("btcusdt")["side"], "HOLD")

        seeded = ex.paper.place("btcusdt", "BUY", 100.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0, reason="open")
        self.assertEqual(seeded["status"], "FILLED")
        closed = ex.execute("btcusdt", "SELL", "take_profit")
        self.assertEqual(closed["status"], "FILLED")
        self.assertEqual(ex.paper.snapshot("btcusdt")["side"], "HOLD")

    def test_risk_freeze_blocks_open_and_flip_not_exits(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        book = PaperBroker(journal_path=str(Path(tmp.name) / "j.jsonl"))
        risk = RiskController("btcusdt", position_view=lambda: book.snapshot("btcusdt"))
        risk.state_path = str(Path(tmp.name) / "risk.pkl")
        risk.allow_new_opens = False
        risk.fast_tp_enable = False

        d, reason = risk.judge("BUY", {"p": 100.0, "close": [100.0, 100.1]})
        self.assertEqual(d, "HOLD")
        self.assertEqual(reason, "opens_frozen")

        book.place("btcusdt", "BUY", 100.0, 100.0, slippage_bp=0.0, taker_fee_bp=0.0, reason="open")
        d, reason = risk.judge("SELL", {"p": 100.0, "close": [100.0, 100.1]})
        self.assertEqual(d, "HOLD")
        self.assertEqual(reason, "opens_frozen")

        d, reason = risk.judge("HOLD", {"p": 200.0, "close": [100.0, 100.1, 100.2]})
        self.assertEqual(d, "SELL")
        self.assertEqual(reason, "take_profit")
        self.assertIn(reason, EXIT_REASONS)
        self.assertEqual(book.snapshot("btcusdt")["side"], "BUY")  # ledger until fill


if __name__ == "__main__":
    unittest.main()
