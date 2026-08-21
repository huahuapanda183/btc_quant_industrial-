"""Label construction for a known price path and clock horizon."""
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import yaml

from modules.labels import (
    build_direction_labels,
    direction_label,
    future_return_at_horizon,
    horizon_minutes_from_config,
    kappa_from_config,
    time_holdout_purge_embargo,
)


class LabelTests(unittest.TestCase):
    def test_kappa_from_trading_fees(self):
        cfg = {"trading": {"taker_fee_bp": 2.0, "slippage_bp": 3.0}, "label": {"dead_zone": 0.0}}
        # round-trip 2*(2+3)/1e4 = 0.001
        self.assertAlmostEqual(kappa_from_config(cfg), 0.001)

    def test_known_price_path_20m_horizon(self):
        t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        # 1-minute prints: flat 100 for 20m, then jump to 101.2
        ts, px = [], []
        for i in range(40):
            ts.append((t0 + timedelta(minutes=i)).timestamp())
            px.append(100.0 if i < 20 else 101.2)
        idxs, ys, rs = build_direction_labels(ts, px, horizon_minutes=20, kappa=0.001, drop_deadzone=True)
        # index 0: t=0 -> t+20m price 101.2, r=0.012 > 0.001 -> up
        self.assertIn(0, set(idxs.tolist()))
        self.assertEqual(int(ys[list(idxs).index(0)]), 1)
        self.assertAlmostEqual(float(rs[list(idxs).index(0)]), 0.012, places=6)
        # index 20: t=20m price 101.2 -> t+40m still 101.2, r=0 -> dead zone dropped
        self.assertNotIn(20, set(idxs.tolist()))

    def test_down_and_deadzone(self):
        self.assertEqual(direction_label(-0.02, 0.001), 0)
        self.assertIsNone(direction_label(0.0004, 0.001, drop_deadzone=True))
        self.assertEqual(direction_label(0.0004, 0.001, drop_deadzone=False), 0)

    def test_horizon_is_clock_not_bar_count(self):
        # irregular spacing: 5m then a 20m gap
        ts = np.array([0.0, 300.0, 1500.0])  # 0, 5m, 25m
        px = np.array([100.0, 100.0, 102.0])
        r = future_return_at_horizon(ts, px, 0, horizon_sec=20 * 60)
        self.assertAlmostEqual(r, 0.02)
        # 5-bar-style shift would have wrongly used index 5, which does not exist

    def test_purge_embargo_drops_overlapping_train(self):
        # samples every 5 minutes; horizon 20m; embargo 20m
        ts = np.arange(0, 120 * 60, 5 * 60, dtype=float)  # 24 points, 0..115m
        train, val, info = time_holdout_purge_embargo(ts, horizon_sec=20 * 60, val_frac=0.25, embargo_sec=20 * 60)
        t_val = info["t_val_start"]
        for i in train:
            self.assertLessEqual(ts[i] + 20 * 60 + 20 * 60, t_val + 1e-9)
        self.assertGreater(info["n_purged"], 0)
        self.assertGreater(len(val), 0)

    def test_config_yaml_horizon_is_sweepable(self):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        h = float((cfg.get("label") or {}).get("horizon_minutes"))
        self.assertGreaterEqual(h, 15)
        self.assertLessEqual(h, 60)
        # must not be a hardcoded "optimal" claim in config — just a number
        self.assertNotIn("optimal", str(cfg.get("label") or {}).lower())
        # dead trade-count horizon must not be a live config key
        self.assertNotIn("future_shift", cfg)

    def test_no_code_path_labels_with_future_shift_trades(self):
        """Clock horizon is the only label horizon. leftover future_shift is ignored."""
        leftover = {"future_shift": 5, "label": {"horizon_minutes": 20}}
        self.assertEqual(horizon_minutes_from_config(leftover), 20.0)
        # even if someone deletes label.horizon_minutes, do not fall back to 5 trades
        self.assertEqual(horizon_minutes_from_config({"future_shift": 5}), 20.0)

        live_reads = (
            '["future_shift"]',
            "['future_shift']",
            '.get("future_shift"',
            ".get('future_shift'",
            "cfg[\"future_shift\"]",
            "close.shift(-5)",
            "shift(-5)",
        )
        roots = [
            Path("modules/labels.py"),
            Path("modules/train_clf.py"),
            Path("get_train_data.py"),
            Path("train_models.py"),
            Path("main.py"),
        ]
        for path in roots:
            src = path.read_text(encoding="utf-8")
            for needle in live_reads:
                self.assertNotIn(
                    needle, src,
                    f"{path} still has a live trade-count label path: {needle}",
                )


if __name__ == "__main__":
    unittest.main()
