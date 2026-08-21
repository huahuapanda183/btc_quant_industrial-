"""predict() emits a classification probability, not σ(MSE return)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.labels import logits_to_calibrated_prob
from modules.model import (
    UncalibratedWeightsError,
    classification_meta_allows_probability,
)


class CalibratedProbTests(unittest.TestCase):
    def test_logits_to_prob_is_sigmoid_of_logit_over_T(self):
        z = 2.0
        T = 2.0
        p = logits_to_calibrated_prob(z, T)
        expected = 1.0 / (1.0 + np.exp(-z / T))
        self.assertAlmostEqual(p, expected, places=6)
        self.assertGreater(p, 0.5)
        # old bug: a near-zero *return* residual through sigmoid is ~0.5 by identity
        self.assertNotAlmostEqual(logits_to_calibrated_prob(0.0, 1.0), 0.73, places=2)
        self.assertAlmostEqual(logits_to_calibrated_prob(0.0, 1.0), 0.5, places=6)

    def test_model_manager_predict_uses_classification_logit(self):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            self.skipTest("torch not installed")
        from modules.model import ModelManager

        class ConstLogit(nn.Module):
            def __init__(self, z):
                super().__init__()
                self.z = float(z)

            def forward(self, x):
                b = x.shape[0]
                return torch.full((b, 1), self.z)

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as d:
            meta = {
                "head": "classification",
                "prob_meaning": "P(r_{t→t+20m} > κ) after temperature scaling",
            }
            Path(d, "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            mm = ModelManager("btcusdt", artifact_dir=d)
            mm.tft = ConstLogit(2.0)
            mm.nbeats = None
            mm.fallback = None
            mm.infer_as_probability = True
            mm.head = "classification"
            mm.T_tft = 2.0
            mm.blend_b = 0.0
            seq = np.zeros((30, 12), dtype=np.float32)
            label, p = mm.predict(seq)
            expected = logits_to_calibrated_prob(2.0, 2.0)
            self.assertAlmostEqual(p, expected, places=5)
            self.assertEqual(label, 1)
            self.assertEqual(mm.head, "classification")

    def test_missing_model_meta_refuses_infer_as_probability(self):
        from modules.model import ModelManager

        with tempfile.TemporaryDirectory() as d:
            # leftover MSE weights, no model_meta.json
            Path(d, "tft_model.pth").write_bytes(b"not-a-real-checkpoint")
            Path(d, "nbeats_model.pth").write_bytes(b"not-a-real-checkpoint")
            mm = ModelManager("btcusdt", artifact_dir=d)
            self.assertFalse(mm.infer_as_probability)
            self.assertIsNone(mm.tft)
            self.assertIsNone(mm.nbeats)
            self.assertIsNone(mm.fallback)
            with self.assertRaises(UncalibratedWeightsError) as ctx:
                mm.predict(np.zeros((30, 12), dtype=np.float32))
            msg = str(ctx.exception).lower()
            self.assertIn("mse", msg)
            self.assertIn("retrain", msg)
            self.assertFalse(classification_meta_allows_probability({}))
            self.assertFalse(classification_meta_allows_probability({"head": "classification"}))
            self.assertFalse(classification_meta_allows_probability(
                {"head": "regression", "prob_meaning": "predicted return"}
            ))
            self.assertTrue(classification_meta_allows_probability({
                "head": "classification",
                "prob_meaning": "P(r_{t→t+20m} > κ) after temperature scaling",
            }))

    def test_classification_head_without_prob_meaning_is_refused(self):
        from modules.model import ModelManager

        with tempfile.TemporaryDirectory() as d:
            Path(d, "tft_model.pth").write_bytes(b"not-a-real-checkpoint")
            Path(d, "model_meta.json").write_text(
                json.dumps({"head": "classification"}), encoding="utf-8"
            )
            mm = ModelManager("btcusdt", artifact_dir=d)
            self.assertFalse(mm.infer_as_probability)
            with self.assertRaises(UncalibratedWeightsError):
                mm.predict(np.zeros((8, 12), dtype=np.float32))


class SingleBookGrepTests(unittest.TestCase):
    def test_phase_sim_is_not_a_fill_book(self):
        src = Path("main.py").read_text(encoding="utf-8")
        self.assertIn("bind_ledger", src)
        self.assertIn("不入账", src)
        self.assertNotIn("self.ps.append", src)
        self.assertNotIn("self.closed.append", src)

    def test_paper_place_lives_on_ledger(self):
        ex = Path("modules/executor.py").read_text(encoding="utf-8")
        self.assertIn("from modules.ledger import PaperBroker", ex)
        led = Path("modules/ledger.py").read_text(encoding="utf-8")
        self.assertIn("CLOSE full size", led)
        self.assertIn("Never overwrite qty/side/entry without a CLOSE event", led)


if __name__ == "__main__":
    unittest.main()
