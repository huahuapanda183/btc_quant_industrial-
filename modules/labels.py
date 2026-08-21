# modules/labels.py
"""Clock-horizon direction labels for the classification head.

This is the only label definition used by training. Inference must treat the
model output as a *logit* for the same event, not as a predicted return.

Label (binary, default):
    y = 1 if r_{t→t+h} >  κ
    y = 0 if r_{t→t+h} < -κ
    y = None (dropped) if |r| ≤ κ   when drop_deadzone is true

h is *clock time* (label.horizon_minutes), not N trades.
There is no live `future_shift` / trade-count label path. If a caller
still has that key in a leftover config, it is ignored.
κ defaults to a round-trip cost:
    κ = 2 * (taker_fee_bp + slippage_bp) / 1e4  +  label.dead_zone

15 / 20 / 30 / 60 minute horizons are sweepable via config. The default is a
placeholder in the 15–30m band, not a claimed optimum.

Walk-forward split (used when fitting temperature):
    - Time-held-out: last val_frac of labeled samples, ordered by t.
    - Purge: drop a train sample if its label window [t, t+h] overlaps val.
    - Embargo: extra gap (default = h) before the first validation timestamp.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_cfg(cfg: Optional[dict]) -> dict:
    return cfg or {}


def logits_to_calibrated_prob(z, T: float = 1.0) -> float:
    """Guo 2017: p = σ(z / T) on a *classification* logit, not a return residual."""
    z = float(np.clip(float(z) / max(1e-6, float(T)), -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def horizon_minutes_from_config(cfg: Optional[dict], default: float = 20.0) -> float:
    """Clock minutes only. Never reads a trade-count `future_shift`."""
    cfg = _as_cfg(cfg)
    label = cfg.get("label") or {}
    if "horizon_minutes" in label and label.get("horizon_minutes") is not None:
        return float(label["horizon_minutes"])
    # leftover top-level future_shift is dead — do not train on N-trade shift
    return float(default)


def horizon_seconds_from_config(cfg: Optional[dict], default: float = 20.0) -> float:
    return horizon_minutes_from_config(cfg, default=default) * 60.0


def kappa_from_config(cfg: Optional[dict]) -> float:
    """κ ≈ fees + slippage (+ optional extra dead zone), in return units."""
    cfg = _as_cfg(cfg)
    trading = cfg.get("trading") or {}
    label = cfg.get("label") or {}
    if "kappa" in label and label.get("kappa") is not None:
        return float(label["kappa"])
    taker_bp = float(trading.get("taker_fee_bp", 2.0))
    slip_bp = float(trading.get("slippage_bp", 3.0))
    round_trip = 2.0 * (taker_bp + slip_bp) / 10000.0
    extra = float(label.get("dead_zone", 0.0) or 0.0)
    return float(round_trip + extra)


def drop_deadzone_from_config(cfg: Optional[dict]) -> bool:
    label = (_as_cfg(cfg).get("label") or {})
    return bool(label.get("drop_deadzone", True))


def val_frac_from_config(cfg: Optional[dict], default: float = 0.2) -> float:
    label = (_as_cfg(cfg).get("label") or {})
    return float(label.get("val_frac", default))


def embargo_seconds_from_config(cfg: Optional[dict], horizon_sec: float) -> float:
    label = (_as_cfg(cfg).get("label") or {})
    if label.get("embargo_minutes") is not None:
        return float(label["embargo_minutes"]) * 60.0
    return float(horizon_sec)


def future_return_at_horizon(
    timestamps: Sequence[float],
    prices: Sequence[float],
    index: int,
    horizon_sec: float,
) -> Optional[float]:
    """r_{t→t+h} using the first print at or after t+h. None if the path ends first."""
    ts = np.asarray(timestamps, dtype=float)
    px = np.asarray(prices, dtype=float)
    if index < 0 or index >= len(ts):
        return None
    p0 = float(px[index])
    if p0 <= 0:
        return None
    t_target = float(ts[index]) + float(horizon_sec)
    # search forward; callers usually pass chronological arrays
    j = int(index + 1)
    while j < len(ts) and ts[j] < t_target:
        j += 1
    if j >= len(ts):
        return None
    p1 = float(px[j])
    if p1 <= 0:
        return None
    return (p1 - p0) / p0


def price_at_or_after(tape: Sequence[Tuple[float, float]], t_target: float) -> Optional[float]:
    """First price on a (ts, px) tape with ts >= t_target."""
    for ts, px in tape:
        if ts >= t_target and px > 0:
            return float(px)
    return None


def direction_label(ret: float, kappa: float, drop_deadzone: bool = True) -> Optional[int]:
    """Map a horizon return to {1, 0} or None (dead zone)."""
    r = float(ret)
    k = max(0.0, float(kappa))
    if r > k:
        return 1
    if r < -k:
        return 0
    if drop_deadzone:
        return None
    return 0


def build_direction_labels(
    timestamps: Sequence[float],
    prices: Sequence[float],
    horizon_minutes: float,
    kappa: float,
    drop_deadzone: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label every index that has a complete clock-horizon future.

    Returns (indices, labels, returns) as 1-d arrays. Dropped dead-zone rows
    are omitted when drop_deadzone is True.
    """
    ts = np.asarray(timestamps, dtype=float)
    px = np.asarray(prices, dtype=float)
    if ts.ndim != 1 or px.ndim != 1 or len(ts) != len(px):
        raise ValueError("timestamps and prices must be 1-d and aligned")
    horizon_sec = float(horizon_minutes) * 60.0
    idxs: List[int] = []
    ys: List[int] = []
    rs: List[float] = []
    for i in range(len(ts)):
        r = future_return_at_horizon(ts, px, i, horizon_sec)
        if r is None:
            continue
        y = direction_label(r, kappa, drop_deadzone=drop_deadzone)
        if y is None:
            continue
        idxs.append(i)
        ys.append(int(y))
        rs.append(float(r))
    return (
        np.asarray(idxs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(rs, dtype=np.float64),
    )


def time_holdout_purge_embargo(
    sample_times: Sequence[float],
    horizon_sec: float,
    val_frac: float = 0.2,
    embargo_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Time-held-out split with purge + embargo.

    A train sample at t uses prices on [t, t+h]. It is purged if that window
    reaches the first validation timestamp. An embargo of `embargo_sec`
    (default h) is added: keep train only if t + h + embargo <= t_val_start.
    """
    ts = np.asarray(sample_times, dtype=float).ravel()
    n = int(ts.size)
    if n < 2:
        all_idx = np.arange(n, dtype=np.int64)
        return all_idx, np.asarray([], dtype=np.int64), {
            "n_train": n,
            "n_val": 0,
            "n_purged": 0,
            "t_val_start": None,
            "horizon_sec": float(horizon_sec),
            "embargo_sec": float(horizon_sec if embargo_sec is None else embargo_sec),
            "note": "too few samples for a hold-out; caller should skip calibration",
        }

    order = np.argsort(ts, kind="mergesort")
    n_val = max(1, int(round(n * float(val_frac))))
    n_val = min(n_val, n - 1)
    val_ids = order[-n_val:]
    train_cand = order[:-n_val]
    t_val_start = float(ts[val_ids].min())
    embargo = float(horizon_sec if embargo_sec is None else embargo_sec)
    cutoff = t_val_start - float(horizon_sec) - embargo
    keep_mask = ts[train_cand] <= cutoff
    train_ids = train_cand[keep_mask]
    n_purged = int((~keep_mask).sum())

    info = {
        "n_train": int(train_ids.size),
        "n_val": int(val_ids.size),
        "n_purged": n_purged,
        "n_train_candidates": int(train_cand.size),
        "t_val_start": t_val_start,
        "cutoff_train": cutoff,
        "horizon_sec": float(horizon_sec),
        "embargo_sec": embargo,
        "val_frac": float(val_frac),
        "note": (
            "time-held-out last val_frac; train kept iff t + horizon + embargo "
            "<= first validation timestamp (purge + embargo)"
        ),
    }
    return train_ids.astype(np.int64), val_ids.astype(np.int64), info


def label_meta(cfg: Optional[dict]) -> Dict[str, Any]:
    horizon_m = horizon_minutes_from_config(cfg)
    return {
        "head": "classification",
        "objective": "direction_bce",
        "horizon_minutes": horizon_m,
        "kappa": kappa_from_config(cfg),
        "drop_deadzone": drop_deadzone_from_config(cfg),
        "val_frac": val_frac_from_config(cfg),
        "prob_meaning": (
            f"P(r_{{t→t+{horizon_m:.0f}m}} > κ) after temperature scaling; "
            "not a predicted return and not an uncalibrated sigmoid(MSE residual)"
        ),
        "split": (
            "walk-forward time hold-out on labeled sample time; purge overlapping "
            "label windows; embargo default = horizon"
        ),
    }
