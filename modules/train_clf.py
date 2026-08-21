# modules/train_clf.py
"""Classification trainer + Guo-2017 temperature fit.

Uses FeatureBuilder's 12-d sequences. Does not invent a third feature schema.
Does not rewrite EnhancedTFT / EnhancedNBeats — only the objective:
    BCEWithLogits on direction labels, then T on a purged val split.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_definitions import EnhancedNBeats, EnhancedTFT
from modules.calibration import fit_temperature_from_logits
from modules.labels import (
    embargo_seconds_from_config,
    horizon_seconds_from_config,
    label_meta,
    time_holdout_purge_embargo,
    val_frac_from_config,
)

try:
    import joblib
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    joblib = None
    StandardScaler = None


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_feature_scaler(X_train: np.ndarray):
    if StandardScaler is None:
        raise RuntimeError("scikit-learn is required to fit scaler.pkl")
    xt = np.asarray(X_train, dtype=np.float32)
    if xt.ndim == 3:
        flat = xt.reshape(-1, xt.shape[-1])
    else:
        flat = xt
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    xt = np.asarray(X, dtype=np.float32)
    if scaler is None:
        return xt
    if xt.ndim == 3:
        t, f = xt.shape[1], xt.shape[2]
        out = scaler.transform(xt.reshape(-1, f)).reshape(xt.shape[0], t, f)
    else:
        out = scaler.transform(xt)
    return np.clip(np.asarray(out, dtype=np.float32), -10.0, 10.0)


def _train_one_model(model, optimizer, xb, yb, pool_time: bool, criterion, max_norm=1.0):
    optimizer.zero_grad()
    inp = xb.mean(dim=1) if pool_time else xb
    pred = model(inp)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred = pred.reshape(-1)
    yb = yb.reshape(-1)
    loss = criterion(pred, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    return float(loss.item()), pred.detach()


@torch.no_grad()
def _collect_logits(model, x, pool_time: bool, device):
    model.eval()
    inp = x.to(device)
    if pool_time and inp.ndim == 3:
        inp = inp.mean(dim=1)
    y = model(inp)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y.reshape(-1).detach().cpu()


def save_thresholds_temp(path: str, t_tft: float, t_nbt: float):
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r")) or {}
        except Exception:
            data = {}
    data["temp"] = {"tft": float(t_tft), "nbt": float(t_nbt)}
    # do not invent blend / runtime thresholds here (autotune owns those)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_model_meta(path: str, cfg: Optional[dict], extra: Optional[dict] = None):
    meta = label_meta(cfg)
    if extra:
        meta.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def train_classification(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    cfg: Optional[dict] = None,
    *,
    input_size: int = 12,
    batch_size: int = 32,
    epochs: int = 4,
    lr: float = 2e-5,
    device=None,
    tft_model=None,
    nbeats_model=None,
    scaler=None,
    save: bool = True,
    tft_path: str = "tft_model.pth",
    nbeats_path: str = "nbeats_model.pth",
    scaler_path: str = "scaler.pkl",
    thresholds_path: str = "thresholds.json",
    meta_path: str = "model_meta.json",
) -> Dict[str, Any]:
    """Train both heads with BCEWithLogits, fit T on purged val logits."""
    cfg = cfg or {}
    device = device or _device()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if X.ndim != 3:
        raise ValueError(f"X must be (N,T,F), got {X.shape}")
    if X.shape[0] != y.shape[0] or X.shape[0] != times.shape[0]:
        raise ValueError("X/y/times length mismatch")
    if X.shape[2] != input_size:
        # keep FeatureBuilder schema: pad / crop, never a third schema
        if X.shape[2] > input_size:
            X = X[:, :, :input_size]
        else:
            pad = np.zeros((X.shape[0], X.shape[1], input_size - X.shape[2]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=2)

    horizon_sec = horizon_seconds_from_config(cfg)
    embargo_sec = embargo_seconds_from_config(cfg, horizon_sec)
    val_frac = val_frac_from_config(cfg)
    train_idx, val_idx, split_info = time_holdout_purge_embargo(
        times, horizon_sec, val_frac=val_frac, embargo_sec=embargo_sec
    )

    if train_idx.size < 8:
        # not enough after purge: train on all but the last val fold without T
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx, assume_unique=False)
        split_info = dict(split_info)
        split_info["fallback"] = "purge emptied train; used pre-val candidates without extra embargo"
        if train_idx.size < 4:
            train_idx = np.arange(max(1, len(y) - max(1, len(val_idx))))
            val_idx = np.arange(train_idx.size, len(y))

    if scaler is None:
        scaler = fit_feature_scaler(X[train_idx])
    Xs = apply_scaler(X, scaler)

    if tft_model is None:
        tft_model = EnhancedTFT(input_size=input_size).to(device)
    else:
        tft_model.to(device)
    if nbeats_model is None:
        nbeats_model = EnhancedNBeats(input_size=input_size).to(device)
    else:
        nbeats_model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    tft_opt = torch.optim.Adam(tft_model.parameters(), lr=lr)
    nbt_opt = torch.optim.Adam(nbeats_model.parameters(), lr=lr)

    x_tr = torch.tensor(Xs[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
    ds = TensorDataset(x_tr, y_tr)
    dl = DataLoader(ds, batch_size=min(int(batch_size), max(1, len(ds))), shuffle=True)

    history = []
    tft_model.train()
    nbeats_model.train()
    for ep in range(int(epochs)):
        tft_sum = nbt_sum = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            lt, _ = _train_one_model(tft_model, tft_opt, xb, yb, False, criterion)
            ln, _ = _train_one_model(nbeats_model, nbt_opt, xb, yb, True, criterion)
            tft_sum += lt
            nbt_sum += ln
            n += 1
        history.append({
            "epoch": ep + 1,
            "tft_loss": tft_sum / max(1, n),
            "nbeats_loss": nbt_sum / max(1, n),
        })

    t_tft = t_nbt = 1.0
    calibrated = False
    if val_idx.size >= 8:
        x_va = torch.tensor(Xs[val_idx], dtype=torch.float32)
        y_va = torch.tensor(y[val_idx], dtype=torch.float32)
        lg_tft = _collect_logits(tft_model, x_va, False, device)
        lg_nbt = _collect_logits(nbeats_model, x_va, True, device)
        _, t_tft = fit_temperature_from_logits(lg_tft, y_va)
        _, t_nbt = fit_temperature_from_logits(lg_nbt, y_va)
        calibrated = True

    if save:
        torch.save(tft_model.state_dict(), tft_path)
        torch.save(nbeats_model.state_dict(), nbeats_path)
        if joblib is not None and scaler is not None:
            joblib.dump(scaler, scaler_path)
        save_thresholds_temp(thresholds_path, t_tft, t_nbt)
        save_model_meta(meta_path, cfg, extra={"split": split_info, "calibrated": calibrated})

    return {
        "tft": tft_model,
        "nbeats": nbeats_model,
        "scaler": scaler,
        "T_tft": t_tft,
        "T_nbt": t_nbt,
        "calibrated": calibrated,
        "split": split_info,
        "history": history,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
    }


def online_bce_step(
    tft_model,
    nbeats_model,
    tft_opt,
    nbt_opt,
    seqs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device,
) -> Tuple[float, float]:
    """One shuffled mini-batch pass (live trainer convenience)."""
    criterion = nn.BCEWithLogitsLoss()
    x = torch.tensor(np.asarray(seqs, dtype=np.float32), dtype=torch.float32).to(device)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32).to(device)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=min(int(batch_size), max(1, len(ds))), shuffle=True)
    tft_model.train()
    nbeats_model.train()
    tft_sum = nbt_sum = 0.0
    n = 0
    for xb, yb in dl:
        lt, _ = _train_one_model(tft_model, tft_opt, xb, yb, False, criterion)
        ln, _ = _train_one_model(nbeats_model, nbt_opt, xb, yb, True, criterion)
        tft_sum += lt
        nbt_sum += ln
        n += 1
    return tft_sum / max(1, n), nbt_sum / max(1, n)
