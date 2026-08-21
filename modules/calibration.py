"""Guo 2017 temperature scaling for *classification logits*.

Temperature T is a single scalar that divides logits before the sigmoid/softmax.
It must be fit on a time-held-out validation segment, not the train window,
and must never be applied to a return-regression head.

train_models.py previously ignored this module and trained MSE(return).
The classification trainer reuses TemperatureScaler / fit_temperature_from_logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        init_T = max(float(init_T), 1e-6)
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32)))

    def forward(self, logits):
        T = torch.exp(self.logT) + 1e-8
        return logits / T

    def temperature(self) -> float:
        return float(torch.exp(self.logT).detach().cpu().item())


def _align_logits_labels(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.reshape(-1).float()
    labels = labels.reshape(-1).float()
    if logits.numel() != labels.numel():
        raise ValueError(f"logits/labels size mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    return logits, labels


def fit_temperature_from_logits(logits, labels, device=None, max_iter: int = 50):
    """Fit T on cached classification logits (Guo 2017, BCEWithLogits).

    Returns (scaler, T) with T clamped to [0.05, 10] for numerical sanity.
    """
    if device is None:
        device = torch.device("cpu")
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.float32)
    logits, labels = _align_logits_labels(logits, labels)
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler().to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=int(max_iter))

    def closure():
        opt.zero_grad()
        loss = crit(scaler(logits), labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass

    T = float(min(10.0, max(0.05, scaler.temperature())))
    with torch.no_grad():
        scaler.logT.copy_(torch.log(torch.tensor([T], device=scaler.logT.device)))
    return scaler, T


def collect_model_logits(model, val_loader, device, pool_time: bool = False):
    """Run a classifier on a loader and return (logits, labels) tensors on CPU."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            if pool_time and xb.ndim == 3:
                xb = xb.mean(dim=1)
            lg = model(xb)
            if isinstance(lg, (tuple, list)):
                lg = lg[0]
            logits.append(lg.detach().cpu())
            labels.append(yb.detach().cpu())
    if not logits:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(logits, 0), torch.cat(labels, 0)


def fit_temperature(model, val_loader, device, pool_time: bool = False):
    """Original helper: collect logits from `model(xb)` then fit T."""
    logits, labels = collect_model_logits(model, val_loader, device, pool_time=pool_time)
    if logits.numel() == 0:
        return TemperatureScaler().to(device)
    scaler, _ = fit_temperature_from_logits(logits, labels, device=device)
    return scaler
