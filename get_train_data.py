"""Offline FeatureBuilder dataset builder — *not* a second trainer.

Builds the same 12-d FeatureBuilder schema used at infer, plus clock-horizon
direction labels. Writes train_data.npz (and a wide CSV for inspection).

Does NOT write scaler.pkl / tft_model.pth / nbeats_model.pth. Those artifacts
are owned by train_models.py so this script cannot poison a 7-d kline scaler
into the live 12-d infer path.

Usage:
    python get_train_data.py
    python train_models.py --offline train_data.npz
"""
import numpy as np
import pandas as pd
import requests
import time
import yaml

from modules.features import FeatureBuilder
from modules.labels import (
    build_direction_labels,
    horizon_minutes_from_config,
    kappa_from_config,
    label_meta,
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

market = config.get("market", {}) or {}
provider = str(market.get("provider", "okx")).lower()
symbol = config.get("symbol", "BTCUSDT")
interval = config.get("interval", "1m")
lookback_hours = int(config.get("lookback_hours", 48))
symbol_map = market.get("symbol_map", {}) or {}
seq_len = int(config.get("seq_len", 30))
input_size = int(config.get("input_size", 12))


def fetch_klines_binance(sym: str):
    print("📥 从 Binance 拉取历史K线...")
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_hours * 60 * 60 * 1000
    data = []
    while start_time < end_time:
        params = {"symbol": sym.upper(), "interval": interval, "startTime": start_time, "endTime": end_time, "limit": 1000}
        batch = requests.get(url, params=params, timeout=15).json()
        if not batch:
            break
        data.extend(batch)
        start_time = int(batch[-1][0]) + 1
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def fetch_klines_okx(inst_id: str):
    print("📥 从 OKX 拉取历史K线...")
    bar_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H"}
    bar = bar_map.get(interval, "1m")
    url = "https://www.okx.com/api/v5/market/history-candles"
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - lookback_hours * 60 * 60 * 1000

    rows = []
    after = None
    for _ in range(120):
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)
        r = requests.get(url, params=params, timeout=15).json()
        data = (r or {}).get("data", [])
        if not data:
            break
        rows.extend(data)
        oldest = int(data[-1][0])
        if oldest <= start_ts:
            break
        after = oldest

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def resolve_symbol():
    s = str(symbol)
    if provider == "okx":
        if s.lower() in symbol_map:
            return str(symbol_map[s.lower()])
        if s in symbol_map:
            return str(symbol_map[s])
        if s.lower().endswith("usdt"):
            base = s[:-4].upper()
            return f"{base}-USDT-SWAP"
    return s.upper()


def _synth_events(o, h, l, c, v):
    spread = max((h - l) * 0.05, c * 0.00015)
    bid = c - spread / 2
    ask = c + spread / 2
    depth_evt = {"b": [[str(bid), str(max(v * 0.5, 1.0))]], "a": [[str(ask), str(max(v * 0.5, 1.0))]]}
    trade_evt = {"p": str(c), "q": str(max(v * 0.1, 1.0)), "m": False}
    return trade_evt, depth_evt


def build_offline_dataset(cfg: dict = None, df: pd.DataFrame = None) -> dict:
    """Replay klines through FeatureBuilder; label with clock-horizon direction."""
    cfg = cfg or config
    if df is None:
        resolved = resolve_symbol()
        if provider == "okx":
            df = fetch_klines_okx(resolved)
        else:
            df = fetch_klines_binance(resolved)
        if df is None or df.empty:
            raise SystemExit("❌ 未拉取到K线数据")
    else:
        resolved = "fixture"

    fb = FeatureBuilder(seq_len=int(cfg.get("seq_len", seq_len)), k_levels=3)
    horizon_m = horizon_minutes_from_config(cfg)
    kappa = kappa_from_config(cfg)
    drop_dz = bool((cfg.get("label") or {}).get("drop_deadzone", True))

    seqs = []
    ts_list = []
    px_list = []
    for _, row in df.iterrows():
        c = float(row["close"])
        o = float(row.get("open", c))
        h = float(row.get("high", c))
        l = float(row.get("low", c))
        v = float(row.get("volume", 1.0))
        trade_evt, depth_evt = _synth_events(o, h, l, c, v)
        seq = fb.build(trade_evt, depth_evt)
        t = row["time"]
        ts = t.timestamp() if hasattr(t, "timestamp") else float(t)
        if seq is None:
            continue
        if seq.shape[1] > input_size:
            seq = seq[:, :input_size]
        elif seq.shape[1] < input_size:
            pad = np.zeros((seq.shape[0], input_size - seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=1)
        seqs.append(seq.astype(np.float32))
        ts_list.append(ts)
        px_list.append(c)

    if not seqs:
        raise SystemExit("❌ FeatureBuilder 未产出任何序列（K线过短？）")

    idxs, ys, rs = build_direction_labels(ts_list, px_list, horizon_m, kappa, drop_deadzone=drop_dz)
    X = np.stack([seqs[i] for i in idxs], axis=0) if len(idxs) else np.zeros((0, seq_len, input_size), dtype=np.float32)
    meta = label_meta(cfg)
    meta.update({
        "n_seq": len(seqs),
        "n_labeled": int(len(ys)),
        "provider": provider,
        "symbol": resolved,
        "interval": interval,
        "note": "FeatureBuilder 12-d; clock-horizon classification; does not write scaler.pkl",
    })
    return {
        "X": X,
        "y": ys.astype(np.float32),
        "ts": np.asarray(ts_list, dtype=np.float64)[idxs] if len(idxs) else np.zeros((0,), dtype=np.float64),
        "ret": rs.astype(np.float32),
        "meta": meta,
        "last_row": np.stack([seqs[i][-1] for i in idxs], axis=0) if len(idxs) else np.zeros((0, input_size), dtype=np.float32),
    }


def write_offline_dataset(ds: dict, npz_path: str = "train_data.npz", csv_path: str = "train_data.csv"):
    np.savez_compressed(
        npz_path,
        X=ds["X"],
        y=ds["y"],
        ts=ds["ts"],
        ret=ds["ret"],
        last_row=ds["last_row"],
        meta=np.array([ds["meta"]], dtype=object),
    )
    # inspection CSV: last FeatureBuilder row + label (not a 7-d kline schema)
    cols = [
        "close", "ret", "dH", "macdH", "macd", "rsi",
        "vol_abs", "imb", "spread_prop", "micro_bias", "buy_dom", "vwap_dev",
    ]
    if ds["last_row"].size:
        df = pd.DataFrame(ds["last_row"], columns=cols[: ds["last_row"].shape[1]])
        df["label"] = ds["y"]
        df["horizon_return"] = ds["ret"]
        df["ts"] = ds["ts"]
        df.to_csv(csv_path, index=False)
    print(
        f"✅ 已生成 {npz_path} / {csv_path} | labeled={len(ds['y'])} | "
        f"horizon={ds['meta'].get('horizon_minutes')}m κ={ds['meta'].get('kappa')} | "
        f"未写入 scaler.pkl（由 train_models.py 拟合 12-d scaler）"
    )


def main():
    ds = build_offline_dataset(config)
    write_offline_dataset(ds)
    print("下一步: python train_models.py --offline train_data.npz")


if __name__ == "__main__":
    main()
