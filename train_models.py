"""Classification trainer (live stream or --offline).

This is the only trainer that may write scaler.pkl / tft_model.pth /
nbeats_model.pth / thresholds.json temperature. get_train_data.py builds a
FeatureBuilder-aligned dataset and must not poison those artifacts.

Objective: BCEWithLogits on clock-horizon direction labels, then Guo 2017
temperature on a purged time-held-out validation segment.
"""
import argparse
import asyncio
import json
import os
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import torch.optim as optim
import yaml

from model_definitions import EnhancedNBeats, EnhancedTFT
from modules.features import FeatureBuilder
from modules.labels import (
    direction_label,
    horizon_minutes_from_config,
    kappa_from_config,
    label_meta,
    price_at_or_after,
)
from modules.train_clf import online_bce_step, train_classification

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f) or {}

market = (config.get("market") or {})
provider = str(market.get("provider", "okx")).lower()
symbol = config.get("symbol", "BTCUSDT")
seq_len = int(config.get("seq_len", 30))
input_size = int(config.get("input_size", 12))
batch_size = int(config.get("batch_size", 32))
save_interval = int(config.get("save_interval", 50))
horizon_minutes = horizon_minutes_from_config(config)
horizon_sec = horizon_minutes * 60.0
kappa = kappa_from_config(config)
drop_deadzone = bool((config.get("label") or {}).get("drop_deadzone", True))


def resolve_symbol_for_provider(sym: str):
    m = (market.get("symbol_map") or {})
    if provider == "okx":
        if sym.lower() in m:
            return str(m[sym.lower()])
        if sym in m:
            return str(m[sym])
        if sym.lower().endswith("usdt"):
            return f"{sym[:-4].upper()}-USDT-SWAP"
    return sym.lower()


symbol_resolved = resolve_symbol_for_provider(str(symbol))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tft_model = EnhancedTFT(input_size=input_size).to(device)
nbeats_model = EnhancedNBeats(input_size=input_size).to(device)
tft_optimizer = optim.Adam(tft_model.parameters(), lr=2e-5)
nbeats_optimizer = optim.Adam(nbeats_model.parameters(), lr=2e-5)

latest_depth_evt = None
trade_queue = None
feature_builder = FeatureBuilder(seq_len=seq_len, k_levels=3)
scaler_path = "scaler.pkl"
scaler = None

seq_buffer = deque()  # (ts, seq, price)
price_tape = deque()  # (ts, price)
X_buffer = []
y_buffer = []
t_buffer = []


def _align_seq(seq: np.ndarray) -> np.ndarray:
    if seq.shape[1] > input_size:
        return seq[:, :input_size]
    if seq.shape[1] < input_size:
        pad = np.zeros((seq.shape[0], input_size - seq.shape[1]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=1)
    return seq


def _maybe_label():
    """Promote pending sequences whose clock horizon has elapsed."""
    if not price_tape:
        return
    now_ts = price_tape[-1][0]
    while seq_buffer and now_ts >= seq_buffer[0][0] + horizon_sec:
        t0, seq_old, p_old = seq_buffer.popleft()
        p_new = price_at_or_after(price_tape, t0 + horizon_sec)
        if p_old is None or p_old <= 0 or p_new is None or p_new <= 0:
            continue
        r = (p_new - p_old) / p_old
        y = direction_label(r, kappa, drop_deadzone=drop_deadzone)
        if y is None:
            continue
        X_buffer.append(seq_old.astype(np.float32))
        y_buffer.append(float(y))
        t_buffer.append(float(t0))


async def trade_handler_okx():
    import websockets
    url = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(url, ping_interval=20) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "trades", "instId": symbol_resolved}]}))
        async for msg in ws:
            j = json.loads(msg)
            if "data" not in j:
                continue
            row = (j.get("data") or [None])[0]
            if not row:
                continue
            px = float(row.get("px", 0) or 0)
            if px <= 0:
                continue
            ts = float(row.get("ts") or 0) / 1000.0
            if ts <= 0:
                ts = datetime.now(timezone.utc).timestamp()
            t_evt = {"p": str(px), "q": str(row.get("sz", "0")), "m": False, "ts": ts}
            await trade_queue.put(t_evt)


async def depth_handler_okx():
    global latest_depth_evt
    import websockets
    url = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(url, ping_interval=20) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "books5", "instId": symbol_resolved}]}))
        async for msg in ws:
            j = json.loads(msg)
            if "data" not in j:
                continue
            row = (j.get("data") or [None])[0]
            if not row:
                continue
            bids = row.get("bids", [])
            asks = row.get("asks", [])
            latest_depth_evt = {"b": [[b[0], b[1]] for b in bids], "a": [[a[0], a[1]] for a in asks]}


async def training_loop():
    global scaler
    steps = 0
    warmup_feats = []
    meta = label_meta(config)
    print(
        f"📡 等待数据流… provider={provider} symbol={symbol_resolved} "
        f"horizon={horizon_minutes}m κ={kappa:.6f} (clock labels, BCE)"
    )
    print(f"   split: {meta['split']}")
    while True:
        trade_evt = await trade_queue.get()
        seq = feature_builder.build(trade_evt, latest_depth_evt)
        if seq is None:
            continue
        seq = _align_seq(seq)

        last_feat = seq[-1]
        if scaler is None:
            warmup_feats.append(last_feat)
            if len(warmup_feats) >= 100:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(np.array(warmup_feats, dtype=np.float32))
                joblib.dump(scaler, scaler_path)
                print(f"💾 已生成 FeatureBuilder-12d scaler.pkl ({len(warmup_feats)} 条特征)")
            else:
                continue

        seq_s = scaler.transform(seq)
        seq_s = np.clip(seq_s, -10.0, 10.0).astype(np.float32)

        ts = float(trade_evt.get("ts") or datetime.now(timezone.utc).timestamp())
        price = float(trade_evt.get("p", 0) or 0)
        price_tape.append((ts, price))
        cutoff = ts - 3.0 * horizon_sec
        while price_tape and price_tape[0][0] < cutoff:
            price_tape.popleft()
        seq_buffer.append((ts, seq_s, price))
        _maybe_label()

        if len(y_buffer) >= batch_size:
            tft_loss, nbeats_loss = online_bce_step(
                tft_model, nbeats_model, tft_optimizer, nbeats_optimizer,
                np.array(X_buffer[-batch_size:]), np.array(y_buffer[-batch_size:]),
                batch_size, device,
            )
            steps += 1
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"✅ {now} 分类训练 | Step: {steps} | TFT BCE: {tft_loss:.6f} | "
                f"NBeats BCE: {nbeats_loss:.6f} | labeled={len(y_buffer)}"
            )
            if steps % save_interval == 0:
                # artifacts: walk-forward + temperature on held-out (not the train window)
                result = train_classification(
                    np.array(X_buffer, dtype=np.float32),
                    np.array(y_buffer, dtype=np.float32),
                    np.array(t_buffer, dtype=np.float64),
                    cfg=config,
                    input_size=input_size,
                    batch_size=batch_size,
                    epochs=2,
                    device=device,
                    tft_model=tft_model,
                    nbeats_model=nbeats_model,
                    scaler=scaler,
                    save=True,
                )
                print(
                    f"💾 walk-forward save | train={result['n_train']} val={result['n_val']} "
                    f"purged={result['split'].get('n_purged')} "
                    f"T_tft={result['T_tft']:.3f} T_nbt={result['T_nbt']:.3f} "
                    f"calibrated={result['calibrated']}"
                )


async def status_monitor():
    while True:
        qn = trade_queue.qsize()
        if latest_depth_evt is None:
            print("⏳ 等待 books5 盘口数据...")
        elif qn == 0:
            print("⏳ 等待 trades 成交数据...")
        else:
            print(
                f"✅ 数据流正常，queue={qn}，已生成分类样本={len(y_buffer)} "
                f"(需满 {horizon_minutes:.0f} 分钟时钟窗口才出首个标签)"
            )
        await asyncio.sleep(3)


async def live_main():
    global trade_queue
    if provider != "okx":
        raise SystemExit("当前训练脚本仅启用 okx provider")
    trade_queue = asyncio.Queue(maxsize=5000)
    print(f"🚀 实时分类训练启动 provider={provider}")
    await asyncio.gather(trade_handler_okx(), depth_handler_okx(), training_loop(), status_monitor())


def offline_main(npz_path: str, epochs: int):
    if not os.path.exists(npz_path):
        from get_train_data import build_offline_dataset, write_offline_dataset
        print(f"📦 {npz_path} 不存在，先用 FeatureBuilder 回放 K 线构建数据集…")
        ds = build_offline_dataset(config)
        write_offline_dataset(ds, npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    ts = data["ts"]
    print(f"📂 offline dataset {npz_path}: N={len(y)} T={X.shape[1]} F={X.shape[2]}")
    result = train_classification(
        X, y, ts, cfg=config,
        input_size=input_size,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        save=True,
    )
    print(
        f"✅ offline 分类训练完成 | train={result['n_train']} val={result['n_val']} "
        f"purged={result['split'].get('n_purged')} "
        f"T_tft={result['T_tft']:.3f} T_nbt={result['T_nbt']:.3f} "
        f"calibrated={result['calibrated']}"
    )
    print(f"   split note: {result['split'].get('note')}")


def main():
    parser = argparse.ArgumentParser(description="BTC classification trainer")
    parser.add_argument("--offline", nargs="?", const="train_data.npz", default=None,
                        help="Train from FeatureBuilder npz (default train_data.npz). Does not use get_train_data's old 7-d scaler.")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.offline:
        offline_main(args.offline, args.epochs)
        return
    asyncio.run(live_main())


if __name__ == "__main__":
    main()
