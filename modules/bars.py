"""Closed 1m bar clock for live infer.

Offline training (get_train_data) replays one FeatureBuilder row per 1m kline.
Live must do the same: infer only after a minute *closes*, and feed that candle
once. Forming / incomplete minutes are skipped. The live L2 book is not part
of this path — see features.build_from_kline / synth_kline_events.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.features import FeatureBuilder, build_from_kline


BAR_MS = 60_000


def minute_floor_ms(ts_ms: int) -> int:
    return int(ts_ms) - (int(ts_ms) % BAR_MS)


@dataclass(frozen=True)
class ClosedBar:
    open_ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def ohlcv(self) -> Tuple[float, float, float, float, float]:
        return self.open, self.high, self.low, self.close, self.volume


def parse_kline_message(msg: dict) -> Optional[dict]:
    """Normalize Binance / OKX / already-normalized kline WS payloads.

    Returns dict with keys: open_ts_ms, o, h, l, c, v, closed.
    Forming vs closed is the caller's responsibility (see ClosedBarClock).
    """
    if not isinstance(msg, dict):
        return None
    if msg.get("channel") == "kline" and "o" in msg and "closed" in msg:
        try:
            return {
                "open_ts_ms": int(msg.get("open_ts_ms") or msg.get("t") or 0),
                "o": float(msg["o"]),
                "h": float(msg["h"]),
                "l": float(msg["l"]),
                "c": float(msg["c"]),
                "v": float(msg.get("v") or msg.get("volume") or 0.0),
                "closed": bool(msg["closed"]),
            }
        except (TypeError, ValueError, KeyError):
            return None

    # Binance kline event
    k = msg.get("k")
    if isinstance(k, dict) and k.get("o") is not None:
        try:
            return {
                "open_ts_ms": int(k.get("t") or 0),
                "o": float(k["o"]),
                "h": float(k["h"]),
                "l": float(k["l"]),
                "c": float(k["c"]),
                "v": float(k.get("v") or 0.0),
                "closed": bool(k.get("x", False)),
            }
        except (TypeError, ValueError, KeyError):
            return None

    # OKX candle row already extracted: data[0] = [ts,o,h,l,c,vol,...,confirm]
    row = msg.get("candle") or msg.get("data")
    if isinstance(row, list) and row and not isinstance(row[0], list):
        try:
            confirm = str(row[8]) if len(row) > 8 else "0"
            return {
                "open_ts_ms": int(row[0]),
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
                "v": float(row[5]) if len(row) > 5 else 0.0,
                "closed": confirm in ("1", "true", "True"),
            }
        except (TypeError, ValueError, IndexError):
            return None
    if isinstance(row, list) and row and isinstance(row[0], list):
        return parse_kline_message({"candle": row[0]})
    return None


class ClosedBarClock:
    """Emit each 1m bar *once*, and only after it is confirmed closed.

    Forming (incomplete) minutes never leave this clock. Duplicate closes of
    the same open_ts are ignored so FeatureBuilder sees one kline → one row.
    """

    def __init__(self):
        self._emitted: set = set()
        self._forming: Optional[dict] = None

    def already_emitted(self, open_ts_ms: int) -> bool:
        return minute_floor_ms(open_ts_ms) in self._emitted

    def mark_emitted(self, open_ts_ms: int) -> None:
        self._emitted.add(minute_floor_ms(open_ts_ms))

    def ingest_kline(
        self,
        open_ts_ms: int,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        closed: bool,
    ) -> Optional[ClosedBar]:
        if not closed:
            return None
        try:
            o = float(o); h = float(h); l = float(l); c = float(c); v = float(v)
        except (TypeError, ValueError):
            return None
        if c <= 0:
            return None
        key = minute_floor_ms(open_ts_ms)
        if key in self._emitted:
            return None
        self._emitted.add(key)
        return ClosedBar(open_ts_ms=key, open=o, high=h, low=l, close=c, volume=max(v, 0.0))

    def ingest_kline_msg(self, msg: dict) -> Optional[ClosedBar]:
        parsed = parse_kline_message(msg) if msg else None
        if not parsed:
            return None
        return self.ingest_kline(
            parsed["open_ts_ms"],
            parsed["o"], parsed["h"], parsed["l"], parsed["c"], parsed["v"],
            parsed["closed"],
        )

    def ingest_trade(self, ts_ms: int, price: float, qty: float = 0.0) -> Optional[ClosedBar]:
        """Aggregate prints. Emit the previous minute when a trade opens a new one."""
        try:
            p = float(price); q = abs(float(qty)); t = int(ts_ms)
        except (TypeError, ValueError):
            return None
        if p <= 0 or t <= 0:
            return None
        minute = minute_floor_ms(t)
        if self._forming is None:
            self._forming = {"open_ts_ms": minute, "o": p, "h": p, "l": p, "c": p, "v": q}
            return None
        if minute == self._forming["open_ts_ms"]:
            f = self._forming
            f["h"] = max(f["h"], p)
            f["l"] = min(f["l"], p)
            f["c"] = p
            f["v"] = float(f["v"]) + q
            return None
        # New minute: the forming bar is now closed. Do not emit the new one.
        prev = self._forming
        self._forming = {"open_ts_ms": minute, "o": p, "h": p, "l": p, "c": p, "v": q}
        return self.ingest_kline(
            prev["open_ts_ms"], prev["o"], prev["h"], prev["l"], prev["c"], prev["v"],
            closed=True,
        )


def replay_closed_klines(
    klines: Sequence[Any],
    fb: Optional[FeatureBuilder] = None,
    seq_len: int = 30,
    clock: Optional[ClosedBarClock] = None,
) -> List[Any]:
    """Feed already-closed 1m klines through FeatureBuilder (offline-identical).

    Each item is a ClosedBar, a dict with o/h/l/c/v, or (o,h,l,c,v) / (ts,o,h,l,c,v).
    Forming bars (if a clock is given and they arrive as closed=False) are skipped.
    """
    fb = fb or FeatureBuilder(seq_len=seq_len, k_levels=3)
    clock = clock if clock is not None else ClosedBarClock()
    seqs = []
    for item in klines:
        bar = _coerce_closed_bar(item)
        if bar is None:
            continue
        emitted = clock.ingest_kline(bar.open_ts_ms, bar.open, bar.high, bar.low, bar.close, bar.volume, closed=True)
        if emitted is None:
            continue
        seq = build_from_kline(fb, emitted.open, emitted.high, emitted.low, emitted.close, emitted.volume)
        if seq is not None:
            seqs.append(seq)
    return seqs


def _coerce_closed_bar(item: Any) -> Optional[ClosedBar]:
    if isinstance(item, ClosedBar):
        return item
    if isinstance(item, dict):
        try:
            ts = int(item.get("open_ts_ms") or item.get("time") or item.get("t") or 0)
            return ClosedBar(
                open_ts_ms=minute_floor_ms(ts) if ts else 0,
                open=float(item.get("open", item.get("o"))),
                high=float(item.get("high", item.get("h"))),
                low=float(item.get("low", item.get("l"))),
                close=float(item.get("close", item.get("c"))),
                volume=float(item.get("volume", item.get("v", 1.0))),
            )
        except (TypeError, ValueError, KeyError):
            return None
    if isinstance(item, (tuple, list)):
        try:
            if len(item) >= 6:
                ts, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
                return ClosedBar(minute_floor_ms(int(ts)), float(o), float(h), float(l), float(c), float(v))
            if len(item) == 5:
                o, h, l, c, v = item
                return ClosedBar(0, float(o), float(h), float(l), float(c), float(v))
        except (TypeError, ValueError):
            return None
    return None


def fetch_recent_closed_klines(cfg: dict, symbol: str, limit: int = 80) -> List[ClosedBar]:
    """REST warmup: last *closed* 1m candles. Drops the forming minute. Best-effort."""
    import requests

    market = (cfg or {}).get("market") or {}
    provider = str(market.get("provider", "okx")).lower()
    symbol_map = market.get("symbol_map") or {}
    sym = str(symbol).lower()
    inst = symbol_map.get(sym) or symbol_map.get(symbol) or (
        f"{symbol[:-4].upper()}-USDT-SWAP" if provider == "okx" and sym.endswith("usdt") else str(symbol).upper()
    )
    now_ms = int(time.time() * 1000)
    cur_minute = minute_floor_ms(now_ms)
    out: List[ClosedBar] = []
    try:
        if provider == "okx":
            url = "https://www.okx.com/api/v5/market/candles"
            r = requests.get(url, params={"instId": inst, "bar": "1m", "limit": str(int(limit))}, timeout=15)
            rows = (r.json() or {}).get("data") or []
            for row in rows:
                confirm = str(row[8]) if len(row) > 8 else "0"
                ts = int(row[0])
                if confirm not in ("1", "true", "True"):
                    continue
                if minute_floor_ms(ts) >= cur_minute:
                    continue
                out.append(ClosedBar(
                    open_ts_ms=minute_floor_ms(ts),
                    open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]),
                ))
        else:
            url = "https://api.binance.com/api/v3/klines"
            r = requests.get(
                url,
                params={"symbol": str(inst).upper(), "interval": "1m", "limit": int(limit)},
                timeout=15,
            )
            for row in r.json() or []:
                ts = int(row[0])
                close_time = int(row[6]) if len(row) > 6 else ts + BAR_MS - 1
                if close_time >= now_ms or minute_floor_ms(ts) >= cur_minute:
                    continue
                out.append(ClosedBar(
                    open_ts_ms=minute_floor_ms(ts),
                    open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]),
                ))
    except Exception:
        return []
    out.sort(key=lambda b: b.open_ts_ms)
    # unique by open_ts
    uniq: Dict[int, ClosedBar] = {}
    for b in out:
        uniq[b.open_ts_ms] = b
    return [uniq[k] for k in sorted(uniq)]
