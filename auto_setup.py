"""
Auto-fetch setup (Bar 1/2 + SQZMOM 1/2) untuk ticker+date+hour
dengan menjalankan sqzmom_export.py sebagai subprocess lalu
membaca xlsx hasilnya.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone, date as date_cls
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# Lokasi sqzmom_export.py — naik 2 level dari folder v4 ke root project
SQZMOM_EXPORT_PATH = Path(__file__).resolve().parents[2] / "sqzmom_export.py"


def _yahoo_to_binance(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith("-USD"):
        return t.replace("-USD", "") + "USDT"
    if t.endswith("USDT"):
        return t
    return t + "USDT"


def fetch_setup(
    ticker: str,
    target_date: date_cls,
    target_hour: int,
    interval: str = "1h",
    warmup_days: int = 90,
) -> Dict[str, object]:
    """Return dict with Bar 1/2 + SQZMOM 1/2 fields (or 'error' key).

    Calls sqzmom_export.py as subprocess (idle ~5-10s for Binance fetch).
    """
    if not SQZMOM_EXPORT_PATH.exists():
        return {"error": f"sqzmom_export.py tidak ditemukan di {SQZMOM_EXPORT_PATH}"}

    symbol = _yahoo_to_binance(ticker)
    target_dt_utc = datetime(target_date.year, target_date.month, target_date.day,
                             target_hour, 0, tzinfo=timezone.utc)
    target_dt_naive = target_dt_utc.replace(tzinfo=None)
    fetch_start = (target_dt_utc - timedelta(days=warmup_days)).strftime("%Y-%m-%dT%H:%M")
    fetch_end = (target_dt_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{symbol}_{interval}_setup.xlsx"
        cmd = [
            sys.executable, str(SQZMOM_EXPORT_PATH),
            "--symbol", symbol,
            "--interval", interval,
            "--start", fetch_start,
            "--end", fetch_end,
            "--out", str(out_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"error": "Timeout saat fetch data (>120s)"}
        if res.returncode != 0 or not out_path.exists():
            return {"error": f"sqzmom_export gagal: {res.stderr[-400:] or res.stdout[-400:]}"}

        df = pd.read_excel(out_path)

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)").reset_index(drop=True)

    cur = df[df["Datetime (UTC)"] == target_dt_naive]
    if cur.empty:
        return {"error": f"Bar untuk {target_dt_utc:%Y-%m-%d %H:%M} UTC tidak tersedia di Binance"}
    cur_idx = int(cur.index[0])
    if cur_idx == 0:
        return {"error": "Tidak ada bar sebelumnya untuk Bar 2 / SQZMOM 2"}
    prev = df.iloc[cur_idx - 1]
    cur = df.iloc[cur_idx]

    def _bar_label(row) -> Optional[str]:
        c, z = row.get("Bar Color"), row.get("Fib Zone")
        if pd.isna(c) or pd.isna(z):
            return None
        return f"{c} Bar Line {int(z)}"

    setup = {
        "Bar 1": _bar_label(cur),
        "Bar 2": _bar_label(prev),
        "SQZMOM 1 Value": float(cur["SQZMOM Value"]) if not pd.isna(cur["SQZMOM Value"]) else None,
        "SQZMOM 1 Momentum": cur["Momentum Color"] if not pd.isna(cur["Momentum Color"]) else None,
        "SQZMOM 1 Squeeze": cur["Squeeze Status"] if not pd.isna(cur["Squeeze Status"]) else None,
        "SQZMOM 2 Value": float(prev["SQZMOM Value"]) if not pd.isna(prev["SQZMOM Value"]) else None,
        "SQZMOM 2 Momentum": prev["Momentum Color"] if not pd.isna(prev["Momentum Color"]) else None,
        "SQZMOM 2 Squeeze": prev["Squeeze Status"] if not pd.isna(prev["Squeeze Status"]) else None,
        # Untuk informasi/debug
        "_target_dt": str(target_dt_utc),
        "_close": float(cur["Close"]) if "Close" in cur and not pd.isna(cur["Close"]) else None,
    }

    missing = [k for k in ("Bar 1", "Bar 2", "SQZMOM 1 Value", "SQZMOM 2 Value") if setup.get(k) is None]
    if missing:
        setup["error"] = f"Field tidak lengkap (warmup belum cukup?): {missing}"
    return setup


if __name__ == "__main__":
    import json
    # Smoke test
    out = fetch_setup("ETH-USD", date_cls(2026, 5, 15), 10)
    print(json.dumps(out, indent=2, default=str))
