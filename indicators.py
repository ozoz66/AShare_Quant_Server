#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared technical indicator calculations and utility functions.

Used by scanner.py and analyzer.py to avoid code duplication.
All indicator functions are pure computations with no side effects.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys

import baostock as bs
import pandas as pd


# ---------------------------------------------------------------------------
# I/O Utilities
# ---------------------------------------------------------------------------

def configure_stdio() -> None:
    """Configure stdout/stderr for UTF-8 encoding (works on both Windows and Linux)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            continue


def emit_event(event: str, **data):
    """Emit structured progress events to stderr for front-end consumption."""
    payload = {"event": event, **data}
    try:
        print("EVENT_JSON:" + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data Conversion Utilities
# ---------------------------------------------------------------------------

def to_float(value):
    """Convert various value types to float, handling formatted numbers."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--", "None", "nan"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first matching column name from a list of candidates."""
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


# ---------------------------------------------------------------------------
# BaoStock Helpers
# ---------------------------------------------------------------------------

def bs_login():
    """Login to BaoStock with suppressed stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        bs.login()


def bs_logout():
    """Logout from BaoStock with suppressed stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        bs.logout()


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) using SMA smoothing.

    Note: Uses simple moving average for avg gain/loss rather than Wilder's
    original EMA. The scoring thresholds in this project are calibrated
    for this variant.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD: DIF, DEA, and histogram (DIF-DEA)*2."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def calculate_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 9, m1: int = 3, m2: int = 3):
    """KDJ indicator, popular in A-share market."""
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calculate_bollinger(close: pd.Series, period: int = 20,
                        num_std: float = 2.0):
    """Bollinger Bands: upper, middle, lower."""
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """Average True Range for volatility measurement and stop-loss sizing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def detect_macd_divergence(close: pd.Series, dif: pd.Series,
                           lookback: int = 30) -> str | None:
    """
    Detect MACD divergence - high-probability reversal signal.

    Bottom divergence: price makes new low but DIF doesn't -> bullish
    Top divergence: price makes new high but DIF doesn't -> bearish
    Returns: 'bottom_divergence', 'top_divergence', or None
    """
    if len(close) < lookback + 5 or len(dif) < lookback + 5:
        return None

    recent_close = close.iloc[-lookback:]
    recent_dif = dif.iloc[-lookback:]
    prev_close = close.iloc[-lookback * 2:-lookback] if len(close) >= lookback * 2 else None
    prev_dif = dif.iloc[-lookback * 2:-lookback] if len(dif) >= lookback * 2 else None

    if prev_close is None or prev_dif is None:
        return None

    # Bottom divergence: price new low, DIF higher low
    if (recent_close.min() < prev_close.min()
            and recent_dif.min() > prev_dif.min()
            and recent_dif.iloc[-1] > recent_dif.min()):
        return "bottom_divergence"

    # Top divergence: price new high, DIF lower high
    if (recent_close.max() > prev_close.max()
            and recent_dif.max() < prev_dif.max()
            and recent_dif.iloc[-1] < recent_dif.max()):
        return "top_divergence"

    return None


def check_pullback_to_ma(close: pd.Series, ma: pd.Series,
                         tolerance: float = 0.02) -> bool:
    """
    Check if price pulled back to MA support - better entry timing than chasing.
    Returns True if price is within 'tolerance' above the MA.
    """
    if len(close) < 3 or len(ma) < 3:
        return False
    price = close.iloc[-1]
    ma_val = ma.iloc[-1]
    if pd.isna(ma_val) or ma_val <= 0:
        return False
    ratio = (price - ma_val) / ma_val
    # Price is slightly above MA (0% to +tolerance), and was higher before
    was_higher = any(close.iloc[-5:-1] > ma.iloc[-5:-1] * (1 + tolerance * 2))
    return 0 <= ratio <= tolerance and was_higher
