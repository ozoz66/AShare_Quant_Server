#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-share deep analyzer (headless server edition).

Features:
- Resolve stock by code or name (no interactive input - must provide via CLI).
- Fetch real-time quote, fundamentals, technical indicators, and news.
- Output Markdown report or structured JSON (--json flag).
- Provide target price + hold/sell advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import akshare as ak
import baostock as bs
import pandas as pd
import requests

from indicators import (
    configure_stdio, emit_event, to_float, pick_col,
    bs_login, bs_logout,
    calculate_rsi, calculate_macd, calculate_kdj, calculate_bollinger,
    calculate_atr, detect_macd_divergence, check_pullback_to_ma,
    check_market_regime, retry
)
from news_analyzer import (
    fetch_stock_news, fetch_stock_notices,
    fetch_institute_recommendations,
    analyze_sentiment_rule_based, default_sentiment,
)

warnings.filterwarnings("ignore")

for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)
os.environ["NO_PROXY"] = "*"


configure_stdio()


# Removed local check_market_regime as it is imported from indicators.py


def normalize_quote_from_ak(row: dict) -> dict:
    def get(*keys):
        for k in keys:
            if k in row:
                return row.get(k)
        return None

    return {
        "name": str(get("名称", "股票简称") or "").strip(),
        "code": str(get("代码", "股票代码") or "").strip(),
        "price": to_float(get("最新价")),
        "change_pct": to_float(get("涨跌幅")),
        "change_amt": to_float(get("涨跌额")),
        "open": to_float(get("今开")),
        "high": to_float(get("最高")),
        "low": to_float(get("最低")),
        "prev_close": to_float(get("昨收")),
        "volume": to_float(get("成交量")),
        "amount": to_float(get("成交额")),
        "turnover_rate": to_float(get("换手率")),
        "amplitude": to_float(get("振幅")),
        "pe": to_float(get("市盈率-动态", "市盈率动态")),
        "pb": to_float(get("市净率")),
        "mkt_cap": to_float(get("总市值")),
        "float_cap": to_float(get("流通市值")),
        "vol_ratio": to_float(get("量比")),
    }


def fetch_tencent_quote(code: str) -> dict | None:
    try:
        prefix = "sh" if code.startswith("6") else "sz"
        url = f"http://qt.gtimg.cn/q={prefix}{code}"
        resp = requests.get(url, timeout=15)
        resp.encoding = "gbk"
        text = resp.text.strip()
        if "~" not in text:
            return None
        parts = text.split("~")
        if len(parts) < 47:
            return None
        mkt_cap_yi = to_float(parts[44])
        mkt_cap = mkt_cap_yi * 1e8 if mkt_cap_yi is not None else None
        return {
            "name": parts[1].strip(),
            "code": parts[2].strip(),
            "price": to_float(parts[3]),
            "prev_close": to_float(parts[4]),
            "open": to_float(parts[5]),
            "high": to_float(parts[33]),
            "low": to_float(parts[34]),
            "volume": to_float(parts[6]),
            "amount": to_float(parts[37]),
            "change_pct": to_float(parts[32]),
            "change_amt": to_float(parts[31]),
            "turnover_rate": to_float(parts[38]),
            "amplitude": None,
            "pe": to_float(parts[39]),
            "pb": to_float(parts[46]),
            "mkt_cap": mkt_cap,
            "float_cap": mkt_cap,
            "vol_ratio": None,
        }
    except Exception as e:
        print(f"  [WARN] 腾讯行情获取失败: {e}", file=sys.stderr)
        return None


@retry(max_attempts=2, delay=1.0)
def resolve_symbol(user_input: str) -> tuple[str | None, str | None, dict | None]:
    try:
        print("  正在获取股票列表以匹配输入...", file=sys.stderr)
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return None, None, None

        code_col = pick_col(df, ["代码"])
        name_col = pick_col(df, ["名称"])
        if code_col is None or name_col is None:
            return None, None, None

        value = user_input.strip()

        # exact code
        m = df[df[code_col].astype(str) == value]
        if not m.empty:
            row = m.iloc[0].to_dict()
            q = normalize_quote_from_ak(row)
            return str(row[code_col]), str(row[name_col]), q

        # exact name
        m = df[df[name_col].astype(str) == value]
        if not m.empty:
            row = m.iloc[0].to_dict()
            q = normalize_quote_from_ak(row)
            return str(row[code_col]), str(row[name_col]), q

        # fuzzy name
        m = df[df[name_col].astype(str).str.contains(value, na=False)]
        if not m.empty:
            row = m.iloc[0].to_dict()
            q = normalize_quote_from_ak(row)
            return str(row[code_col]), str(row[name_col]), q

        # partial code
        m = df[df[code_col].astype(str).str.contains(value, na=False)]
        if not m.empty:
            row = m.iloc[0].to_dict()
            q = normalize_quote_from_ak(row)
            return str(row[code_col]), str(row[name_col]), q

        return None, None, None
    except Exception as e:
        print(f"  [ERROR] 股票解析失败: {e}", file=sys.stderr)
        return None, None, None


@retry(max_attempts=3, delay=1.0)
def fetch_daily_history(symbol: str, days: int = 180) -> pd.DataFrame | None:
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            close_col = pick_col(df, ["收盘"])
            if close_col is not None:
                return df
    except Exception as e:
        print(f"  [WARN] akshare历史行情获取失败: {e}", file=sys.stderr)

    try:
        print("  正在尝试BaoStock备用数据源...", file=sys.stderr)
        bs_login()
        try:
            bs_code = f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
            sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=sd,
                end_date=ed,
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None

            tmp = pd.DataFrame(
                rows,
                columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"],
            )
            for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
                tmp[col] = tmp[col].map(to_float)
            tmp = tmp.dropna(subset=["收盘"])
            return tmp if not tmp.empty else None
        finally:
            bs_logout()
    except Exception as e:
        print(f"  [ERROR] BaoStock历史行情获取失败: {e}", file=sys.stderr)
        return None


@retry(max_attempts=2, delay=1.0)
def fetch_individual_info(symbol: str) -> dict:
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is None or df.empty:
            return {}

        item_col = pick_col(df, ["item", "项目"])
        value_col = pick_col(df, ["value", "值"])

        info = {}
        if item_col and value_col:
            for _, row in df.iterrows():
                info[str(row[item_col])] = row[value_col]
        else:
            for _, row in df.iterrows():
                info[str(row.iloc[0])] = row.iloc[1]
        return info
    except Exception as e:
        print(f"  [WARN] 个股信息获取失败: {e}", file=sys.stderr)
        return {}


def fetch_financial_indicators(symbol: str) -> dict:
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=symbol)
        if df is None or df.empty:
            return {}

        latest = df.iloc[0]
        out = {}

        date_col = pick_col(df, ["日期", "报告期"])
        if date_col:
            out["报告期"] = str(latest.get(date_col, ""))

        for col in df.columns:
            lc = str(col).lower()
            val = to_float(latest.get(col))
            if val is None:
                continue
            if "roe" in lc or "净资产收益率" in str(col):
                out["ROE"] = val
            elif "毛利率" in str(col):
                out["毛利率"] = val
            elif "净利率" in str(col) or "净利润率" in str(col):
                out["净利率"] = val

        return out
    except Exception as e:
        print(f"  [WARN] 财务指标获取失败: {e}", file=sys.stderr)
        return {}


def compute_technicals(hist: pd.DataFrame) -> dict | None:
    if hist is None or hist.empty or len(hist) < 30:
        return None

    close_col = pick_col(hist, ["收盘", "close"])
    if close_col is None:
        return None

    close = hist[close_col].map(to_float).dropna().astype(float).reset_index(drop=True)
    if len(close) < 30:
        return None

    n = len(close)

    high_col = pick_col(hist, ["最高", "high"])
    low_col = pick_col(hist, ["最低", "low"])
    vol_col = pick_col(hist, ["成交量", "volume"])

    high = hist[high_col].map(to_float).dropna().astype(float).reset_index(drop=True) if high_col else close
    low = hist[low_col].map(to_float).dropna().astype(float).reset_index(drop=True) if low_col else close
    volume = hist[vol_col].map(to_float).dropna().astype(float).reset_index(drop=True) if vol_col else None

    price = float(close.iloc[-1])

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean() if n >= 60 else None

    def _v(s, idx=-1):
        try:
            v = s.iloc[idx]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    ma5_now, ma10_now, ma20_now = _v(ma5), _v(ma10), _v(ma20)
    ma60_now = _v(ma60) if ma60 is not None else None

    dif, dea, macd_h = calculate_macd(close)
    dif_now, dea_now, macd_now = _v(dif), _v(dea), _v(macd_h)
    dif_prev, dea_prev, macd_prev = _v(dif, -2), _v(dea, -2), _v(macd_h, -2)

    rsi6 = calculate_rsi(close, 6)
    rsi12 = calculate_rsi(close, 12)
    rsi24 = calculate_rsi(close, 24)
    rsi6_now, rsi12_now, rsi24_now = _v(rsi6), _v(rsi12), _v(rsi24)

    k_val = d_val = j_val = None
    k_s = d_s = j_s = None
    if len(high) >= n and len(low) >= n:
        k_s, d_s, j_s = calculate_kdj(high[:n], low[:n], close)
        k_val, d_val, j_val = _v(k_s), _v(d_s), _v(j_s)

    bb_upper, bb_mid, bb_lower = calculate_bollinger(close)
    bb_u, bb_m, bb_l = _v(bb_upper), _v(bb_mid), _v(bb_lower)
    bb_width = (
        round((bb_u - bb_l) / bb_m * 100, 1)
        if bb_m is not None and bb_m > 0 and bb_u is not None and bb_l is not None
        else None
    )
    bb_pos = (
        round((price - bb_l) / (bb_u - bb_l) * 100, 1)
        if bb_u is not None and bb_l is not None and (bb_u - bb_l) > 0
        else None
    )

    vol_ratio = vol_trend_ratio = None
    vol_label = "N/A"
    if volume is not None and len(volume) >= 20:
        vol_20_avg = volume.iloc[-20:].mean()
        vol_5_avg = volume.iloc[-5:].mean()
        if vol_20_avg > 0:
            vol_ratio = round(volume.iloc[-1] / vol_20_avg, 2)
            vol_trend_ratio = round(vol_5_avg / vol_20_avg, 2)
            if vol_ratio > 2.5:
                vol_label = "显著放量"
            elif vol_ratio > 1.3:
                vol_label = "温和放量"
            elif vol_ratio > 0.7:
                vol_label = "量能平稳"
            else:
                vol_label = "明显缩量"

    support = round(low.iloc[-20:].min(), 2) if len(low) >= 20 else round(low.min(), 2)
    resistance = round(high.iloc[-20:].max(), 2) if len(high) >= 20 else round(high.max(), 2)

    atr_val = None
    stop_loss = None
    risk_reward = None
    atr_series = calculate_atr(high, low, close, 14)
    if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
        atr_val = float(atr_series.iloc[-1])
        stop_loss = round(price - 2 * atr_val, 2)
        potential_loss = price - stop_loss
        potential_gain = resistance - price
        if potential_loss > 0:
            risk_reward = round(potential_gain / potential_loss, 2)

    divergence = detect_macd_divergence(close, dif, lookback=30)
    pullback_ma20 = check_pullback_to_ma(close, ma20, tolerance=0.02)

    # =================================================================
    # Trend Signals
    # =================================================================
    signals = []
    macd_golden = (dif_prev is not None and dea_prev is not None and
                   dif_now is not None and dea_now is not None and
                   dif_prev <= dea_prev and dif_now > dea_now)
    macd_death = (dif_prev is not None and dea_prev is not None and
                  dif_now is not None and dea_now is not None and
                  dif_prev >= dea_prev and dif_now < dea_now)

    if macd_golden:
        signals.append("MACD 金叉形成 (短期看多)")
    elif macd_death:
        signals.append("MACD 死叉形成 (短期看空)")
    elif dif_now is not None and dea_now is not None:
        signals.append("MACD 多头运行" if dif_now > dea_now else "MACD 空头运行")

    if divergence == "bottom_divergence":
        signals.append("MACD 底背离 (极强底部信号)")
    elif divergence == "top_divergence":
        signals.append("MACD 顶背离 (波段见顶信号, 警戒)")

    if pullback_ma20:
        signals.append("价格回踩 MA20 支撑 (低风险切入点)")

    if dif_now is not None:
        signals.append(f"DIF {'>' if dif_now > 0 else '<'} 0, 中期趋势偏{'多' if dif_now > 0 else '空'}")

    if ma5_now is not None and ma10_now is not None and ma20_now is not None:
        if price > ma5_now > ma10_now > ma20_now:
            signals.append("均线完美多头排列 (价格>MA5>MA10>MA20)")
        elif price > ma5_now > ma20_now:
            signals.append("均线多头排列 (价格>MA5>MA20)")
        elif price < ma5_now < ma20_now:
            signals.append("均线空头排列 (价格<MA5<MA20)")
        elif price > ma20_now:
            signals.append("价格站上MA20, 中期支撑有效")
        else:
            signals.append("价格跌破MA20, 中期支撑失守")

    if ma60_now is not None:
        signals.append(f"价格{'站上' if price > ma60_now else '跌破'}MA60 (长期趋势)")

    ma5_prev, ma20_prev_v = _v(ma5, -2), _v(ma20, -2)
    if (
        ma5_prev is not None
        and ma20_prev_v is not None
        and ma5_now is not None
        and ma20_now is not None
    ):
        if ma5_prev <= ma20_prev_v and ma5_now > ma20_now:
            signals.append("MA5/MA20 金叉")
        elif ma5_prev >= ma20_prev_v and ma5_now < ma20_now:
            signals.append("MA5/MA20 死叉")

    if rsi6_now is not None:
        if rsi6_now > 80:
            signals.append(f"RSI(6)={rsi6_now:.1f} 超买, 注意回调")
        elif rsi6_now < 20:
            signals.append(f"RSI(6)={rsi6_now:.1f} 超卖, 关注反弹")
        elif rsi6_now > 50:
            signals.append(f"RSI(6)={rsi6_now:.1f} 偏强")
        else:
            signals.append(f"RSI(6)={rsi6_now:.1f} 偏弱")

    if k_val is not None and d_val is not None and j_val is not None:
        if j_val > 100:
            signals.append(f"KDJ J={j_val:.1f} 超买")
        elif j_val < 0:
            signals.append(f"KDJ J={j_val:.1f} 超卖")
        k_prev = _v(k_s, -2) if k_s is not None else None
        d_prev_v = _v(d_s, -2) if d_s is not None else None
        if k_prev is not None and d_prev_v is not None:
            if k_prev <= d_prev_v and k_val > d_val:
                signals.append("KDJ 金叉")
            elif k_prev >= d_prev_v and k_val < d_val:
                signals.append("KDJ 死叉")

    if bb_u is not None and bb_l is not None:
        if price >= bb_u:
            signals.append("价格触及布林上轨, 注意压力")
        elif price <= bb_l:
            signals.append("价格触及布林下轨, 关注支撑")
        if bb_width is not None and bb_width < 5:
            signals.append("布林带收窄, 可能即将变盘")

    if vol_label != "N/A":
        price_up = close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else False
        signals.append(f"量能: {vol_label} (量比{vol_ratio})")
        if vol_ratio is not None and vol_ratio > 1.3 and price_up:
            signals.append("量价配合良好 (放量上涨)")
        elif vol_ratio is not None and vol_ratio > 1.3 and not price_up:
            signals.append("放量下跌, 注意风险")

    # =================================================================
    # Composite Scoring (0-100)
    # =================================================================
    score = 0.0

    if ma5_now is not None and ma10_now is not None and ma20_now is not None:
        if price > ma5_now > ma10_now > ma20_now: score += 15
        elif price > ma5_now > ma20_now: score += 12
        elif price > ma20_now: score += 8
        elif price > ma5_now: score += 4
    if ma60_now is not None and price > ma60_now: score += 8
    elif ma60_now is None: score += 4
    if ma20_now is not None and len(ma20) >= 6:
        ma20_5ago = _v(ma20, -6)
        if ma20_5ago is not None and ma20_5ago > 0:
            slope = (ma20_now - ma20_5ago) / ma20_5ago
            if slope > 0.01: score += 7
            elif slope > 0: score += 4
            elif slope > -0.005: score += 2

    if macd_golden: score += 12
    elif (
        dif_now is not None
        and dea_now is not None
        and dif_now > dea_now
        and macd_now is not None
        and macd_prev is not None
        and macd_now > macd_prev
    ):
        score += 10
    elif dif_now is not None and dea_now is not None and dif_now > dea_now:
        score += 7
    elif macd_now is not None and macd_prev is not None and macd_now > macd_prev:
        score += 4
    if dif_now is not None and dif_now > 0:
        score += 5
    elif dif_now is not None and dif_now > -0.5:
        score += 2
    if rsi6_now is not None:
        if 40 <= rsi6_now <= 55: score += 8
        elif 55 < rsi6_now <= 65: score += 6
        elif 30 <= rsi6_now < 40: score += 5
        elif 65 < rsi6_now <= 75: score += 3
        else: score += 1
    if (
        k_val is not None
        and d_val is not None
        and j_val is not None
        and k_val > d_val
        and 20 < j_val < 80
    ):
        score += 5

    if vol_ratio is not None:
        if 1.2 <= vol_ratio <= 2.5: score += 12
        elif 0.8 <= vol_ratio < 1.2: score += 8
        elif 2.5 < vol_ratio <= 4.0: score += 6
        elif 0.5 <= vol_ratio < 0.8: score += 4
        else: score += 2
        if vol_trend_ratio is not None and 1.1 <= vol_trend_ratio <= 2.0: score += 5
        elif vol_trend_ratio is not None and 0.9 <= vol_trend_ratio < 1.1: score += 3
        else: score += 1
    else:
        score += 10

    if bb_pos is not None:
        if 30 <= bb_pos <= 70: score += 8
        elif 20 <= bb_pos < 30 or 70 < bb_pos <= 80: score += 5
        else: score += 2

    bonus = 0.0
    if divergence == "bottom_divergence": bonus += 8
    elif divergence == "top_divergence": bonus -= 8
    if pullback_ma20: bonus += 5
    if risk_reward is not None and risk_reward > 2.0: bonus += 3
    elif risk_reward is not None and risk_reward < 0.8: bonus -= 3

    score = max(0, min(100, round(score + bonus, 1)))
    if score >= 80: rating = "★★★★★ 强烈看多"
    elif score >= 65: rating = "★★★★☆ 偏多"
    elif score >= 50: rating = "★★★☆☆ 中性"
    elif score >= 35: rating = "★★☆☆☆ 偏空"
    else: rating = "★☆☆☆☆ 看空"

    def _r(v, d=2):
        return round(v, d) if v is not None else "N/A"

    return {
        "close": round(price, 2),
        "ma5": _r(ma5_now), "ma10": _r(ma10_now),
        "ma20": _r(ma20_now), "ma60": _r(ma60_now),
        "dif": _r(dif_now, 4), "dea": _r(dea_now, 4), "macd": _r(macd_now, 4),
        "rsi6": _r(rsi6_now), "rsi12": _r(rsi12_now), "rsi24": _r(rsi24_now),
        "k": _r(k_val, 1), "d": _r(d_val, 1), "j": _r(j_val, 1),
        "bb_upper": _r(bb_u), "bb_mid": _r(bb_m), "bb_lower": _r(bb_l),
        "bb_width": bb_width, "bb_pos": bb_pos,
        "vol_ratio": vol_ratio, "vol_label": vol_label,
        "support": support, "resistance": resistance,
        "atr": _r(atr_val), "stop_loss": stop_loss, "risk_reward": risk_reward,
        "divergence": divergence,
        "score": score, "rating": rating,
        "signals": signals,
    }


def build_trade_advice(quote: dict | None, tech: dict | None, fin_indicators: dict,
                       sentiment: dict | None = None) -> dict:
    """Build trade advice based on composite technical score + fundamentals + sentiment."""
    score = 0
    reasons = []

    current_price = None
    if tech:
        current_price = to_float(tech.get("close"))
    if current_price is None and quote:
        current_price = to_float(quote.get("price"))

    if tech:
        tech_score = tech.get("score", 0)
        if tech_score >= 80:
            score += 3
            reasons.append(f"技术面极强({tech_score}分)")
        elif tech_score >= 65:
            score += 2
            reasons.append(f"技术综合评分{tech_score}分 (偏多)")
        elif tech_score >= 50:
            score += 1
            reasons.append(f"技术综合评分{tech_score}分 (中性)")
        elif tech_score >= 35:
            score -= 1
            reasons.append(f"技术综合评分{tech_score}分 (偏弱)")
        else:
            score -= 2
            reasons.append(f"技术综合评分{tech_score}分 (看空)")

        div = tech.get("divergence")
        if div == "bottom_divergence":
            score += 2
            reasons.append("检测到MACD底背离 (高胜率反转点)")
        elif div == "top_divergence":
            score -= 3
            reasons.append("检测到MACD顶背离 (波段筑顶, 风险极大)")

        vol_label = tech.get("vol_label", "N/A")
        if vol_label == "温和放量" and tech.get("score", 0) >= 50:
            score += 1
            reasons.append("量价配合良好")
        elif vol_label == "显著放量" and tech.get("score", 0) < 50:
            score -= 1
            reasons.append("放量但技术面偏弱, 注意风险")

    roe = None
    for k, v in (fin_indicators or {}).items():
        if "roe" in str(k).lower():
            roe = to_float(v)
            break

    if roe is not None:
        if roe >= 15:
            score += 1
            reasons.append(f"ROE={roe:.1f}% 优秀")
        elif roe >= 10:
            reasons.append(f"ROE={roe:.1f}% 良好")
        elif roe < 5:
            score -= 1
            reasons.append(f"ROE={roe:.1f}% 偏弱")

    if sentiment:
        sent_score = sentiment.get("score", 5)
        sent_label = sentiment.get("label", "中性")
        sent_summary = sentiment.get("summary", "")
        if sent_score >= 7:
            score += 2
            reasons.append(f"消息面利好({sent_label} {sent_score}/10): {sent_summary}")
        elif sent_score <= 3:
            score -= 2
            reasons.append(f"消息面利空({sent_label} {sent_score}/10): {sent_summary}")
        elif sent_score != 5:
            reasons.append(f"消息面{sent_label}({sent_score}/10)")

    if score >= 3:
        action = "强烈看多 (积极布局)"
    elif score >= 2:
        action = "积极持有"
    elif score >= 1:
        action = "持有"
    elif score >= 0:
        action = "观望"
    else:
        action = "减仓/卖出"

    if current_price is None:
        return {"action": action, "target_price": None, "score": score,
                "reasons": reasons or ["数据不足"]}

    if score >= 2:
        target = current_price * 1.15
    elif score >= 1:
        target = current_price * 1.08
    elif score >= 0:
        target = current_price * 1.03
    else:
        target = current_price * 0.95

    return {"action": action, "target_price": round(target, 2),
            "score": score, "reasons": reasons or ["数据不足"]}


def format_large_number(value) -> str:
    v = to_float(value)
    if v is None:
        return "N/A"
    if v >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if v >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:.2f}"


def output_report_text(name: str, code: str, quote: dict | None, info: dict, fin: dict,
                       tech: dict | None, news: list[dict], advice: dict,
                       mkt: dict | None = None, sentiment: dict | None = None,
                       notices: list[dict] | None = None, reports: list[dict] | None = None) -> str:
    """Generate markdown report as string (no direct printing)."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# 深度分析报告: {name} ({code})\n")
    lines.append(f"> 生成时间: {now}  ")
    lines.append("> 数据来源: 东方财富(akshare) + BaoStock/腾讯备用\n")

    if mkt:
        regime = mkt.get("regime", "neutral")
        label = {"bull": "牛市 (向上趋势)", "bear": "熊市 (风险较高)", "neutral": "震荡 (趋势不明)"}.get(regime, "震荡")
        lines.append(f"**大盘环境: {label}** | 原因: {mkt.get('reason', '')}")
        if regime == "bear":
            lines.append("> WARNING: 当前大盘处于弱势，个体指标可靠性降低，建议控制仓位。")
        lines.append("")

    lines.append("## 一、实时行情\n")
    if quote:
        lines.append("| 指标 | 数值 | 指标 | 数值 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| 最新价 | {quote.get('price', 'N/A')} | 涨跌幅 | {quote.get('change_pct', 'N/A')}% |")
        lines.append(f"| 涨跌额 | {quote.get('change_amt', 'N/A')} | 振幅 | {quote.get('amplitude', 'N/A')}% |")
        lines.append(f"| 今开 | {quote.get('open', 'N/A')} | 昨收 | {quote.get('prev_close', 'N/A')} |")
        lines.append(f"| 最高 | {quote.get('high', 'N/A')} | 最低 | {quote.get('low', 'N/A')} |")
        lines.append(f"| 成交量 | {format_large_number(quote.get('volume'))}手 | 成交额 | {format_large_number(quote.get('amount'))} |")
        lines.append(f"| 换手率 | {quote.get('turnover_rate', 'N/A')}% | 量比 | {quote.get('vol_ratio', 'N/A')} |")
        lines.append(f"| 市盈率(动态) | {quote.get('pe', 'N/A')} | 市净率 | {quote.get('pb', 'N/A')} |")
        lines.append(f"| 总市值 | {format_large_number(quote.get('mkt_cap'))} | 流通市值 | {format_large_number(quote.get('float_cap'))} |")
    else:
        lines.append("*实时行情数据获取失败*\n")

    lines.append("\n## 二、基本面信息\n")
    if info:
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        for k, v in info.items():
            show = format_large_number(v) if k in {"总市值", "流通市值", "总股本", "流通股"} else v
            lines.append(f"| {k} | {show} |")
    else:
        lines.append("*基本面数据获取失败*\n")

    if fin:
        lines.append("\n### 核心财务指标\n")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for k, v in fin.items():
            lines.append(f"| {k} | {v} |")

    lines.append("\n## 三、技术分析\n")
    if tech:
        lines.append(f"### 综合评级: {tech.get('score', 'N/A')}/100  {tech.get('rating', '')}\n")

        lines.append("### 均线 & MACD\n")
        lines.append("| 指标 | 数值 | 指标 | 数值 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| MA5 | {tech['ma5']} | MA10 | {tech['ma10']} |")
        lines.append(f"| MA20 | {tech['ma20']} | MA60 | {tech['ma60']} |")
        lines.append(f"| MACD DIF | {tech['dif']} | DEA | {tech['dea']} |")
        lines.append(f"| MACD 柱状 | {tech['macd']} | 背离 | {tech.get('divergence') or '无'} |")

        lines.append(f"\n### RSI & KDJ\n")
        lines.append("| 指标 | 数值 | 指标 | 数值 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| RSI(6) | {tech['rsi6']} | RSI(12) | {tech['rsi12']} |")
        lines.append(f"| RSI(24) | {tech['rsi24']} | | |")
        lines.append(f"| KDJ K | {tech['k']} | KDJ D | {tech['d']} |")
        lines.append(f"| KDJ J | {tech['j']} | | |")

        lines.append(f"\n### 风控 & 支撑压力\n")
        lines.append("| 指标 | 数值 | 指标 | 数值 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| 止损价(ATR) | **{tech.get('stop_loss', 'N/A')}** | 盈亏比 | **{tech.get('risk_reward', 'N/A')}** |")
        lines.append(f"| 14日ATR | {tech.get('atr', 'N/A')} | 支撑位 | {tech['support']} |")
        lines.append(f"| 压力位 | {tech['resistance']} | | |")

        lines.append(f"\n### 布林带 & 量能\n")
        lines.append("| 指标 | 数值 | 指标 | 数值 |")
        lines.append("|------|------|------|------|")
        lines.append(f"| 布林上轨 | {tech['bb_upper']} | 布林下轨 | {tech['bb_lower']} |")
        lines.append(f"| 布林中轨 | {tech['bb_mid']} | 带宽% | {tech.get('bb_width', 'N/A')} |")
        lines.append(f"| 布林位置% | {tech.get('bb_pos', 'N/A')} | 量比 | {tech.get('vol_ratio', 'N/A')} |")
        lines.append(f"| 量能状态 | {tech['vol_label']} | | |")

        lines.append("\n### 趋势信号\n")
        for s in tech.get("signals", []):
            lines.append(f"- {s}")
    else:
        lines.append("*技术指标计算失败（历史数据不足）*\n")

    lines.append("\n## 四、最新新闻动态\n")
    if news:
        for i, item in enumerate(news, 1):
            lines.append(f"{i}. **{item.get('title', '无标题')}**")
            lines.append(f"   > 来源: {item.get('source', '')} | 时间: {item.get('time', '')}")
    else:
        lines.append("*暂无相关新闻*\n")

    lines.append("\n## 五、消息面分析\n")
    if sentiment and sentiment.get("score", 5) != 5:
        sent_score = sentiment.get("score", 5)
        sent_label = sentiment.get("label", "中性")
        lines.append(f"### 情绪判断: {sent_label} ({sent_score}/10)\n")
        lines.append("| 指标 | 结果 |")
        lines.append("|------|------|")
        lines.append(f"| 情绪评分 | {sent_score}/10 ({sent_label}) |")
        themes = ", ".join(sentiment.get("key_themes", [])) or "无"
        risks = ", ".join(sentiment.get("risk_flags", [])) or "暂无"
        lines.append(f"| 关键主题 | {themes} |")
        lines.append(f"| 风险提示 | {risks} |")
        summary = sentiment.get("summary", "")
        if summary:
            lines.append(f"| 一句话总结 | {summary} |")
    else:
        lines.append("*消息面情绪: 中性 (数据不足)*\n")

    if notices:
        lines.append(f"\n### 公司公告 (最近{len(notices)}条)\n")
        for i, item in enumerate(notices, 1):
            lines.append(f"{i}. **{item.get('title', '无标题')}**")
            lines.append(f"   > 来源: {item.get('source', '')} | 时间: {item.get('time', '')}")

    if reports:
        lines.append(f"\n### 机构评级 (最近{len(reports)}条)\n")
        for i, item in enumerate(reports, 1):
            rating = item.get("rating", "")
            tp = item.get("target_price")
            tp_str = f" | 目标价: {tp}" if tp else ""
            analyst = item.get("analyst", "")
            analyst_str = f" | 分析师: {analyst}" if analyst else ""
            lines.append(f"{i}. **{item.get('title', '无标题')}**")
            lines.append(f"   > {item.get('source', '')}{tp_str}{analyst_str} | {item.get('time', '')}")

    lines.append("\n## 六、目标价与操作建议\n")
    target = advice.get("target_price")
    action = advice.get("action", "卖出")
    adv_score = advice.get("score", 0)
    if target is None:
        lines.append("- 目标价: 数据不足，暂无法给出可靠数值")
    elif "减仓" in action or "卖出" in action:
        lines.append(f"- 反弹减仓参考价: **{target}**")
    elif "观望" in action:
        lines.append(f"- 观察参考价: **{target}**")
    else:
        lines.append(f"- 3个月参考目标价: **{target}**")
    lines.append(f"- 操作建议: **{action}**")
    lines.append(f"- 综合评分: **{adv_score}**")
    lines.append("- 主要依据:")
    for r in advice.get("reasons", [])[:5]:
        lines.append(f"  - {r}")

    lines.append("\n---")
    lines.append("*免责声明: 本报告由程序自动生成，仅供参考研究使用，不构成任何投资建议。投资有风险，入市需谨慎。*")

    return "\n".join(lines)


def build_json_report(name: str, code: str, quote: dict | None, info: dict, fin: dict,
                      tech: dict | None, news: list[dict], advice: dict,
                      mkt: dict | None = None, sentiment: dict | None = None,
                      notices: list[dict] | None = None, reports: list[dict] | None = None) -> dict:
    """Build structured JSON report for programmatic consumption."""
    return {
        "report_type": "analyzer",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock": {"code": code, "name": name},
        "market_regime": mkt or {},
        "quote": quote or {},
        "fundamentals": info,
        "financial_indicators": fin,
        "technicals": tech or {},
        "news": news,
        "notices": notices or [],
        "reports": reports or [],
        "sentiment": sentiment or default_sentiment(),
        "trade_advice": advice,
        "disclaimer": "本报告由程序自动生成，仅供参考研究使用，不构成任何投资建议。",
    }


def run_analyzer(symbol: str, output_json: bool = False) -> dict | str:
    """
    Core analyzer logic. Returns JSON dict if output_json=True, else markdown string.
    Can be called programmatically or from CLI.
    """
    if not symbol:
        raise ValueError("未提供股票代码或名称")

    emit_event("step", current=1, total=10, desc="检测大盘环境")
    print(">>> [1/10] 检测大盘环境...", file=sys.stderr)
    mkt = check_market_regime()

    emit_event("step", current=2, total=10, desc="解析股票并获取实时行情")
    print(f">>> [2/10] 解析股票并获取实时行情: {symbol}", file=sys.stderr)
    code, name, quote = resolve_symbol(symbol)

    if code is None or name is None:
        if symbol.isdigit() and len(symbol) == 6:
            print("  尝试腾讯行情备用源...", file=sys.stderr)
            q = fetch_tencent_quote(symbol)
            if q and q.get("price") is not None:
                code = symbol
                name = q.get("name", symbol)
                quote = q

    if code is None or name is None:
        raise ValueError(f"无法找到匹配的股票: {symbol}")

    print(f"    已匹配: {code} {name}", file=sys.stderr)

    emit_event("step", current=3, total=10, desc="获取历史数据并计算技术指标")
    print(">>> [3/10] 获取历史数据并计算技术指标...", file=sys.stderr)
    hist = fetch_daily_history(code, days=180)
    tech = compute_technicals(hist)

    emit_event("step", current=4, total=10, desc="获取基本面数据")
    print(">>> [4/10] 获取基本面数据...", file=sys.stderr)
    info = fetch_individual_info(code)

    emit_event("step", current=5, total=10, desc="获取核心财务指标")
    print(">>> [5/10] 获取核心财务指标...", file=sys.stderr)
    fin = fetch_financial_indicators(code)

    emit_event("step", current=6, total=10, desc="获取最新新闻")
    print(">>> [6/10] 获取最新新闻...", file=sys.stderr)
    news = fetch_stock_news(code, count=5)

    emit_event("step", current=7, total=10, desc="获取公司公告与机构评级")
    print(">>> [7/10] 获取公司公告与机构评级...", file=sys.stderr)
    notices = fetch_stock_notices(code, count=5)
    reports_data = fetch_institute_recommendations(code, count=5)

    if quote is None:
        print(">>> [bonus] 尝试腾讯行情补充实时数据...", file=sys.stderr)
        quote = fetch_tencent_quote(code)

    emit_event("step", current=8, total=10, desc="规则消息面情感分析")
    print(">>> [8/10] 规则消息面情感分析...", file=sys.stderr)
    sentiment = analyze_sentiment_rule_based(news, notices, reports_data)
    print(f"    情感分析: {sentiment.get('label', '中性')} ({sentiment.get('score', 5)}/10)", file=sys.stderr)

    emit_event("step", current=9, total=10, desc="生成交易建议")
    print(">>> [9/10] 生成交易建议...", file=sys.stderr)
    advice = build_trade_advice(quote, tech, fin, sentiment=sentiment)

    emit_event("step", current=10, total=10, desc="输出深度分析报告")
    print(">>> [10/10] 输出深度分析报告...", file=sys.stderr)

    if output_json:
        return build_json_report(name, code, quote, info, fin, tech, news, advice,
                                 mkt=mkt, sentiment=sentiment, notices=notices, reports=reports_data)
    else:
        return output_report_text(name, code, quote, info, fin, tech, news, advice,
                                  mkt=mkt, sentiment=sentiment, notices=notices, reports=reports_data)


def main():
    parser = argparse.ArgumentParser(description="A-Share Deep Analyzer (headless server)")
    parser.add_argument("symbol_arg", nargs="?", help="股票代码或名称")
    parser.add_argument("--symbol", required=False, help="股票代码或名称")
    parser.add_argument("--json", action="store_true", help="输出JSON格式（供OpenClaw等程序解析）")
    args = parser.parse_args()

    user_input = (args.symbol or args.symbol_arg or "").strip()
    if not user_input:
        print("ERROR: 必须提供股票代码或名称。用法: python analyzer.py --symbol 600519", file=sys.stderr)
        sys.exit(1)

    result = run_analyzer(user_input, output_json=args.json)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
