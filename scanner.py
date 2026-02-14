#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-share scanner (v3 - headless server edition):
1) Screen stock pool by fundamentals and technicals.
2) Score candidates with multi-factor composite model.
3) Output top-N recommendations (default 15).
4) Supports --json flag for structured JSON output (OpenClaw integration).

Multi-factor scoring dimensions:
  - Trend: MA arrangement, MACD status, price vs MA20
  - Momentum: RSI position, MACD histogram direction
  - Volume: Volume ratio vs 20-day average
  - Valuation: PE/PB relative scoring
  - Sentiment: LLM-based news/announcement/research sentiment
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import baostock as bs
import pandas as pd
import requests

from indicators import (
    configure_stdio, emit_event, to_float, pick_col,
    bs_login, bs_logout,
    calculate_rsi, calculate_macd, calculate_kdj, calculate_atr,
    detect_macd_divergence, check_pullback_to_ma,
)
from news_analyzer import (
    fetch_stock_news, fetch_stock_notices,
    fetch_institute_recommendations,
    analyze_sentiment_batch, try_load_llm_config, default_sentiment,
)

warnings.filterwarnings("ignore")


for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)
os.environ["NO_PROXY"] = "*"


configure_stdio()


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def check_market_regime() -> dict:
    """
    Check overall market regime by analyzing Shanghai Composite.
    Returns regime info: 'bull', 'neutral', or 'bear' with confidence.
    """
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df is None or df.empty or len(df) < 60:
            return {"regime": "neutral", "confidence": 0.0, "reason": "数据不足"}
        close_col = pick_col(df, ["close", "收盘"])
        if close_col is None:
            return {"regime": "neutral", "confidence": 0.0, "reason": "无收盘列"}
        close = df[close_col].astype(float).iloc[-120:]
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        p = close.iloc[-1]
        ma20_now = ma20.iloc[-1]
        ma60_now = ma60.iloc[-1] if not pd.isna(ma60.iloc[-1]) else None

        ma20_slope = (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] if not pd.isna(ma20.iloc[-6]) and ma20.iloc[-6] > 0 else 0

        score = 0.0
        reasons = []
        if p > ma20_now:
            score += 1
            reasons.append("指数>MA20")
        else:
            score -= 1
            reasons.append("指数<MA20")
        if ma60_now and p > ma60_now:
            score += 1
            reasons.append("指数>MA60")
        elif ma60_now:
            score -= 1
            reasons.append("指数<MA60")
        if ma20_slope > 0.005:
            score += 1
            reasons.append("MA20上升")
        elif ma20_slope < -0.005:
            score -= 1
            reasons.append("MA20下降")

        if score >= 2:
            regime = "bull"
        elif score <= -2:
            regime = "bear"
        else:
            regime = "neutral"
        return {"regime": regime, "confidence": abs(score) / 3.0,
                "reason": "; ".join(reasons), "score": score}
    except Exception as e:
        print(f"  [WARN] 大盘环境检测失败: {e}", file=sys.stderr)
        return {"regime": "neutral", "confidence": 0.0, "reason": f"检测异常: {e}"}

def load_stock_codes(pool_path: Path) -> list[str]:
    with pool_path.open("r", encoding="utf-8") as f:
        pool = json.load(f)
    stocks = pool.get("stocks", [])
    return [str(s.get("code", "")).strip() for s in stocks if str(s.get("code", "")).strip()]


def _fetch_tencent_batch(codes: list[str]) -> list[dict]:
    results: list[dict] = []
    batch_size = 80
    for i in range(0, len(codes), batch_size):
        batch = codes[i : i + batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)
        try:
            resp = requests.get(url, timeout=15)
            resp.encoding = "gbk"
            lines = resp.text.strip().split(";")
            for line in lines:
                if "~" not in line:
                    continue
                parts = line.split("~")
                if len(parts) < 47:
                    continue
                code = parts[2].strip()
                name = parts[1].strip()
                price = to_float(parts[3])
                pe = to_float(parts[39])
                pb = to_float(parts[46])
                mkt_cap_yi = to_float(parts[44])
                if not code or price is None or price <= 0:
                    continue
                results.append(
                    {
                        "code": code,
                        "name": name,
                        "price": price,
                        "pe": pe,
                        "pb": pb,
                        "mkt_cap": (mkt_cap_yi * 1e8) if mkt_cap_yi is not None else None,
                    }
                )
        except Exception as e:
            print(f"[WARN] Tencent batch {i // batch_size + 1} failed: {e}", file=sys.stderr)
        time.sleep(0.1)
    return results


def fetch_pool_quotes(codes: list[str]) -> pd.DataFrame | None:
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            code_col = pick_col(df, ["代码"])
            name_col = pick_col(df, ["名称"])
            price_col = pick_col(df, ["最新价"])
            pe_col = pick_col(df, ["市盈率-动态", "市盈率动态"])
            pb_col = pick_col(df, ["市净率"])
            cap_col = pick_col(df, ["总市值"])

            if code_col and name_col and price_col and pe_col and pb_col and cap_col:
                subset = df[df[code_col].astype(str).isin(codes)].copy()
                if not subset.empty:
                    out = pd.DataFrame(
                        {
                            "code": subset[code_col].astype(str),
                            "name": subset[name_col].astype(str),
                            "price": subset[price_col].map(to_float),
                            "pe": subset[pe_col].map(to_float),
                            "pb": subset[pb_col].map(to_float),
                            "mkt_cap": subset[cap_col].map(to_float),
                        }
                    )
                    out = out.dropna(subset=["price"])
                    if not out.empty:
                        return out
    except Exception:
        pass

    tencent_codes = [("sh" + c) if c.startswith("6") else ("sz" + c) for c in codes]
    data = _fetch_tencent_batch(tencent_codes)
    if not data:
        return None
    return pd.DataFrame(data)


def filter_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced fundamental filter:
      - PE: (0, 80]
      - PB: (0, 15]
      - Market cap: > 200亿
      - Exclude negative PE (loss-making)
    """
    f = df.copy()
    f = f[f["pe"].notna() & (f["pe"] > 0) & (f["pe"] < 80)]
    f = f[f["pb"].notna() & (f["pb"] > 0) & (f["pb"] < 15)]
    f = f[f["mkt_cap"].notna() & (f["mkt_cap"] > 2e10)]
    return f


def fetch_history(symbol: str, days: int = 150) -> pd.DataFrame | None:
    """Fetch history with close, high, low, volume columns for scoring."""
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
            high_col = pick_col(df, ["最高"])
            low_col = pick_col(df, ["最低"])
            vol_col = pick_col(df, ["成交量"])
            if close_col is None:
                return None
            out_cols = {"close": df[close_col].map(to_float)}
            if high_col:
                out_cols["high"] = df[high_col].map(to_float)
            if low_col:
                out_cols["low"] = df[low_col].map(to_float)
            if vol_col:
                out_cols["volume"] = df[vol_col].map(to_float)
            out = pd.DataFrame(out_cols).dropna(subset=["close"])
            if not out.empty:
                return out
    except Exception:
        pass

    try:
        bs_code = f"sh.{symbol}" if symbol.startswith("6") else f"sz.{symbol}"
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
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
        tmp = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for c in ["high", "low", "close", "volume"]:
            tmp[c] = tmp[c].map(to_float)
        tmp = tmp.dropna(subset=["close"])
        return tmp[["close", "high", "low", "volume"]] if not tmp.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-factor Technical Analysis & Scoring
# ---------------------------------------------------------------------------

def check_technicals(symbol: str, market_regime: str = "neutral") -> dict | None:
    """
    Enhanced technical check with multi-factor scoring + win-rate boosters.

    Scoring model (0-100):
      Trend (30pts):  MA arrangement + price vs MA20/MA60
      Momentum (25pts):  MACD status + RSI zone + KDJ
      Volume (20pts):  Volume ratio (today vs 20d avg)
      Valuation (25pts): Filled later in composite_score()
    """
    hist = fetch_history(symbol, days=150)
    if hist is None or hist.empty or len(hist) < 30:
        return None

    close = hist["close"].astype(float).reset_index(drop=True)
    has_volume = "volume" in hist.columns
    has_high = "high" in hist.columns
    has_low = "low" in hist.columns
    volume = hist["volume"].astype(float).reset_index(drop=True) if has_volume else None
    high = hist["high"].astype(float).reset_index(drop=True) if has_high else close
    low = hist["low"].astype(float).reset_index(drop=True) if has_low else close
    n = len(close)

    current_price = close.iloc[-1]

    # --- Moving Averages ---
    ma5 = close.rolling(window=5).mean()
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    ma60 = close.rolling(window=60).mean() if n >= 60 else None

    ma5_now = ma5.iloc[-1]
    ma10_now = ma10.iloc[-1]
    ma20_now = ma20.iloc[-1]
    ma60_now = ma60.iloc[-1] if ma60 is not None and not pd.isna(ma60.iloc[-1]) else None

    if pd.isna(ma20_now):
        return None

    # --- RSI ---
    rsi6 = calculate_rsi(close, 6).iloc[-1]
    rsi12 = calculate_rsi(close, 12).iloc[-1]
    rsi24 = calculate_rsi(close, 24).iloc[-1]
    if any(pd.isna(v) for v in [rsi6, rsi12, rsi24]):
        return None

    # --- MACD ---
    dif, dea, macd_hist = calculate_macd(close)
    dif_now = dif.iloc[-1]
    dea_now = dea.iloc[-1]
    macd_now = macd_hist.iloc[-1]
    dif_prev = dif.iloc[-2]
    dea_prev = dea.iloc[-2]
    macd_prev = macd_hist.iloc[-2]

    # --- KDJ ---
    k_val = d_val = j_val = None
    k_s = d_s = j_s = None
    if has_high and has_low and len(high) >= n and len(low) >= n:
        k_s, d_s, j_s = calculate_kdj(high[:n], low[:n], close)
        if len(k_s) > 1 and not pd.isna(k_s.iloc[-1]):
            k_val, d_val, j_val = k_s.iloc[-1], d_s.iloc[-1], j_s.iloc[-1]

    # --- ATR (stop-loss/position sizing) ---
    atr_val = None
    if has_high and has_low:
        atr_series = calculate_atr(high, low, close, 14)
        if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
            atr_val = float(atr_series.iloc[-1])

    # --- Volume Ratio ---
    vol_ratio = None
    if volume is not None and len(volume) >= 20:
        vol_20_avg = volume.iloc[-20:].mean()
        if vol_20_avg > 0:
            vol_ratio = volume.iloc[-1] / vol_20_avg

    # --- MACD Divergence (key win-rate booster) ---
    divergence = detect_macd_divergence(close, dif, lookback=30)

    # --- Pullback to MA entry ---
    pullback_ma20 = check_pullback_to_ma(close, ma20, tolerance=0.02)

    # --- Support / Resistance for risk-reward ---
    support_20d = low.iloc[-20:].min() if n >= 20 else low.min()
    resistance_20d = high.iloc[-20:].max() if n >= 20 else high.max()

    # ===================================================================
    # Quality Gate
    # ===================================================================
    above_ma20 = current_price > ma20_now
    macd_golden_cross = (dif_prev <= dea_prev and dif_now > dea_now)
    macd_bullish = dif_now > dea_now
    rsi_not_extreme = rsi6 < 82

    # REJECT: top divergence is a strong sell signal
    if divergence == "top_divergence" and rsi6 > 65:
        return None

    if not rsi_not_extreme:
        return None

    has_bottom_div = divergence == "bottom_divergence"
    if not (above_ma20 or macd_golden_cross or macd_bullish or has_bottom_div or pullback_ma20):
        return None

    # ===================================================================
    # SCORING
    # ===================================================================
    trend_score = 0.0    # max 30
    momentum_score = 0.0 # max 25
    volume_score = 0.0   # max 20

    # --- Trend Score (30 pts) ---
    if not pd.isna(ma5_now) and not pd.isna(ma10_now):
        if current_price > ma5_now > ma10_now > ma20_now:
            trend_score += 15
        elif current_price > ma5_now > ma20_now:
            trend_score += 12
        elif current_price > ma20_now:
            trend_score += 8
        elif current_price > ma5_now:
            trend_score += 4

    if ma60_now is not None:
        if current_price > ma60_now:
            trend_score += 8
        elif current_price > ma60_now * 0.95:
            trend_score += 4
    else:
        trend_score += 4

    if len(ma20) >= 6:
        ma20_5d_ago = ma20.iloc[-6]
        if not pd.isna(ma20_5d_ago) and ma20_5d_ago > 0:
            ma20_slope = (ma20_now - ma20_5d_ago) / ma20_5d_ago
            if ma20_slope > 0.01:
                trend_score += 7
            elif ma20_slope > 0:
                trend_score += 4
            elif ma20_slope > -0.005:
                trend_score += 2

    # --- Momentum Score (25 pts) ---
    if macd_golden_cross:
        momentum_score += 12
    elif macd_bullish and macd_now > macd_prev:
        momentum_score += 10
    elif macd_bullish:
        momentum_score += 7
    elif macd_now > macd_prev:
        momentum_score += 4

    if dif_now > 0:
        momentum_score += 5
    elif dif_now > -0.5:
        momentum_score += 2

    if 40 <= rsi6 <= 55:
        momentum_score += 8
    elif 55 < rsi6 <= 65:
        momentum_score += 6
    elif 30 <= rsi6 < 40:
        momentum_score += 5
    elif rsi6 < 30:
        momentum_score += 4
    elif 65 < rsi6 <= 75:
        momentum_score += 3
    else:
        momentum_score += 1

    # --- Volume Score (20 pts) ---
    if vol_ratio is not None:
        if 1.2 <= vol_ratio <= 2.5:
            volume_score += 15
        elif 0.8 <= vol_ratio < 1.2:
            volume_score += 10
        elif 2.5 < vol_ratio <= 4.0:
            volume_score += 8
        elif 0.5 <= vol_ratio < 0.8:
            volume_score += 6
        elif vol_ratio > 4.0:
            volume_score += 3
        else:
            volume_score += 2

        if len(volume) >= 20:
            vol_5_avg = volume.iloc[-5:].mean()
            vol_20_avg2 = volume.iloc[-20:].mean()
            if vol_20_avg2 > 0:
                vol_trend = vol_5_avg / vol_20_avg2
                if 1.1 <= vol_trend <= 2.0:
                    volume_score += 5
                elif 0.9 <= vol_trend < 1.1:
                    volume_score += 3
                else:
                    volume_score += 1
    else:
        volume_score += 10

    tech_total = trend_score + momentum_score + volume_score  # max 75

    # ===================================================================
    # Win-Rate Bonus/Penalty (adjusts tech_total, max +/-10)
    # ===================================================================
    bonus = 0.0
    signal_tags = []

    if has_bottom_div:
        bonus += 8
        signal_tags.append("MACD底背离(强)")

    if k_val is not None and d_val is not None and k_s is not None and d_s is not None and len(k_s) >= 2:
        if k_s.iloc[-2] <= d_s.iloc[-2] and k_val > d_val and j_val and j_val < 80:
            bonus += 3
            signal_tags.append("KDJ金叉")
        elif k_val > d_val:
            bonus += 1

    if pullback_ma20:
        bonus += 5
        signal_tags.append("回踩MA20")

    bullish_count = sum([
        macd_golden_cross,
        has_bottom_div,
        pullback_ma20,
        above_ma20 and (ma60_now is not None and current_price > ma60_now),
        40 <= rsi6 <= 55,
    ])
    if bullish_count >= 3:
        bonus += 5
        signal_tags.append("多信号共振")
    elif bullish_count >= 2:
        bonus += 2

    if market_regime == "bear":
        bonus -= 8
        signal_tags.append("熊市环境")
    elif market_regime == "bull":
        bonus += 3
        signal_tags.append("牛市环境")

    if rsi6 > 70 and current_price > ma20_now * 1.10:
        bonus -= 5
        signal_tags.append("追高风险")

    if vol_ratio is not None and vol_ratio < 0.7 and current_price > close.iloc[-2]:
        bonus -= 3
        signal_tags.append("缩量上涨")

    tech_total = max(0, min(75, tech_total + bonus))

    if macd_golden_cross:
        signal_tags.append("MACD金叉")
    if trend_score >= 20:
        signal_tags.append("多头排列")
    if vol_ratio and 1.5 <= vol_ratio <= 3.0:
        signal_tags.append("温和放量")
    if 35 <= rsi6 <= 55:
        signal_tags.append("RSI适中")

    # --- Stop-loss & Risk-Reward ---
    stop_loss = None
    risk_reward = None
    if atr_val and atr_val > 0:
        stop_loss = round(current_price - 2 * atr_val, 2)
        potential_gain = resistance_20d - current_price
        potential_loss = current_price - stop_loss
        if potential_loss > 0:
            risk_reward = round(potential_gain / potential_loss, 2)

    return {
        "price": round(current_price, 2),
        "ma5": round(ma5_now, 2) if not pd.isna(ma5_now) else None,
        "ma20": round(ma20_now, 2),
        "rsi6": round(rsi6, 1),
        "rsi12": round(rsi12, 1),
        "rsi24": round(rsi24, 1),
        "dif": round(dif_now, 4),
        "dea": round(dea_now, 4),
        "macd": round(macd_now, 4),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio else None,
        "trend_score": round(trend_score, 1),
        "momentum_score": round(momentum_score, 1),
        "volume_score": round(volume_score, 1),
        "tech_total": round(tech_total, 1),
        "signal_tags": signal_tags,
        "divergence": divergence,
        "pullback_ma20": pullback_ma20,
        "atr": round(atr_val, 2) if atr_val else None,
        "stop_loss": stop_loss,
        "risk_reward": risk_reward,
        "support": round(support_20d, 2),
        "resistance": round(resistance_20d, 2),
    }


def composite_score(tech: dict, pe: float, pb: float,
                    sentiment_score: int = 5) -> float:
    """
    Compute final composite score:
      Technical (max 75) + Valuation (max 15) + Sentiment (max 10) = 100
    """
    tech_total = tech["tech_total"]

    val_score = 0.0
    if pe is not None:
        if 8 <= pe <= 15:
            val_score += 10
        elif 15 < pe <= 25:
            val_score += 8
        elif 25 < pe <= 35:
            val_score += 6
        elif 35 < pe <= 50:
            val_score += 4
        elif 5 <= pe < 8:
            val_score += 7
        elif pe > 50:
            val_score += 2
        else:
            val_score += 1

    if pb is not None:
        if 1.0 <= pb <= 2.5:
            val_score += 5
        elif 2.5 < pb <= 4.0:
            val_score += 4
        elif 0.5 <= pb < 1.0:
            val_score += 3
        elif 4.0 < pb <= 6.0:
            val_score += 2
        else:
            val_score += 1

    sent_score = max(0, min(10, sentiment_score))

    return round(tech_total + val_score + sent_score, 1)


def score_to_rating(score: float) -> str:
    if score >= 80:
        return "★★★★★ 强烈推荐"
    elif score >= 65:
        return "★★★★☆ 推荐关注"
    elif score >= 50:
        return "★★★☆☆ 中性偏多"
    elif score >= 35:
        return "★★☆☆☆ 谨慎参与"
    else:
        return "★☆☆☆☆ 暂不推荐"


# ---------------------------------------------------------------------------
# Report (Text)
# ---------------------------------------------------------------------------

def build_txt_report(
    candidates: list[dict],
    top_n: int,
    total_scanned: int,
    after_fundamental: int,
    after_technical: int,
    market_info: dict | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("A股多因子选股扫描报告 (v3 Server Edition)")
    lines.append(f"生成时间: {now}")
    lines.append("=" * 60)

    if market_info:
        regime_label = {"bull": "牛市", "neutral": "震荡", "bear": "熊市"}.get(
            market_info.get("regime", "neutral"), "震荡")
        lines.append(f"\n大盘环境: {regime_label}  ({market_info.get('reason', '')})")
        if market_info.get("regime") == "bear":
            lines.append("WARNING: 熊市环境下选股标准更严格，建议轻仓操作")
    lines.append("")

    lines.append("筛选条件:")
    lines.append("  基本面: PE(0,80), PB(0,15), 总市值>200亿")
    lines.append("  技术面: MA/MACD/RSI门槛 + 背离检测 + 回踩入场")
    lines.append("  风控: ATR止损 + 盈亏比计算 + 追高惩罚 + 大盘环境调节")
    lines.append("  评分: 趋势(30) + 动量(25) + 量能(20) + 估值(15) + 消息面(10) + 胜率加减分")
    lines.append("")
    lines.append(
        f"统计: 扫描总数={total_scanned}, 基本面通过={after_fundamental}, "
        f"技术面通过={after_technical}"
    )
    lines.append("")

    top = candidates[:top_n]
    if not top:
        lines.append("未找到符合条件的股票。")
        return "\n".join(lines)

    lines.append(f"最推荐的{len(top)}支股票 (按综合评分排序):")
    lines.append("")

    for i, stock in enumerate(top, 1):
        rating = score_to_rating(stock["composite_score"])
        tags = ", ".join(stock.get("signal_tags", [])) if stock.get("signal_tags") else "无特殊信号"
        lines.append(f"{'─' * 55}")
        lines.append(
            f"  {i}. [{stock['code']}] {stock['name']}  "
            f"综合评分: {stock['composite_score']:.1f}/100  {rating}"
        )
        lines.append(
            f"     现价:{stock['price']}  PE:{stock['pe']}  PB:{stock['pb']}  "
            f"总市值:{stock['mkt_cap_yi']}亿"
        )
        sent = stock.get("sentiment", {})
        sent_label = sent.get("label", "中性")
        sent_score = sent.get("score", 5)
        lines.append(
            f"     趋势:{stock['trend_score']}  动量:{stock['momentum_score']}  "
            f"量能:{stock['volume_score']}  估值:{stock['val_score']}  "
            f"消息面:{sent_label}({sent_score}/10)"
        )
        lines.append(
            f"     RSI(6/12/24): {stock['rsi6']}/{stock['rsi12']}/{stock['rsi24']}  "
            f"MACD: {stock['macd']}  量比: {stock.get('vol_ratio', 'N/A')}"
        )

        sl = stock.get('stop_loss')
        rr = stock.get('risk_reward')
        sup = stock.get('support')
        res = stock.get('resistance')
        risk_line = f"     风控: 止损:{sl or 'N/A'}  盈亏比:{rr or 'N/A'}  支撑:{sup or 'N/A'}  压力:{res or 'N/A'}"
        lines.append(risk_line)

        lines.append(f"     信号: {tags}")

        sent_themes = ", ".join(sent.get("key_themes", [])) or "无"
        sent_risks = ", ".join(sent.get("risk_flags", [])) or "暂无"
        sent_summary = sent.get("summary", "")
        if sent_summary:
            lines.append(f"     消息面: {sent_summary}")
        lines.append(f"     关键主题: {sent_themes}  风险提示: {sent_risks}")

        news = stock.get("news_list", [])
        if news:
            lines.append("     近期新闻:")
            for n in news:
                title = n.get("title", "") or "无标题"
                src = n.get("source", "") or "未知来源"
                ts = n.get("time", "") or "未知时间"
                lines.append(f"       - {title} [{src} {ts}]")
        else:
            lines.append("     近期新闻: 暂无")
        lines.append("")

    lines.append("─" * 55)
    lines.append("")
    lines.append("免责声明: 本报告由程序自动生成，仅供参考研究，不构成投资建议。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report (JSON) - for OpenClaw integration
# ---------------------------------------------------------------------------

def build_json_report(
    candidates: list[dict],
    top_n: int,
    total_scanned: int,
    after_fundamental: int,
    after_technical: int,
    market_info: dict | None = None,
) -> dict:
    """Build structured JSON report for programmatic consumption."""
    top = candidates[:top_n]
    stocks = []
    for stock in top:
        stocks.append({
            "code": stock["code"],
            "name": stock["name"],
            "composite_score": stock["composite_score"],
            "rating": score_to_rating(stock["composite_score"]),
            "price": stock["price"],
            "pe": stock["pe"],
            "pb": stock["pb"],
            "mkt_cap_yi": stock["mkt_cap_yi"],
            "scores": {
                "trend": stock["trend_score"],
                "momentum": stock["momentum_score"],
                "volume": stock["volume_score"],
                "valuation": stock["val_score"],
            },
            "rsi": {"rsi6": stock["rsi6"], "rsi12": stock["rsi12"], "rsi24": stock["rsi24"]},
            "macd": stock["macd"],
            "vol_ratio": stock.get("vol_ratio"),
            "risk_management": {
                "stop_loss": stock.get("stop_loss"),
                "risk_reward": stock.get("risk_reward"),
                "support": stock.get("support"),
                "resistance": stock.get("resistance"),
            },
            "signal_tags": stock.get("signal_tags", []),
            "sentiment": stock.get("sentiment", {}),
            "news": stock.get("news_list", []),
        })

    return {
        "report_type": "scanner",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_regime": market_info or {},
        "statistics": {
            "total_scanned": total_scanned,
            "after_fundamental": after_fundamental,
            "after_technical": after_technical,
            "top_n": top_n,
        },
        "recommendations": stocks,
        "disclaimer": "本报告由程序自动生成，仅供参考研究，不构成投资建议。",
    }


def save_report(report_text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"stock_recommendations_{ts}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scanner(top_n: int = 15, pool_path: str = "tech_stock_pool.json",
                output_dir: str = "output", output_json: bool = False) -> dict | str:
    """
    Core scanner logic. Returns JSON dict if output_json=True, else text report string.
    Can be called programmatically or from CLI.
    """
    pool = Path(pool_path)
    if not pool.is_absolute():
        pool = Path(__file__).resolve().parent / pool

    if not pool.exists():
        raise FileNotFoundError(f"股票池文件不存在: {pool}")

    llm_cfg = try_load_llm_config()
    if llm_cfg:
        print("  [INFO] LLM已配置，将启用消息面情感分析", file=sys.stderr)
    else:
        print("  [INFO] 未配置LLM，消息面评分使用默认中性值", file=sys.stderr)

    emit_event("step", current=1, total=7, desc="检测大盘环境")
    print("[1/7] 检测大盘环境...", file=sys.stderr)
    market_info = check_market_regime()
    regime = market_info.get("regime", "neutral")
    regime_label = {"bull": "牛市", "neutral": "震荡", "bear": "熊市"}.get(regime, "震荡")
    print(f"    大盘环境: {regime_label} ({market_info.get('reason', '')})", file=sys.stderr)

    emit_event("step", current=2, total=7, desc="加载股票池并获取实时行情")
    print("[2/7] 加载股票池并获取实时行情...", file=sys.stderr)
    codes = load_stock_codes(pool)
    if not codes:
        raise ValueError("股票池为空")

    quotes = fetch_pool_quotes(codes)
    if quotes is None or quotes.empty:
        raise RuntimeError("无法获取行情数据")

    total_scanned = len(quotes)
    print(f"    扫描股票数: {total_scanned}", file=sys.stderr)

    emit_event("step", current=3, total=7, desc="基本面筛选")
    print("[3/7] 基本面筛选...", file=sys.stderr)
    fundamentals = filter_fundamentals(quotes)
    after_fundamental = len(fundamentals)
    print(f"    基本面通过: {after_fundamental}", file=sys.stderr)

    if fundamentals.empty:
        if output_json:
            return build_json_report([], top_n, total_scanned, 0, 0, market_info=market_info)
        report_text = build_txt_report([], top_n, total_scanned, 0, 0, market_info=market_info)
        save_report(report_text, Path(__file__).resolve().parent / output_dir)
        return report_text

    emit_event("step", current=4, total=7, desc="技术面分析与多因子评分")
    print("[4/7] 技术面分析 & 多因子评分...", file=sys.stderr)
    fundamentals = fundamentals.sort_values("mkt_cap", ascending=False).reset_index(drop=True)

    candidates: list[dict] = []
    checked = 0
    bs_login()
    try:
        for _, row in fundamentals.iterrows():
            checked += 1
            code = str(row["code"])
            name = str(row["name"])
            print(f"\r    进度: {checked}/{len(fundamentals)} - {code} {name}",
                  end="", file=sys.stderr)

            tech = check_technicals(code, market_regime=regime)
            if tech is None:
                continue

            pe_val = round(float(row["pe"]), 2)
            pb_val = round(float(row["pb"]), 2)
            total_score = composite_score(tech, pe_val, pb_val)
            val_score = round(total_score - tech["tech_total"], 1)

            if regime == "bear" and total_score < 55:
                continue

            candidates.append(
                {
                    "code": code,
                    "name": name,
                    "price": tech["price"],
                    "pe": pe_val,
                    "pb": pb_val,
                    "mkt_cap_yi": round(float(row["mkt_cap"]) / 1e8, 1),
                    "rsi6": tech["rsi6"],
                    "rsi12": tech["rsi12"],
                    "rsi24": tech["rsi24"],
                    "dif": tech["dif"],
                    "dea": tech["dea"],
                    "macd": tech["macd"],
                    "vol_ratio": tech["vol_ratio"],
                    "trend_score": tech["trend_score"],
                    "momentum_score": tech["momentum_score"],
                    "volume_score": tech["volume_score"],
                    "val_score": val_score,
                    "composite_score": total_score,
                    "signal_tags": tech["signal_tags"],
                    "stop_loss": tech.get("stop_loss"),
                    "risk_reward": tech.get("risk_reward"),
                    "support": tech.get("support"),
                    "resistance": tech.get("resistance"),
                    "news_list": [],
                }
            )
            time.sleep(0.03)
    finally:
        bs_logout()
        print("", file=sys.stderr)

    after_technical = len(candidates)
    print(f"    技术面通过: {after_technical}", file=sys.stderr)

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    # --- Step 5: Fetch news for top candidates ---
    emit_event("step", current=5, total=7, desc="获取候选股票新闻、公告、研报")
    n_candidates = min(len(candidates), top_n * 2)
    top_candidates = candidates[:n_candidates]
    print(f"[5/7] 获取前{n_candidates}只候选股票的新闻、公告、研报...", file=sys.stderr)

    stocks_data: dict[str, dict] = {}
    for idx, stock in enumerate(top_candidates):
        code = stock["code"]
        name = stock["name"]
        print(f"\r    进度: {idx + 1}/{n_candidates} - {code} {name}",
              end="", file=sys.stderr)
        news = fetch_stock_news(code, count=5)
        notices = fetch_stock_notices(code, count=3)
        reports = fetch_institute_recommendations(code, count=3)
        stock["news_list"] = news
        stock["notices"] = notices
        stock["reports"] = reports
        stocks_data[code] = {
            "name": name,
            "news": news,
            "notices": notices,
            "reports": reports,
        }
        time.sleep(0.1)
    print("", file=sys.stderr)

    # --- Step 6: LLM batch sentiment analysis ---
    emit_event("step", current=6, total=7, desc="LLM消息面情感分析")
    print("[6/7] LLM消息面情感分析...", file=sys.stderr)
    sentiment_results = analyze_sentiment_batch(llm_cfg, stocks_data)

    for stock in top_candidates:
        code = stock["code"]
        sentiment = sentiment_results.get(code, default_sentiment())
        stock["sentiment"] = sentiment
        sent_score = sentiment.get("score", 5)
        tech_data = {
            "tech_total": stock["trend_score"] + stock["momentum_score"] + stock["volume_score"],
        }
        stock["composite_score"] = composite_score(
            tech_data, stock["pe"], stock["pb"], sentiment_score=sent_score
        )
        stock["val_score"] = round(
            stock["composite_score"] - tech_data["tech_total"] - max(0, min(10, sent_score)), 1
        )

    for stock in candidates[n_candidates:]:
        stock["sentiment"] = default_sentiment()

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"    消息面分析完成，已更新{n_candidates}只股票的评分", file=sys.stderr)

    # --- Step 7: Generate report ---
    emit_event("step", current=7, total=7, desc="生成报告")
    print("[7/7] 生成报告...", file=sys.stderr)

    if output_json:
        result = build_json_report(
            candidates=candidates,
            top_n=top_n,
            total_scanned=total_scanned,
            after_fundamental=after_fundamental,
            after_technical=after_technical,
            market_info=market_info,
        )
        # Also save text report
        report_text = build_txt_report(
            candidates=candidates,
            top_n=top_n,
            total_scanned=total_scanned,
            after_fundamental=after_fundamental,
            after_technical=after_technical,
            market_info=market_info,
        )
        save_report(report_text, Path(__file__).resolve().parent / output_dir)
        return result
    else:
        report_text = build_txt_report(
            candidates=candidates,
            top_n=top_n,
            total_scanned=total_scanned,
            after_fundamental=after_fundamental,
            after_technical=after_technical,
            market_info=market_info,
        )
        save_report(report_text, Path(__file__).resolve().parent / output_dir)
        return report_text


def main():
    parser = argparse.ArgumentParser(description="A-share multi-factor scanner (headless server)")
    parser.add_argument("--top", type=int, default=15, help="输出最推荐的前N支股票（默认: 15）")
    parser.add_argument("--pool", type=str, default="tech_stock_pool.json", help="股票池JSON文件路径")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录（默认: output）")
    parser.add_argument("--json", action="store_true", help="输出JSON格式（供OpenClaw等程序解析）")
    args = parser.parse_args()

    result = run_scanner(
        top_n=args.top,
        pool_path=args.pool,
        output_dir=args.output_dir,
        output_json=args.json,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
