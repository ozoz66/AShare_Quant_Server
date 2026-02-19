#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News & sentiment analysis module.

Provides:
  - Multi-source news data fetching (news, announcements, research reports)
  - Rule-based sentiment analysis (no LLM dependency)
  - Raw data output for external LLM agents (e.g., OpenClaw) to analyze
"""

from __future__ import annotations

from datetime import datetime

import akshare as ak

from indicators import to_float, pick_col, retry


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

@retry(max_attempts=3, delay=1.0)
def fetch_stock_news(symbol: str, count: int = 5) -> list[dict]:
    """Fetch recent news for a stock from East Money."""
    try:
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return []
        title_col = pick_col(df, ["新闻标题"])
        time_col = pick_col(df, ["发布时间"])
        source_col = pick_col(df, ["文章来源"])
        if title_col is None:
            return []
        out = []
        for _, row in df.head(count).iterrows():
            out.append({
                "title": str(row.get(title_col, "")).strip(),
                "time": str(row.get(time_col, "")).strip() if time_col else "",
                "source": str(row.get(source_col, "")).strip() if source_col else "",
            })
        return out
    except Exception:
        return []


@retry(max_attempts=3, delay=1.5)
def fetch_stock_notices(symbol: str, count: int = 5) -> list[dict]:
    """Fetch recent company announcements via stock_notice_report."""
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = ak.stock_notice_report(symbol=symbol, date=today)
        if df is None or df.empty:
            return []
        title_col = pick_col(df, ["公告标题", "公告名称"])
        type_col = pick_col(df, ["公告类型", "类型"])
        date_col = pick_col(df, ["公告日期", "日期"])
        if title_col is None:
            return []
        out = []
        for _, row in df.head(count).iterrows():
            out.append({
                "title": str(row.get(title_col, "")).strip(),
                "time": str(row.get(date_col, "")).strip() if date_col else "",
                "source": str(row.get(type_col, "公告")).strip() if type_col else "公告",
            })
        return out
    except Exception:
        return []


@retry(max_attempts=3, delay=1.5)
def fetch_institute_recommendations(symbol: str, count: int = 5) -> list[dict]:
    """Fetch recent analyst ratings/research reports."""
    try:
        df = ak.stock_institute_recommend_detail(symbol=symbol)
        if df is None or df.empty:
            return []
        code_col = pick_col(df, ["股票代码"])
        rating_col = pick_col(df, ["最新评级", "评级"])
        org_col = pick_col(df, ["评级机构", "机构"])
        date_col = pick_col(df, ["评级日期", "日期"])
        target_col = pick_col(df, ["目标价"])
        analyst_col = pick_col(df, ["分析师"])
        if rating_col is None:
            return []
        out = []
        for _, row in df.head(count).iterrows():
            target_price = to_float(row.get(target_col)) if target_col else None
            title_parts = []
            if rating_col:
                title_parts.append(str(row.get(rating_col, "")))
            if target_price:
                title_parts.append(f"目标价{target_price}")
            title = " | ".join(title_parts)
            out.append({
                "title": title,
                "time": str(row.get(date_col, "")).strip() if date_col else "",
                "source": str(row.get(org_col, "")).strip() if org_col else "",
                "rating": str(row.get(rating_col, "")).strip() if rating_col else "",
                "analyst": str(row.get(analyst_col, "")).strip() if analyst_col else "",
                "target_price": target_price,
            })
        return out
    except Exception:
        return []


def fetch_stock_comment(symbol: str) -> dict | None:
    """
    Fetch market comment data for a specific stock.
    Returns institutional participation and composite score.
    """
    try:
        df = ak.stock_comment_em()
        if df is None or df.empty:
            return None
        code_col = pick_col(df, ["代码"])
        if code_col is None:
            return None
        row = df[df[code_col].astype(str) == str(symbol)]
        if row.empty:
            return None
        r = row.iloc[0]
        score_col = pick_col(df, ["综合得分"])
        inst_col = pick_col(df, ["机构参与度"])
        rank_col = pick_col(df, ["目前排名"])
        attention_col = pick_col(df, ["关注指数"])
        return {
            "composite_score": to_float(r.get(score_col)) if score_col else None,
            "institutional_participation": to_float(r.get(inst_col)) if inst_col else None,
            "rank": to_float(r.get(rank_col)) if rank_col else None,
            "attention_index": to_float(r.get(attention_col)) if attention_col else None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rule-Based Sentiment Analysis (no LLM required)
# ---------------------------------------------------------------------------

# Positive keywords in news/announcement titles
_POSITIVE_KEYWORDS = [
    "利好", "大涨", "涨停", "突破", "创新高", "超预期", "业绩增长", "净利润增",
    "营收增", "分红", "回购", "增持", "战略合作", "中标", "获批", "订单",
    "签约", "扩产", "产能", "新产品", "专利", "技术突破", "行业龙头",
    "景气", "复苏", "放量", "资金流入", "北向资金", "机构加仓",
]

_NEGATIVE_KEYWORDS = [
    "利空", "大跌", "跌停", "暴跌", "亏损", "下滑", "下降", "减持",
    "质押", "违规", "处罚", "诉讼", "风险", "警示", "ST", "*ST",
    "退市", "业绩下滑", "净利润降", "营收降", "商誉减值", "计提",
    "资金流出", "股东减持", "高管辞职", "立案调查", "监管",
]

# Analyst rating mapping to score delta
_RATING_SCORES = {
    "买入": 2.0, "强烈推荐": 2.0, "强推": 2.0,
    "增持": 1.5, "推荐": 1.5, "优于大市": 1.5,
    "审慎增持": 1.0, "谨慎增持": 1.0,
    "中性": 0.0, "持有": 0.0, "同步大市": 0.0,
    "减持": -1.5, "卖出": -2.0, "回避": -2.0,
    "弱于大市": -1.5, "不推荐": -2.0,
}


def default_sentiment() -> dict:
    """Return neutral sentiment (used when data is insufficient)."""
    return {
        "score": 5,
        "label": "中性",
        "key_themes": [],
        "risk_flags": [],
        "summary": "消息面数据不足",
    }


def _score_from_keywords(titles: list[str]) -> float:
    """Score news/notice titles by keyword matching. Returns delta from 0."""
    if not titles:
        return 0.0
    pos_count = 0
    neg_count = 0
    for title in titles:
        for kw in _POSITIVE_KEYWORDS:
            if kw in title:
                pos_count += 1
                break
        for kw in _NEGATIVE_KEYWORDS:
            if kw in title:
                neg_count += 1
                break
    total = len(titles)
    if total == 0:
        return 0.0
    # Net sentiment ratio scaled to [-2.5, +2.5]
    return (pos_count - neg_count) / total * 2.5


def _score_from_ratings(reports: list[dict]) -> float:
    """Score from analyst ratings. Returns delta from 0."""
    if not reports:
        return 0.0
    total_delta = 0.0
    counted = 0
    for r in reports:
        rating = r.get("rating", "").strip()
        for key, delta in _RATING_SCORES.items():
            if key in rating:
                total_delta += delta
                counted += 1
                break
    if counted == 0:
        return 0.0
    # Average rating delta, capped to [-2.5, +2.5]
    avg = total_delta / counted
    return max(-2.5, min(2.5, avg))


def analyze_sentiment_rule_based(
    news: list[dict],
    notices: list[dict] | None = None,
    reports: list[dict] | None = None,
) -> dict:
    """
    Rule-based sentiment analysis using keyword matching and analyst ratings.

    Returns sentiment dict with score (0-10), label, key_themes, risk_flags, summary.
    Base score is 5 (neutral), adjusted by news keywords and analyst ratings.
    """
    all_titles = [item.get("title", "") for item in (news or [])]
    all_titles += [item.get("title", "") for item in (notices or [])]

    news_delta = _score_from_keywords(all_titles)
    rating_delta = _score_from_ratings(reports or [])

    # Combine: news keywords + analyst ratings
    # news_delta range: [-2.5, +2.5]
    # rating_delta range: [-2.5, +2.5]
    raw_score = 5.0 + news_delta + rating_delta
    score = max(0, min(10, round(raw_score)))

    # Extract themes and risks
    key_themes = []
    risk_flags = []
    for title in all_titles:
        for kw in _POSITIVE_KEYWORDS[:15]:
            if kw in title:
                if kw not in key_themes:
                    key_themes.append(kw)
                break
        for kw in _NEGATIVE_KEYWORDS[:15]:
            if kw in title:
                if kw not in risk_flags:
                    risk_flags.append(kw)
                break

    # Label
    if score >= 7:
        label = "利好"
    elif score <= 3:
        label = "利空"
    else:
        label = "中性"

    # Summary
    parts = []
    if reports:
        ratings = [r.get("rating", "") for r in reports[:3] if r.get("rating")]
        if ratings:
            parts.append(f"机构评级: {'/'.join(ratings)}")
    if key_themes:
        parts.append(f"利好: {','.join(key_themes[:3])}")
    if risk_flags:
        parts.append(f"风险: {','.join(risk_flags[:3])}")
    summary = "; ".join(parts) if parts else "消息面平淡"

    return {
        "score": score,
        "label": label,
        "key_themes": key_themes[:5],
        "risk_flags": risk_flags[:5],
        "summary": summary,
        "raw_score": round(raw_score, 2),
    }


def analyze_sentiment_batch_rule_based(
    stocks_data: dict[str, dict],
) -> dict[str, dict]:
    """
    Batch rule-based sentiment analysis for multiple stocks.

    Args:
        stocks_data: {code: {"name": str, "news": [...], "notices": [...], "reports": [...]}}

    Returns:
        {code: sentiment_result, ...}
    """
    result = {}
    for code, data in stocks_data.items():
        news = data.get("news", [])
        notices = data.get("notices", [])
        reports = data.get("reports", [])
        result[code] = analyze_sentiment_rule_based(news, notices, reports)
    return result
