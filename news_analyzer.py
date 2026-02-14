#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News & sentiment analysis module.

Provides:
  - Multi-source news data fetching (news, announcements, research reports)
  - LLM-based sentiment analysis (single stock & batch)
  - Graceful fallback when LLM is unavailable
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from indicators import to_float, pick_col


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

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
# LLM Sentiment Analysis
# ---------------------------------------------------------------------------

def default_sentiment() -> dict:
    """Return neutral sentiment (used when LLM is unavailable or data is empty)."""
    return {
        "score": 5,
        "label": "中性",
        "key_themes": [],
        "risk_flags": [],
        "summary": "消息面数据不足",
    }


def _build_sentiment_prompt(
    news: list[dict],
    notices: list[dict] | None = None,
    reports: list[dict] | None = None,
) -> str:
    """Build the user prompt for single-stock sentiment analysis."""
    sections = []

    if news:
        lines = []
        for i, item in enumerate(news[:8], 1):
            t = item.get("time", "")
            lines.append(f"{i}. {item['title']} ({t})")
        sections.append("【近期新闻】\n" + "\n".join(lines))

    if notices:
        lines = []
        for i, item in enumerate(notices[:5], 1):
            t = item.get("time", "")
            lines.append(f"{i}. {item['title']} ({t})")
        sections.append("【公司公告】\n" + "\n".join(lines))

    if reports:
        lines = []
        for i, item in enumerate(reports[:5], 1):
            src = item.get("source", "")
            rating = item.get("rating", "")
            tp = item.get("target_price")
            tp_str = f" 目标价{tp}" if tp else ""
            lines.append(f"{i}. {src}: {rating}{tp_str} ({item.get('time', '')})")
        sections.append("【机构评级】\n" + "\n".join(lines))

    if not sections:
        return ""

    return "\n\n".join(sections)


_SENTIMENT_SYSTEM_PROMPT = (
    "你是A股消息面分析专家。请分析给定股票的近期消息，给出情绪判断。\n"
    "请严格以JSON格式输出（不要输出其他内容）：\n"
    '{"score": <0到10的整数, 5为中性, 越高越利好>, '
    '"label": "<利好/中性/利空>", '
    '"key_themes": ["<关键主题>"], '
    '"risk_flags": ["<风险提示>"], '
    '"summary": "<一句话总结消息面>"}'
)


def _parse_sentiment_json(text: str) -> dict | None:
    """Try to parse LLM response as sentiment JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            score = int(data["score"])
            data["score"] = max(0, min(10, score))
            if "label" not in data:
                data["label"] = "利好" if score >= 7 else ("利空" if score <= 3 else "中性")
            if "key_themes" not in data:
                data["key_themes"] = []
            if "risk_flags" not in data:
                data["risk_flags"] = []
            if "summary" not in data:
                data["summary"] = ""
            return data
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    m = re.search(r'"score"\s*:\s*(\d+)', text)
    if m:
        score = max(0, min(10, int(m.group(1))))
        label_m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        summary_m = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
        return {
            "score": score,
            "label": label_m.group(1) if label_m else ("利好" if score >= 7 else ("利空" if score <= 3 else "中性")),
            "key_themes": [],
            "risk_flags": [],
            "summary": summary_m.group(1) if summary_m else "",
        }

    return None


def analyze_sentiment(
    llm_config,
    news: list[dict],
    notices: list[dict] | None = None,
    reports: list[dict] | None = None,
) -> dict:
    """
    Analyze sentiment for a single stock using LLM.

    Args:
        llm_config: LLMConfig or None. If None, returns neutral default.
        news: News items list.
        notices: Company announcements list.
        reports: Research report/rating list.

    Returns:
        Sentiment dict with score (0-10), label, key_themes, risk_flags, summary.
    """
    if llm_config is None:
        return default_sentiment()

    content = _build_sentiment_prompt(news, notices, reports)
    if not content:
        return default_sentiment()

    from llm_client import chat_completion

    try:
        response = chat_completion(
            config=llm_config,
            messages=[
                {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        result = _parse_sentiment_json(response)
        if result:
            return result
        print(f"  [WARN] LLM情感分析返回格式异常，使用中性默认值", file=sys.stderr)
        return default_sentiment()
    except Exception as e:
        print(f"  [WARN] LLM情感分析失败: {e}，使用中性默认值", file=sys.stderr)
        return default_sentiment()


# ---------------------------------------------------------------------------
# Batch Sentiment Analysis (for scanner)
# ---------------------------------------------------------------------------

_BATCH_SYSTEM_PROMPT = (
    "你是A股消息面分析专家。请对以下多只股票的消息面分别给出情绪评分。\n"
    "请严格以JSON格式输出（不要输出其他内容），每只股票一个条目：\n"
    '{"<股票代码>": {"score": <0-10整数>, "label": "<利好/中性/利空>", "summary": "<一句话>"}, ...}'
)


def analyze_sentiment_batch(
    llm_config,
    stocks_data: dict[str, dict],
) -> dict[str, dict]:
    """
    Batch sentiment analysis for multiple stocks in a single LLM call.

    Args:
        llm_config: LLMConfig or None.
        stocks_data: {code: {"name": str, "news": [...], "notices": [...], "reports": [...]}}

    Returns:
        {code: sentiment_result, ...}
    """
    if llm_config is None or not stocks_data:
        return {code: default_sentiment() for code in stocks_data}

    sections = []
    for code, data in stocks_data.items():
        name = data.get("name", code)
        header = f"=== {name} ({code}) ==="
        news = data.get("news", [])
        notices = data.get("notices", [])
        reports = data.get("reports", [])

        lines = [header]
        if news:
            news_str = "; ".join(item["title"] for item in news[:3])
            lines.append(f"新闻: {news_str}")
        if notices:
            notice_str = "; ".join(item["title"] for item in notices[:2])
            lines.append(f"公告: {notice_str}")
        if reports:
            report_str = "; ".join(
                f"{item.get('source', '')}:{item.get('rating', '')}" for item in reports[:3]
            )
            lines.append(f"评级: {report_str}")
        if not news and not notices and not reports:
            lines.append("暂无消息")

        sections.append("\n".join(lines))

    content = "\n\n".join(sections)

    from llm_client import chat_completion

    try:
        response = chat_completion(
            config=llm_config,
            messages=[
                {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        result = _parse_batch_response(response, list(stocks_data.keys()))
        return result
    except Exception as e:
        print(f"  [WARN] LLM批量情感分析失败: {e}，使用中性默认值", file=sys.stderr)
        return {code: default_sentiment() for code in stocks_data}


def _parse_batch_response(text: str, codes: list[str]) -> dict[str, dict]:
    """Parse batch sentiment LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    result = {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for code in codes:
                entry = data.get(code, {})
                if isinstance(entry, dict) and "score" in entry:
                    score = max(0, min(10, int(entry["score"])))
                    result[code] = {
                        "score": score,
                        "label": entry.get("label", "中性"),
                        "key_themes": entry.get("key_themes", []),
                        "risk_flags": entry.get("risk_flags", []),
                        "summary": entry.get("summary", ""),
                    }
                else:
                    result[code] = default_sentiment()
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    for code in codes:
        if code not in result:
            result[code] = default_sentiment()

    return result


# ---------------------------------------------------------------------------
# LLM Config Helper
# ---------------------------------------------------------------------------

def try_load_llm_config():
    """
    Try to load LLM config without crashing.
    Returns LLMConfig or None.
    """
    try:
        from llm_client import load_llm_config
        config_path = Path(__file__).resolve().parent / "llm_config.json"
        return load_llm_config(config_path=config_path)
    except Exception:
        return None
