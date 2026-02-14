#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared LLM client utilities for this project.

Supports OpenAI-compatible chat completion APIs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMConfig:
    endpoint: str
    api_key: str
    model: str
    timeout: int = 120


def _normalize_endpoint(url: str) -> str:
    """
    Normalize user-provided URL to a chat-completions endpoint.
    If a full endpoint is already provided, keep it unchanged.
    """
    clean = (url or "").strip().rstrip("/")
    if not clean:
        return clean
    if clean.endswith("/chat/completions") or clean.endswith("/completions"):
        return clean
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return data


def load_llm_config(
    config_path: Path | None = None,
    llm_url: str | None = None,
    llm_key: str | None = None,
    llm_model: str | None = None,
    timeout: int | None = None,
) -> LLMConfig:
    """
    Load LLM config with priority:
    1) CLI args
    2) environment variables
    3) config file
    """
    file_data = _read_json(config_path) if config_path else {}

    def pick(*values: Any) -> str | None:
        for v in values:
            if v is not None and str(v).strip():
                return str(v).strip()
        return None

    url = pick(
        llm_url,
        os.getenv("LLM_URL"),
        os.getenv("OPENAI_BASE_URL"),
        file_data.get("llm_url"),
        file_data.get("base_url"),
    )
    key = pick(
        llm_key,
        os.getenv("LLM_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        file_data.get("llm_api_key"),
        file_data.get("api_key"),
    )
    model = pick(
        llm_model,
        os.getenv("LLM_MODEL"),
        os.getenv("OPENAI_MODEL"),
        file_data.get("llm_model"),
        file_data.get("model"),
    )
    conf_timeout = int(
        pick(
            timeout,
            os.getenv("LLM_TIMEOUT"),
            file_data.get("timeout"),
            120,
        )
    )

    if not url:
        raise ValueError("Missing LLM URL. Set --llm-url, LLM_URL, or llm_config.json")
    if not key:
        raise ValueError("Missing LLM API key. Set --llm-key, LLM_API_KEY, or llm_config.json")
    if not model:
        raise ValueError("Missing LLM model. Set --llm-model, LLM_MODEL, or llm_config.json")

    return LLMConfig(
        endpoint=_normalize_endpoint(url),
        api_key=key,
        model=model,
        timeout=conf_timeout,
    )


def chat_completion(
    config: LLMConfig,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    resp = requests.post(
        config.endpoint,
        headers=headers,
        json=payload,
        timeout=config.timeout,
    )
    if resp.status_code >= 400:
        body = resp.text[:1000].replace("\n", " ")
        raise RuntimeError(f"LLM request failed: HTTP {resp.status_code} - {body}")

    data = resp.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            if parts:
                return "\n".join(parts).strip()
        text = choices[0].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    raise RuntimeError("LLM response format not recognized: missing text content.")
