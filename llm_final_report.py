#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run scanner.py + analyzer.py, then ask one shared LLM to generate final output.

The final output is produced by the LLM, not by rule-based string concatenation.
Supports --json flag for structured output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from llm_client import chat_completion, load_llm_config


DEFAULT_SYSTEM_PROMPT = (
    "你是A股研究助手。请严格基于输入内容总结，输出结构化结论，"
    "包含核心观点、关键依据、风险提示和可执行建议。"
)

DEFAULT_SCANNER_PROMPT = (
    "你将收到市场扫描结果。请先提炼市场层面的候选机会，"
    "重点关注估值、技术指标和新闻催化。"
)

DEFAULT_ANALYZER_PROMPT = (
    "你将收到单只股票深度分析结果。请提炼个股逻辑，"
    "重点关注趋势信号、基本面质量和近期事件。"
)


def _run_python_script(script: Path, args: list[str], timeout: int = 1200, retries: int = 1) -> str:
    cmd = [sys.executable, str(script), *args]
    last_err = None
    attempts = max(1, retries + 1)

    for i in range(attempts):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_err = RuntimeError(f"Command timeout ({timeout}s): {' '.join(cmd)}")
            if i < attempts - 1:
                print(f"[WARN] Script timeout, retrying ({i + 1}/{attempts - 1}) ...", file=sys.stderr)
                continue
            raise last_err

        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        if proc.returncode == 0:
            return proc.stdout.strip()

        last_err = RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        if i < attempts - 1:
            print(f"[WARN] Script failed, retrying ({i + 1}/{attempts - 1}) ...", file=sys.stderr)
            continue

    if last_err:
        raise last_err
    raise RuntimeError(f"Unexpected execution state: {' '.join(cmd)}")


def _pick_symbol_from_scanner_output(scanner_output: str) -> str | None:
    match = re.search(r"\b(\d{6})\b", scanner_output)
    return match.group(1) if match else None


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    return f"{clipped}\n\n[内容过长，已截断，总长度={len(text)}字符]"


def _build_user_prompt(
    scanner_prompt: str,
    analyzer_prompt: str,
    scanner_output: str,
    analyzer_output: str,
) -> str:
    return (
        "请根据以下两份程序输出，给出一份最终综合结论。\n\n"
        "【提示词A：市场扫描】\n"
        f"{scanner_prompt}\n\n"
        "【程序A输出：scanner.py】\n"
        f"{scanner_output}\n\n"
        "【提示词B：个股深度分析】\n"
        f"{analyzer_prompt}\n\n"
        "【程序B输出：analyzer.py】\n"
        f"{analyzer_output}\n\n"
        "输出要求：\n"
        "1. 先给结论摘要\n"
        "2. 再给关键证据点（分条）\n"
        "3. 最后给风险与免责声明（必须包含不构成投资建议）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine scanner+analyzer outputs into final LLM answer (headless server)."
    )
    parser.add_argument("--symbol", type=str, default="", help="分析股票代码/名称；为空时从scanner输出自动提取")
    parser.add_argument("--top", type=int, default=10, help="scanner.py --top 参数")
    parser.add_argument("--pool", type=str, default="tech_stock_pool.json", help="scanner.py --pool 参数")
    parser.add_argument("--output-dir", type=str, default="output", help="最终LLM输出目录")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")

    parser.add_argument("--llm-config", type=str, default="llm_config.json", help="LLM配置文件路径")
    parser.add_argument("--llm-url", type=str, default=None, help="LLM URL")
    parser.add_argument("--llm-key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM模型名")
    parser.add_argument("--llm-timeout", type=int, default=120, help="LLM超时秒数")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=1200, help="LLM max_tokens")
    parser.add_argument("--max-input-chars", type=int, default=12000, help="每个程序输出的最大字符数")
    parser.add_argument("--script-timeout", type=int, default=1200, help="子进程超时秒数")
    parser.add_argument("--script-retries", type=int, default=1, help="失败重试次数")

    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--scanner-prompt", type=str, default=DEFAULT_SCANNER_PROMPT)
    parser.add_argument("--analyzer-prompt", type=str, default=DEFAULT_ANALYZER_PROMPT)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    scanner_script = root / "scanner.py"
    analyzer_script = root / "analyzer.py"
    if not scanner_script.exists() or not analyzer_script.exists():
        raise FileNotFoundError("scanner.py or analyzer.py not found in current project directory.")

    print("[1/5] Running scanner.py ...", file=sys.stderr)
    scanner_output = _run_python_script(
        scanner_script,
        ["--top", str(args.top), "--pool", args.pool],
        timeout=args.script_timeout,
        retries=args.script_retries,
    )

    symbol = args.symbol.strip() or _pick_symbol_from_scanner_output(scanner_output)
    if not symbol:
        raise RuntimeError("No symbol provided and failed to extract 6-digit code from scanner output.")
    print(f"[2/5] Running analyzer.py for symbol: {symbol}", file=sys.stderr)
    analyzer_output = _run_python_script(
        analyzer_script,
        ["--symbol", symbol],
        timeout=args.script_timeout,
        retries=args.script_retries,
    )

    print("[3/5] Loading shared LLM config ...", file=sys.stderr)
    config = load_llm_config(
        config_path=(root / args.llm_config),
        llm_url=args.llm_url,
        llm_key=args.llm_key,
        llm_model=args.llm_model,
        timeout=args.llm_timeout,
    )

    scanner_for_llm = _clip(scanner_output, args.max_input_chars)
    analyzer_for_llm = _clip(analyzer_output, args.max_input_chars)
    user_prompt = _build_user_prompt(
        scanner_prompt=args.scanner_prompt,
        analyzer_prompt=args.analyzer_prompt,
        scanner_output=scanner_for_llm,
        analyzer_output=analyzer_for_llm,
    )

    print("[4/5] Calling LLM for final output ...", file=sys.stderr)
    final_text = chat_completion(
        config=config,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"llm_final_report_{ts}.txt"
    out_path.write_text(final_text, encoding="utf-8")

    print("[5/5] Done.", file=sys.stderr)

    if args.json:
        result = {
            "report_type": "final_report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "content": final_text,
            "output_path": str(out_path),
            "disclaimer": "本报告由程序自动生成，仅供参考研究，不构成投资建议。",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(final_text)

    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
