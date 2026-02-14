#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AShare Quant Server - Unified CLI Entry Point

Designed for headless server deployment (e.g., Tencent Cloud) and
programmatic invocation by LLM agents (e.g., OpenClaw).

Subcommands:
  scan     - Batch multi-factor stock screening
  analyze  - Single stock deep analysis
  report   - Combined scanner + analyzer + LLM final report

All subcommands support --json flag for structured JSON output.

Usage:
  python main.py scan --top 10 --json
  python main.py analyze --symbol 600519 --json
  python main.py report --symbol 600519 --top 10 --json
"""

import argparse
import json
import sys

from indicators import configure_stdio

configure_stdio()


def cmd_scan(args):
    """Execute batch stock screening."""
    from scanner import run_scanner
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


def cmd_analyze(args):
    """Execute single stock deep analysis."""
    from analyzer import run_analyzer
    if not args.symbol:
        print("ERROR: --symbol is required for analyze command.", file=sys.stderr)
        sys.exit(1)
    result = run_analyzer(args.symbol, output_json=args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


def cmd_report(args):
    """Execute full pipeline: scan + analyze + LLM summary."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent
    script = root / "llm_final_report.py"

    cmd = [sys.executable, str(script)]
    if args.symbol:
        cmd.extend(["--symbol", args.symbol])
    cmd.extend(["--top", str(args.top)])
    cmd.extend(["--pool", args.pool])
    cmd.extend(["--output-dir", args.output_dir])
    if args.json:
        cmd.append("--json")

    proc = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="ashare-quant",
        description="AShare Quant Server - A-Share Quantitative Trading Toolkit (Headless Server Edition)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- scan ---
    p_scan = subparsers.add_parser("scan", help="Batch multi-factor stock screening")
    p_scan.add_argument("--top", type=int, default=15, help="Top N results (default: 15)")
    p_scan.add_argument("--pool", type=str, default="tech_stock_pool.json", help="Stock pool JSON path")
    p_scan.add_argument("--output-dir", type=str, default="output", help="Output directory")
    p_scan.add_argument("--json", action="store_true", help="Output JSON format")
    p_scan.set_defaults(func=cmd_scan)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Single stock deep analysis")
    p_analyze.add_argument("--symbol", type=str, required=True, help="Stock code or name (e.g., 600519 or 贵州茅台)")
    p_analyze.add_argument("--json", action="store_true", help="Output JSON format")
    p_analyze.set_defaults(func=cmd_analyze)

    # --- report ---
    p_report = subparsers.add_parser("report", help="Full pipeline: scan + analyze + LLM summary")
    p_report.add_argument("--symbol", type=str, default="", help="Stock code (auto-extracted from scan if empty)")
    p_report.add_argument("--top", type=int, default=10, help="Top N for scanner")
    p_report.add_argument("--pool", type=str, default="tech_stock_pool.json", help="Stock pool JSON path")
    p_report.add_argument("--output-dir", type=str, default="output", help="Output directory")
    p_report.add_argument("--json", action="store_true", help="Output JSON format")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
