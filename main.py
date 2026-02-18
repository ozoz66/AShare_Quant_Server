#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AShare Quant Server - Unified CLI Entry Point

Designed for headless server deployment (e.g., Tencent Cloud) and
programmatic invocation by LLM agents (e.g., OpenClaw).

Subcommands:
  scan     - Batch multi-factor stock screening
  analyze  - Single stock deep analysis

All subcommands support --json flag for structured JSON output.

Usage:
  python main.py scan --top 10 --json
  python main.py analyze --symbol 600519 --json
  
Custom weights example:
  python main.py scan --top 10 --trend-weight 0.4 --momentum-weight 0.2
"""

import argparse
import json
import sys

from indicators import configure_stdio

configure_stdio()


def cmd_scan(args):
    """Execute batch stock screening."""
    from scanner import run_scanner
    
    # Build weights dict from arguments
    weights = {}
    if args.trend_weight is not None:
        weights['trend'] = args.trend_weight
    if args.momentum_weight is not None:
        weights['momentum'] = args.momentum_weight
    if args.volume_weight is not None:
        weights['volume'] = args.volume_weight
    if args.valuation_weight is not None:
        weights['valuation'] = args.valuation_weight
    if args.sentiment_weight is not None:
        weights['sentiment'] = args.sentiment_weight
    
    result = run_scanner(
        top_n=args.top,
        pool_path=args.pool,
        output_dir=args.output_dir,
        output_json=args.json,
        weights=weights if weights else None,
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
    # Custom scoring weights
    p_scan.add_argument("--trend-weight", type=float, default=None, help="Trend factor weight (default: 0.30)")
    p_scan.add_argument("--momentum-weight", type=float, default=None, help="Momentum factor weight (default: 0.25)")
    p_scan.add_argument("--volume-weight", type=float, default=None, help="Volume factor weight (default: 0.20)")
    p_scan.add_argument("--valuation-weight", type=float, default=None, help="Valuation factor weight (default: 0.15)")
    p_scan.add_argument("--sentiment-weight", type=float, default=None, help="Sentiment factor weight (default: 0.10)")
    p_scan.set_defaults(func=cmd_scan)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Single stock deep analysis")
    p_analyze.add_argument("--symbol", type=str, required=True, help="Stock code or name (e.g., 600519 or 贵州茅台)")
    p_analyze.add_argument("--json", action="store_true", help="Output JSON format")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
