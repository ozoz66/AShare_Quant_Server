# OpenClaw Agent Tools Definition / OpenClaw Agent 工具定义

## Overview / 概述

- English: This document defines the tools available for LLM agents (such as OpenClaw) to interact with the AShare Quant Server. All tools are invoked via CLI commands and return structured output (text or JSON). The server handles data acquisition and quantitative scoring; LLM-based reasoning and report generation should be performed by the calling agent.
- 中文：本文档定义了 LLM Agent（如 OpenClaw）与 AShare Quant Server 交互的可用工具。所有工具通过 CLI 命令调用，返回结构化输出（文本或 JSON）。服务器负责数据获取和量化评分；LLM 推理和报告生成由调用方 Agent 完成。

## Important Notes / 重要说明

- English: All tools run **headlessly** on the server. No GUI or browser is needed. Use `--json` flag for machine-readable output. No LLM API configuration is needed on the server side — sentiment analysis uses rule-based scoring.
- 中文：所有工具在服务器上**无头运行**，无需 GUI 或浏览器。使用 `--json` 标志获取机器可读输出。服务端无需配置 LLM API — 消息面分析使用规则化评分。

---

## Tool 1: `market_scanner`

### Basics / 基本信息

- English: Scans a stock pool in batch and ranks candidates with fundamentals + technicals + rule-based sentiment.
- 中文：批量扫描股票池，执行基本面 + 技术面 + 规则消息面评分并排序。
- Script / 脚本: `scanner.py` or `python main.py scan`

### Typical Intents / 典型触发意图

- English: "Scan market opportunities", "Recommend stocks", "Run a screening pass", "What stocks look good today?"
- 中文："扫一下市场机会""推荐几只股票""按池子跑一次策略""今天什么股票好?"

### Command Templates / 命令模板

```bash
# Recommended: use main.py unified entry
python main.py scan --top 10 --json

# Direct script invocation
python scanner.py --top 10 --json
python scanner.py --top 15 --pool tech_stock_pool.json
```

### Parameters / 参数说明

| Parameter | Description (EN) | 说明 (CN) | Default |
|-----------|-----------------|-----------|---------|
| `--top` | Number of top results | 输出前 N 只 | `15` |
| `--pool` | Stock pool JSON path | 股票池 JSON 路径 | `tech_stock_pool.json` |
| `--output-dir` | Report output directory | 报告输出目录 | `output` |
| `--json` | Output JSON format | 输出 JSON 格式 | `false` |

### JSON Output Schema / JSON 输出结构

```json
{
  "report_type": "scanner",
  "generated_at": "2026-02-15 10:30:00",
  "market_regime": { "regime": "bull|neutral|bear", "reason": "..." },
  "statistics": {
    "total_scanned": 120,
    "after_fundamental": 85,
    "after_technical": 25,
    "top_n": 10
  },
  "recommendations": [
    {
      "code": "600519",
      "name": "贵州茅台",
      "composite_score": 78.53,
      "rating": "★★★★☆ 推荐关注",
      "price": 1800.0,
      "pe": 30.5,
      "pb": 8.2,
      "scores": { "trend": 25.32, "momentum": 20.14, "volume": 15.07, "valuation": 8.0 },
      "signal_tags": ["MACD金叉", "多头排列"],
      "risk_management": { "stop_loss": 1720, "risk_reward": 1.8, "support": 1750, "resistance": 1850 },
      "sentiment": { "score": 7, "label": "利好", "summary": "机构评级: 买入/增持; 利好: 业绩增长" }
    }
  ],
  "disclaimer": "..."
}
```

### Output Interpretation / 输出解读

- English: Start with scan overview (total, fundamentals pass, technicals pass), then highlight top 3-5 picks with key reasons, signal tags, and risk management data. Scores use continuous calculation with 2 decimal places for maximum differentiation.
- 中文：先给扫描概览（总数、基本面通过、技术面通过），再列前 3-5 只候选及核心理由、信号标签和风控数据。评分采用连续化计算，保留2位小数以最大化区分度。

---

## Tool 2: `stock_analyzer`

### Basics / 基本信息

- English: Performs deep analysis for a single stock, including quote, technicals, fundamentals, news/notices, and trade advice.
- 中文：对单只股票进行深度分析，输出行情、技术面、基本面、新闻公告与交易建议。
- Script / 脚本: `analyzer.py` or `python main.py analyze`

### Typical Intents / 典型触发意图

- English: "Analyze 600519", "How is Moutai doing?", "Should I buy this stock?", "Deep dive on 002594"
- 中文："分析 600519""看看贵州茅台现在怎么样""这只股票能不能买""深度分析比亚迪"

### Command Templates / 命令模板

```bash
# Recommended: use main.py unified entry
python main.py analyze --symbol 600519 --json
python main.py analyze --symbol 贵州茅台 --json

# Direct script invocation
python analyzer.py --symbol 600519 --json
python analyzer.py 贵州茅台
```

### Parameters / 参数说明

| Parameter | Description (EN) | 说明 (CN) | Required |
|-----------|-----------------|-----------|----------|
| `--symbol` | Stock code or name | 股票代码或名称 | **Yes** |
| `--json` | Output JSON format | 输出 JSON 格式 | No |

### Input Extraction Rules / 参数提取规则

- English: Prefer 6-digit ticker parsing; fall back to name/keyword matching when no code is provided.
- 中文：优先识别 6 位数字代码；无代码时按名称/关键词匹配。
- English: For follow-up queries after `market_scanner`, prioritize ticker codes from scanner results.
- 中文：如果来自 `market_scanner` 追问，优先使用候选列表里的代码。

### JSON Output Schema / JSON 输出结构

```json
{
  "report_type": "analyzer",
  "generated_at": "2026-02-15 10:35:00",
  "stock": { "code": "600519", "name": "贵州茅台" },
  "market_regime": { "regime": "neutral", "reason": "..." },
  "quote": { "price": 1800, "change_pct": 1.5, "pe": 30.5, "pb": 8.2, "..." : "..." },
  "fundamentals": { "行业": "白酒", "上市日期": "2001-08-27", "..." : "..." },
  "financial_indicators": { "ROE": 25.3, "毛利率": 91.5, "净利率": 52.1 },
  "technicals": {
    "score": 72.5,
    "rating": "★★★★☆ 偏多",
    "signals": ["MACD 多头运行", "均线多头排列", "..."],
    "ma5": 1795, "ma20": 1780, "rsi6": 58.3,
    "stop_loss": 1720, "risk_reward": 1.8
  },
  "news": [ { "title": "...", "source": "...", "time": "..." } ],
  "sentiment": { "score": 7, "label": "利好", "summary": "机构评级: 买入; 利好: 业绩增长" },
  "trade_advice": {
    "action": "积极持有",
    "target_price": 2070,
    "score": 2,
    "reasons": ["技术综合评分72.5分 (偏多)", "ROE=25.3% 优秀"]
  },
  "disclaimer": "..."
}
```

### Output Interpretation / 输出解读

- English: Give a one-line conclusion first, then evidence by technicals/fundamentals/sentiment, and end with risk reminders.
- 中文：先给一句总判断，再分技术面、基本面、消息面说明依据，最后给风险提示。

---

## Orchestration Strategy / 协作策略

1. **Default Flow / 默认流程**: `market_scanner` -> pick top candidates -> `stock_analyzer` on each
2. **Single Stock Query / 单一标的**: Call `stock_analyzer` directly
3. **Quick Market Overview / 快速市场概览**: Call `market_scanner` only
4. **Full Summary / 完整总结**: Call both tools, then use OpenClaw's own LLM to synthesize final conclusions

### Decision Tree / 决策树

```
User Intent
├── "扫描/推荐/选股/市场机会" -> market_scanner (scan)
├── "分析/看看/这只股票" -> stock_analyzer (analyze)
├── "最终结论/完整报告/汇总" -> market_scanner + stock_analyzer, then OpenClaw LLM synthesis
└── Ambiguous -> market_scanner first, then stock_analyzer on top pick
```

### LLM Synthesis Notes / LLM 综合分析说明

- English: When the user requests a comprehensive report, OpenClaw should: (1) run `market_scanner` to get top candidates, (2) run `stock_analyzer` on the top 1-3 picks, (3) use its own LLM reasoning to synthesize a final report combining both outputs. The raw news, announcements, and analyst ratings in the JSON output provide rich context for LLM-based deeper analysis.
- 中文：当用户需要完整报告时，OpenClaw 应：(1) 运行 `market_scanner` 获取候选，(2) 对前 1-3 只运行 `stock_analyzer`，(3) 用自身 LLM 推理能力综合两者输出生成最终报告。JSON 输出中的原始新闻、公告、机构评级数据为 LLM 深度分析提供丰富上下文。

## Risk and Compliance / 风险与合规

- English: All outputs must include risk reminders and avoid guaranteed-return wording. Never expose API keys or other sensitive configuration values.
- 中文：所有输出必须保留风险提示，不得表述为确定性收益承诺。严禁泄露 API Key 等敏感配置。

---

Document Version / 文档版本: `5.0`
Last Updated / 最后更新: `2026-02-15`
