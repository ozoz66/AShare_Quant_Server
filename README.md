# AShare Quant Server

**A-Share Quantitative Trading Toolkit - Headless Server Edition**

**A股量化交易工具箱 - 无头服务器版**

---

[中文](#中文说明) | [English](#english)

---

## English

### Overview

AShare Quant Server is a headless, CLI-based quantitative trading toolkit for Chinese A-share stocks. It is designed to run on cloud servers (e.g., Tencent Cloud Lighthouse) without a GUI and to be invoked programmatically by LLM agents such as OpenClaw.

The server focuses on **data acquisition and quantitative scoring**. LLM-based analysis (e.g., final report generation, deep semantic sentiment analysis) is delegated to the calling agent (OpenClaw), which can use the structured JSON output as context for its own LLM reasoning.

### Features

- **Batch Stock Screening** - Multi-factor scoring across a customizable stock pool (fundamentals + technicals + sentiment)
- **Single Stock Deep Analysis** - Comprehensive analysis with quote, technicals, fundamentals, news, and trade advice
- **Rule-Based Sentiment** - Keyword + analyst rating based sentiment scoring (no external LLM dependency)
- **JSON Output Mode** - Structured JSON output (`--json`) for programmatic consumption by LLM agents
- **Unified CLI Entry Point** - Single `main.py` with subcommands: `scan`, `analyze`
- **No GUI Required** - Fully headless; no Streamlit, no browser, no desktop needed
- **Multi-Source Data** - Primary: East Money (akshare); Backup: BaoStock, Tencent Quotes API
- **Risk Management** - ATR-based stop-loss, risk-reward ratio, support/resistance levels
- **Continuous Scoring** - Smooth, continuous scoring with micro-adjustments to minimize tied scores

### Multi-Factor Scoring Model

| Factor | Max Points | Components |
|--------|-----------|------------|
| Trend | 30 | MA arrangement, price vs MA20/MA60, MA slope (continuous) |
| Momentum | 25 | MACD status, RSI zone (continuous), MACD acceleration |
| Volume | 20 | Volume ratio (continuous peak at 1.6x), volume trend |
| Valuation | 15 | PE sweet spot (continuous, peak at PE=12), PB ratio (continuous, peak at PB=1.5) |
| Sentiment | 10 | Rule-based: news keyword analysis + analyst rating scoring |
| Win-Rate Bonus | +/-10 | MACD divergence, pullback support, confluence, market regime |

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Run

```bash
# Batch screening (text output)
python main.py scan --top 10

# Batch screening (JSON output for LLM agents)
python main.py scan --top 10 --json

# Batch screening with custom weights
python main.py scan --top 10 --trend-weight 0.4 --momentum-weight 0.2 --volume-weight 0.2

# Using different stock pools
python main.py scan --top 10 --pool blue_chip_pool.json

# Single stock analysis
python main.py analyze --symbol 600519
python main.py analyze --symbol 贵州茅台 --json
```

**Custom Weights Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trend-weight` | 0.30 | Trend factor weight (30%) |
| `--momentum-weight` | 0.25 | Momentum factor weight (25%) |
| `--volume-weight` | 0.20 | Volume factor weight (20%) |
| `--valuation-weight` | 0.15 | Valuation factor weight (15%) |
| `--sentiment-weight` | 0.10 | Sentiment factor weight (10%) |

**Stock Pools:**
- `tech_stock_pool.json` - Default (~120 tech/stock pool)
- `blue_chip_pool.json` - Blue chip stocks (50 large caps)

You can also call individual scripts directly:

```bash
python scanner.py --top 15 --pool tech_stock_pool.json
python analyzer.py --symbol 600519 --json
```

### Project Structure

```
AShare_Quant_Server/
├── main.py                 # Unified CLI entry point
├── scanner.py              # Batch multi-factor screening
├── analyzer.py             # Single stock deep analysis
├── indicators.py           # Technical indicator calculations
├── news_analyzer.py        # News fetching & rule-based sentiment analysis
├── tech_stock_pool.json    # Default stock universe (~120 stocks)
├── requirements.txt        # Python dependencies
├── agent_tools_def.md      # OpenClaw tool definitions
└── output/                 # Generated reports
```

### JSON Output Schema

When using `--json`, each command returns a structured JSON object:

**scan**: `{ report_type, generated_at, market_regime, statistics, recommendations[], disclaimer }`

**analyze**: `{ report_type, generated_at, stock, market_regime, quote, fundamentals, technicals, news, sentiment, trade_advice, disclaimer }`

### Deployment on Tencent Cloud

```bash
# 1. SSH into your server
ssh user@your-server-ip

# 2. Clone and setup
git clone <repo-url> && cd AShare_Quant_Server
pip install -r requirements.txt

# 3. Test
python main.py scan --top 5 --json

# 4. (Optional) Setup as cron job
crontab -e
# 0 9 * * 1-5 cd /path/to/AShare_Quant_Server && python main.py scan --top 15 > output/daily_$(date +\%Y\%m\%d).txt 2>&1
```

### Disclaimer

This tool is for **research and educational purposes only**. It does not constitute investment advice. Investing involves risk; please exercise caution.

---

## 中文说明

### 概述

AShare Quant Server 是一个无头（Headless）、纯命令行的A股量化交易工具箱。专为云服务器（如腾讯云轻量应用服务器）设计，无需 GUI 界面，可被 LLM Agent（如 OpenClaw）以程序方式调用执行。

本服务器专注于**数据获取和量化评分**。LLM 相关的分析（如最终报告生成、深度语义情感分析）交由调用方 Agent（OpenClaw）处理，OpenClaw 可以利用结构化 JSON 输出作为自身 LLM 推理的上下文。

### 功能特性

- **批量选股扫描** - 多因子评分模型，覆盖基本面、技术面、消息面
- **单股深度分析** - 行情、技术指标、基本面、新闻公告、交易建议一站式输出
- **规则消息面分析** - 基于关键词匹配 + 机构评级的规则化情感分析（无需外部 LLM）
- **JSON 输出模式** - 使用 `--json` 标志输出结构化 JSON，便于 LLM Agent 解析
- **统一入口** - 单一 `main.py` 入口，子命令：`scan`（扫描）、`analyze`（分析）
- **无需 GUI** - 完全无头运行，不依赖 Streamlit、浏览器或桌面环境
- **多数据源** - 主数据源：东方财富(akshare)；备用：BaoStock、腾讯行情 API
- **风险管理** - ATR 止损、盈亏比、支撑/压力位计算
- **连续化评分** - 平滑连续评分 + 微调系数，最大程度减少同分现象

### 多因子评分模型

| 因子 | 满分 | 组成 |
|------|------|------|
| 趋势 | 30 | 均线排列、价格与MA20/MA60关系、MA斜率（连续化） |
| 动量 | 25 | MACD状态、RSI区间（连续化）、MACD加速度 |
| 量能 | 20 | 量比（连续化，峰值1.6x）、成交量趋势 |
| 估值 | 15 | PE甜蜜区间（连续化，峰值PE=12）、PB比率（连续化，峰值PB=1.5） |
| 消息面 | 10 | 规则化：新闻关键词分析 + 机构评级评分 |
| 胜率加减分 | +/-10 | MACD背离、回踩支撑、多信号共振、大盘环境 |

### 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 运行

```bash
# 批量选股（文本输出）
python main.py scan --top 10

# 批量选股（JSON 输出，供 LLM Agent 解析）
python main.py scan --top 10 --json

# 自定义权重选股
python main.py scan --top 10 --trend-weight 0.4 --momentum-weight 0.2 --volume-weight 0.2

# 使用不同股票池
python main.py scan --top 10 --pool blue_chip_pool.json

# 单股分析
python main.py analyze --symbol 600519
python main.py analyze --symbol 贵州茅台 --json
```

**自定义权重参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trend-weight` | 0.30 | 趋势因子权重 (30%) |
| `--momentum-weight` | 0.25 | 动量因子权重 (25%) |
| `--volume-weight` | 0.20 | 量能因子权重 (20%) |
| `--valuation-weight` | 0.15 | 估值因子权重 (15%) |
| `--sentiment-weight` | 0.10 | 消息面因子权重 (10%) |

**股票池：**
- `tech_stock_pool.json` - 默认股票池（约120只）
- `blue_chip_pool.json` - 蓝筹股池（50只大盘股）

也可以直接调用单独脚本：

```bash
python scanner.py --top 15 --pool tech_stock_pool.json
python analyzer.py --symbol 600519 --json
```

### 项目结构

```
AShare_Quant_Server/
├── main.py                 # 统一 CLI 入口
├── scanner.py              # 批量多因子选股
├── analyzer.py             # 单股深度分析
├── indicators.py           # 技术指标计算库
├── news_analyzer.py        # 新闻获取与规则化情感分析
├── tech_stock_pool.json    # 默认股票池（约120只）
├── requirements.txt        # Python 依赖
├── agent_tools_def.md      # OpenClaw 工具定义
└── output/                 # 生成的报告
```

### JSON 输出格式

使用 `--json` 时，各命令返回结构化 JSON：

**scan**: `{ report_type, generated_at, market_regime, statistics, recommendations[], disclaimer }`

**analyze**: `{ report_type, generated_at, stock, market_regime, quote, fundamentals, technicals, news, sentiment, trade_advice, disclaimer }`

### 腾讯云部署指南

```bash
# 1. SSH 登录服务器
ssh user@your-server-ip

# 2. 克隆并配置
git clone <repo-url> && cd AShare_Quant_Server
pip install -r requirements.txt

# 3. 测试运行
python main.py scan --top 5 --json

# 4. （可选）设置定时任务
crontab -e
# 每个交易日上午9点自动扫描
# 0 9 * * 1-5 cd /path/to/AShare_Quant_Server && python main.py scan --top 15 > output/daily_$(date +\%Y\%m\%d).txt 2>&1
```

### 免责声明

本工具**仅供研究学习使用**，不构成任何投资建议。投资有风险，入市需谨慎。

---

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
