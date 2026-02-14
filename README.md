# AShare Quant Server

**A-Share Quantitative Trading Toolkit - Headless Server Edition**

**A股量化交易工具箱 - 无头服务器版**

---

[中文](#中文说明) | [English](#english)

---

## English

### Overview

AShare Quant Server is a headless, CLI-based quantitative trading toolkit for Chinese A-share stocks. It is designed to run on cloud servers (e.g., Tencent Cloud Lighthouse) without a GUI and to be invoked programmatically by LLM agents such as OpenClaw.

### Features

- **Batch Stock Screening** - Multi-factor scoring across a customizable stock pool (fundamentals + technicals + sentiment)
- **Single Stock Deep Analysis** - Comprehensive analysis with quote, technicals, fundamentals, news, and trade advice
- **LLM-Powered Sentiment** - Optional sentiment analysis via any OpenAI-compatible LLM API
- **JSON Output Mode** - Structured JSON output (`--json`) for programmatic consumption by LLM agents
- **Unified CLI Entry Point** - Single `main.py` with subcommands: `scan`, `analyze`, `report`
- **No GUI Required** - Fully headless; no Streamlit, no browser, no desktop needed
- **Multi-Source Data** - Primary: East Money (akshare); Backup: BaoStock, Tencent Quotes API
- **Risk Management** - ATR-based stop-loss, risk-reward ratio, support/resistance levels

### Multi-Factor Scoring Model

| Factor | Max Points | Components |
|--------|-----------|------------|
| Trend | 30 | MA arrangement, price vs MA20/MA60, MA slope |
| Momentum | 25 | MACD status, RSI zone, KDJ positions |
| Volume | 20 | Volume ratio, 5d vs 20d volume trend |
| Valuation | 15 | PE sweet spot, PB ratio |
| Sentiment | 10 | LLM-based news/announcement analysis |
| Win-Rate Bonus | +/-10 | MACD divergence, pullback support, confluence |

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure LLM (Optional)

```bash
cp llm_config.example.json llm_config.json
# Edit llm_config.json with your API credentials
```

Or use environment variables:

```bash
export LLM_URL="https://api.example.com/v1"
export LLM_API_KEY="sk-your-key"
export LLM_MODEL="your-model"
```

#### 3. Run

```bash
# Batch screening (text output)
python main.py scan --top 10

# Batch screening (JSON output for LLM agents)
python main.py scan --top 10 --json

# Single stock analysis
python main.py analyze --symbol 600519
python main.py analyze --symbol 贵州茅台 --json

# Full pipeline: scan + analyze + LLM summary
python main.py report --symbol 600519 --top 10
```

You can also call individual scripts directly:

```bash
python scanner.py --top 15 --pool tech_stock_pool.json
python analyzer.py --symbol 600519 --json
python llm_final_report.py --top 10 --symbol 600519
```

### Project Structure

```
AShare_Quant_Server/
├── main.py                 # Unified CLI entry point
├── scanner.py              # Batch multi-factor screening
├── analyzer.py             # Single stock deep analysis
├── llm_final_report.py     # Scanner + Analyzer + LLM orchestrator
├── indicators.py           # Technical indicator calculations
├── llm_client.py           # OpenAI-compatible LLM client
├── news_analyzer.py        # News fetching & sentiment analysis
├── tech_stock_pool.json    # Default stock universe (~120 stocks)
├── llm_config.example.json # LLM config template
├── requirements.txt        # Python dependencies
├── agent_tools_def.md      # OpenClaw tool definitions
└── output/                 # Generated reports
```

### Configuration Priority

1. CLI arguments (`--llm-url`, `--llm-key`, `--llm-model`)
2. Environment variables (`LLM_URL` / `OPENAI_BASE_URL`, etc.)
3. Config file (`llm_config.json`)

### JSON Output Schema

When using `--json`, each command returns a structured JSON object:

**scan**: `{ report_type, generated_at, market_regime, statistics, recommendations[], disclaimer }`

**analyze**: `{ report_type, generated_at, stock, market_regime, quote, fundamentals, technicals, news, sentiment, trade_advice, disclaimer }`

**report**: `{ report_type, generated_at, symbol, content, output_path, disclaimer }`

### Deployment on Tencent Cloud

```bash
# 1. SSH into your server
ssh user@your-server-ip

# 2. Clone and setup
git clone <repo-url> && cd AShare_Quant_Server
pip install -r requirements.txt

# 3. Configure LLM
cp llm_config.example.json llm_config.json
vim llm_config.json

# 4. Test
python main.py scan --top 5 --json

# 5. (Optional) Setup as cron job
crontab -e
# 0 9 * * 1-5 cd /path/to/AShare_Quant_Server && python main.py scan --top 15 > output/daily_$(date +\%Y\%m\%d).txt 2>&1
```

### Disclaimer

This tool is for **research and educational purposes only**. It does not constitute investment advice. Investing involves risk; please exercise caution.

---

## 中文说明

### 概述

AShare Quant Server 是一个无头（Headless）、纯命令行的A股量化交易工具箱。专为云服务器（如腾讯云轻量应用服务器）设计，无需 GUI 界面，可被 LLM Agent（如 OpenClaw）以程序方式调用执行。

### 功能特性

- **批量选股扫描** - 多因子评分模型，覆盖基本面、技术面、消息面
- **单股深度分析** - 行情、技术指标、基本面、新闻公告、交易建议一站式输出
- **LLM 消息面分析** - 可选接入任何 OpenAI 兼容 API 进行新闻情感分析
- **JSON 输出模式** - 使用 `--json` 标志输出结构化 JSON，便于 LLM Agent 解析
- **统一入口** - 单一 `main.py` 入口，子命令：`scan`（扫描）、`analyze`（分析）、`report`（报告）
- **无需 GUI** - 完全无头运行，不依赖 Streamlit、浏览器或桌面环境
- **多数据源** - 主数据源：东方财富(akshare)；备用：BaoStock、腾讯行情 API
- **风险管理** - ATR 止损、盈亏比、支撑/压力位计算

### 多因子评分模型

| 因子 | 满分 | 组成 |
|------|------|------|
| 趋势 | 30 | 均线排列、价格与MA20/MA60关系、MA斜率 |
| 动量 | 25 | MACD状态、RSI区间、KDJ位置 |
| 量能 | 20 | 量比、5日/20日成交量趋势 |
| 估值 | 15 | PE甜蜜区间、PB比率 |
| 消息面 | 10 | LLM新闻/公告情感分析 |
| 胜率加减分 | +/-10 | MACD背离、回踩支撑、多信号共振 |

### 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置 LLM（可选）

```bash
cp llm_config.example.json llm_config.json
# 编辑 llm_config.json 填入你的 API 信息
```

或使用环境变量：

```bash
export LLM_URL="https://api.example.com/v1"
export LLM_API_KEY="sk-your-key"
export LLM_MODEL="your-model"
```

#### 3. 运行

```bash
# 批量选股（文本输出）
python main.py scan --top 10

# 批量选股（JSON 输出，供 LLM Agent 解析）
python main.py scan --top 10 --json

# 单股分析
python main.py analyze --symbol 600519
python main.py analyze --symbol 贵州茅台 --json

# 完整流程：扫描 + 分析 + LLM 总结
python main.py report --symbol 600519 --top 10
```

也可以直接调用单独脚本：

```bash
python scanner.py --top 15 --pool tech_stock_pool.json
python analyzer.py --symbol 600519 --json
python llm_final_report.py --top 10 --symbol 600519
```

### 项目结构

```
AShare_Quant_Server/
├── main.py                 # 统一 CLI 入口
├── scanner.py              # 批量多因子选股
├── analyzer.py             # 单股深度分析
├── llm_final_report.py     # 扫描+分析+LLM 编排器
├── indicators.py           # 技术指标计算库
├── llm_client.py           # OpenAI 兼容 LLM 客户端
├── news_analyzer.py        # 新闻获取与情感分析
├── tech_stock_pool.json    # 默认股票池（约120只）
├── llm_config.example.json # LLM 配置模板
├── requirements.txt        # Python 依赖
├── agent_tools_def.md      # OpenClaw 工具定义
└── output/                 # 生成的报告
```

### 配置优先级

1. 命令行参数（`--llm-url`、`--llm-key`、`--llm-model`）
2. 环境变量（`LLM_URL` / `OPENAI_BASE_URL` 等）
3. 配置文件（`llm_config.json`）

### JSON 输出格式

使用 `--json` 时，各命令返回结构化 JSON：

**scan**: `{ report_type, generated_at, market_regime, statistics, recommendations[], disclaimer }`

**analyze**: `{ report_type, generated_at, stock, market_regime, quote, fundamentals, technicals, news, sentiment, trade_advice, disclaimer }`

**report**: `{ report_type, generated_at, symbol, content, output_path, disclaimer }`

### 腾讯云部署指南

```bash
# 1. SSH 登录服务器
ssh user@your-server-ip

# 2. 克隆并配置
git clone <repo-url> && cd AShare_Quant_Server
pip install -r requirements.txt

# 3. 配置 LLM
cp llm_config.example.json llm_config.json
vim llm_config.json

# 4. 测试运行
python main.py scan --top 5 --json

# 5. （可选）设置定时任务
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
