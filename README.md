# TradingView Automated Ingestion Pipeline

An automated, distributed data pipeline for fetching OHLCV candlestick data and corporate actions (stock splits, dividends, earnings & revenues, and continuous futures rollovers) from TradingView directly into Hugging Face Parquet shards.

---

## Architecture Overview

- **Orchestration**: GitHub Actions ([`.github/workflows/weekly_pipeline.yml`](.github/workflows/weekly_pipeline.yml) and [`.github/workflows/node_runner.yml`](.github/workflows/node_runner.yml))
- **Compute Cluster**: 20 concurrent Kaggle notebooks distributed across 4 Kaggle accounts
- **Data Source**: TradingView WebSocket gateway (`wss://prodata.tradingview.com/socket.io/websocket`)
- **Storage Hub**: Hugging Face dataset repository (`thonguyen511/temp`)
- **Symbols Repository**: [tradingview_aio_symbols](https://github.com/thonguyen511/tradingview_aio_symbols.git)

---

## Directory Structure

```
├── .github/
│   └── workflows/
│       ├── weekly_pipeline.yml       # Master Sunday cron orchestrator
│       └── node_runner.yml           # Parameterized single-node runner
├── src/
│   └── src_2/                        # Distributed data ingestion engine
│       ├── config.py                 # Account credentials & global constants
│       ├── core/                     # TV WebSocket client, fetchers, storage engine
│       ├── initial_run/              # Deep historical back-paging loop to Day 1
│       ├── weekly_run/               # Weekly maintenance pipeline (last 300 bars)
│       └── orchestration/            # Kaggle kernel preparation and dispatcher
├── requirements.txt
└── README.md
```

---

## GitHub Actions Secrets Required

Ensure the following secrets are configured in `Settings > Secrets and variables > Actions`:

| Secret Name | Description |
| :--- | :--- |
| `HF_TOKEN` | Hugging Face write token for dataset storage |
| `GH_PAT` | GitHub Personal Access Token (with `repo` & `workflow` scopes) |
| `DATA_PASSWORD` | Optional encryption password for dataset at rest |
| `TV_ULTIMATE_SESSIONID` | Optional TradingView premium session ID for intraday |
| `TV_ULTIMATE_SIGN` | Optional TradingView premium signature |

---

## Manual Execution Commands

### Run Weekly Pipeline on a specific node:
```bash
python -m src.src_2.main --mode weekly --node 0
```

### Dispatch all 20 Kaggle Workers:
```bash
python -m src.src_2.orchestration.kaggle_dispatcher dispatch weekly
```

### Check Status of all Kaggle Nodes:
```bash
python -m src.src_2.orchestration.kaggle_dispatcher status
```
