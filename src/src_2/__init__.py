"""src_2: TradingView Automated OHLCV & Corporate Actions Ingestion Pipeline.

Supports:
- Initial Deep Ingestion (Inception Day 1 to present)
- Weekly Maintenance Ingestion (Candle finalization & new updates)
- Dual-tier authentication (Free pool + Premium/Ultimate pool)
- Hugging Face storage with git history squashing (< 8TB limit)
- 20-node concurrent execution across 4 Kaggle accounts
"""

__version__ = "2.0.0"
