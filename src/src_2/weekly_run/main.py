"""CLI entry point for weekly maintenance ingestion."""
import sys
from .pipeline import run_weekly_pipeline
from ..config import NOTEBOOK_INDEX

def main():
    run_weekly_pipeline(node_index=NOTEBOOK_INDEX)

if __name__ == "__main__":
    main()
