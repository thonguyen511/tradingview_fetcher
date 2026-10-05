"""CLI entry point for initial deep historical ingestion."""
import sys
from .pipeline import run_initial_pipeline
from ..config import NOTEBOOK_INDEX

def main():
    run_initial_pipeline(node_index=NOTEBOOK_INDEX)

if __name__ == "__main__":
    main()
