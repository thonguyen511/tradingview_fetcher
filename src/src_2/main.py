"""Interactive CLI and entrypoint for src_2 pipeline."""
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import argparse

def interactive_menu():
    try:
        import questionary
    except ImportError:
        questionary = None

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("  TRADINGVIEW DATA INGESTION & PIPELINE (SRC_2)")
        print("=" * 80)
        print("  1. Run Initial Deep Ingestion (Single Node)")
        print("  2. Run Weekly Maintenance Ingestion (Single Node)")
        print("  3. Build / Incrementally Update map.json (Greedy Allocation)")
        print("  4. Dispatch 20 Concurrent Kaggle Notebooks")
        print("  5. Check Kaggle Notebooks Status")
        print("  6. Decrypt Encrypted Parquet Files (.parquet.enc)")
        print("  7. Exit")
        print("=" * 80)

        if questionary:
            choice = questionary.select(
                "Select operation:",
                choices=[
                    "1. Run Initial Deep Ingestion (Single Node)",
                    "2. Run Weekly Maintenance Ingestion (Single Node)",
                    "3. Build / Incrementally Update map.json",
                    "4. Dispatch 20 Concurrent Kaggle Notebooks",
                    "5. Check Kaggle Notebooks Status",
                    "6. Decrypt Encrypted Parquet Files",
                    "7. Exit"
                ]
            ).ask()
        else:
            choice = input("\nEnter choice (1-7): ").strip()

        if not choice or "7" in choice or choice.lower() in ["exit", "q"]:
            print("Exiting.")
            break

        if "1" in choice:
            node_str = input("Enter Node Index (0-19) [default 0]: ").strip() or "0"
            from .initial_run.pipeline import run_initial_pipeline
            run_initial_pipeline(node_index=int(node_str))
            input("\nPress Enter to return to menu...")

        elif "2" in choice:
            node_str = input("Enter Node Index (0-19) [default 0]: ").strip() or "0"
            from .weekly_run.pipeline import run_weekly_pipeline
            run_weekly_pipeline(node_index=int(node_str))
            input("\nPress Enter to return to menu...")

        elif "3" in choice:
            from .core.greedy_allocator import update_map_incremental
            update_map_incremental()
            input("\nPress Enter to return to menu...")

        elif "4" in choice:
            mode = input("Enter mode (weekly or initial) [default weekly]: ").strip() or "weekly"
            from .orchestration.kaggle_dispatcher import dispatch_all_nodes
            dispatch_all_nodes(mode=mode)
            input("\nPress Enter to return to menu...")

        elif "5" in choice:
            from .orchestration.kaggle_dispatcher import check_all_status
            check_all_status()
            input("\nPress Enter to return to menu...")

        elif "6" in choice:
            folder = input("Enter path to dataset directory (e.g. 'output/'): ").strip() or "output"
            pwd = input("Enter DATA_PASSWORD: ").strip()
            from .utils.security import decrypt_dataset_directory
            decrypt_dataset_directory(folder, pwd)
            input("\nPress Enter to return to menu...")

def main():
    parser = argparse.ArgumentParser(description="TradingView Ingestion Pipeline (src_2)")
    parser.add_argument("--mode", choices=["initial", "weekly", "map", "dispatch", "status", "interactive"], default="interactive")
    parser.add_argument("--node", type=int, default=0, help="Node index (0 to 19)")
    args = parser.parse_args()

    if args.mode == "interactive":
        interactive_menu()
    elif args.mode == "initial":
        from .initial_run.pipeline import run_initial_pipeline
        run_initial_pipeline(node_index=args.node)
    elif args.mode == "weekly":
        from .weekly_run.pipeline import run_weekly_pipeline
        run_weekly_pipeline(node_index=args.node)
    elif args.mode == "map":
        from .core.greedy_allocator import update_map_incremental
        update_map_incremental()
    elif args.mode == "dispatch":
        from .orchestration.kaggle_dispatcher import dispatch_all_nodes
        dispatch_all_nodes(mode="weekly")
    elif args.mode == "status":
        from .orchestration.kaggle_dispatcher import check_all_status
        check_all_status()

if __name__ == "__main__":
    main()
