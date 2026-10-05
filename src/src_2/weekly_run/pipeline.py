"""Weekly maintenance pipeline: 100% Free Accounts for candle finalization and new bar updates."""
import os
import gc
import time
import queue
import shutil
import random
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional

from ..config import (
    TOTAL_NOTEBOOKS,
    NOTEBOOK_INDEX,
    NOTEBOOK_ID,
    FREE_WORKERS,
    INTRADAY_INTERVALS,
    DAILY_INTERVALS,
    ALL_INTERVALS,
    ADJUSTMENT_MODE,
    CHUNK_SIZE,
    WEEKLY_N_BARS,
    MAX_NETWORK_RETRIES,
    LOG_DIR,
    OUTPUT_DIR,
    HF_REPO_ID,
    get_node_free_cookie,
    fetch_jwt_token
)
from ..core.symbol_reader import get_node_workload
from ..core.tv_fetcher import TradingView1DProbeFetcher, TradingViewWeeklyFetcher
from ..core.events_engine import attach_events_to_df
from ..core.hf_storage import HuggingFaceStorageEngine
from ..utils.logger import log_failure, record_checkpoint, wait_queues_periodic_log

# Thread-safe global containers per chunk
results_dict: Dict[str, Any] = {}
results_lock = threading.Lock()

symbol_events: Dict[str, Any] = {}
events_lock = threading.Lock()

COMPLETED_TASKS: Set[str] = set()
completed_lock = threading.Lock()

IS_STOPPED = False

def weekly_probe_worker_loop(
    worker_id: int,
    sessionid: str,
    sign: str,
    jwt: str,
    task_queue: queue.Queue,
    retry_queue: queue.Queue,
    is_done: threading.Event
):
    """Weekly Step A Worker: Concurrent 1D probe, corporate actions, and metadata extraction."""
    fetcher = TradingView1DProbeFetcher(sessionid, sign, jwt, server="data")
    global IS_STOPPED

    while not IS_STOPPED and not is_done.is_set():
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            fetcher.client.keep_alive()
            continue

        symbol, is_futures, is_retry = task
        task_key = f"{symbol}_1D"
        time.sleep(random.uniform(0.04, 0.08))

        try:
            bars, events, sym_info, err_cat, err_det = fetcher.fetch_1d_and_events(
                symbol=symbol,
                is_futures=is_futures
            )
            if bars:
                with results_lock:
                    results_dict[task_key] = {
                        "symbol": symbol,
                        "interval": "1D",
                        "bars": bars,
                        "sym_info": sym_info
                    }
                with events_lock:
                    symbol_events[symbol] = events
            else:
                if err_cat in ["tradingview_message", "server_network_error"]:
                    log_failure(symbol, "1D", err_cat, err_det, log_dir=LOG_DIR)
                    with completed_lock:
                        for any_int in ALL_INTERVALS:
                            COMPLETED_TASKS.add(f"{symbol} ({any_int})")
                else:
                    if not is_retry:
                        retry_queue.put((symbol, is_futures, True))
        except Exception:
            if not is_retry:
                retry_queue.put((symbol, is_futures, True))
        finally:
            task_queue.task_done()

    fetcher.close()


def weekly_worker_loop(
    worker_id: int,
    sessionid: str,
    sign: str,
    jwt: str,
    task_queue: queue.Queue,
    retry_queue: queue.Queue,
    is_done: threading.Event,
    n_bars: int = WEEKLY_N_BARS
):
    """Weekly Worker: Fetches recent candles for a symbol on any interval."""
    fetcher = TradingViewWeeklyFetcher(sessionid, sign, jwt)
    global IS_STOPPED

    while not IS_STOPPED and not is_done.is_set():
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            fetcher.keep_alive()
            continue

        symbol, interval, is_retry = task
        task_key = f"{symbol}_{interval}"
        time.sleep(random.uniform(0.04, 0.08))

        try:
            bars, sym_info, err_cat, err_det = fetcher.fetch_recent_bars(
                symbol=symbol,
                interval=interval,
                n_bars=n_bars
            )
            if bars:
                with results_lock:
                    results_dict[task_key] = {
                        "symbol": symbol,
                        "interval": interval,
                        "bars": bars,
                        "sym_info": sym_info
                    }
            else:
                if err_cat in ["tradingview_message", "server_network_error"]:
                    log_failure(symbol, interval, err_cat, err_det, log_dir=LOG_DIR)
                    with completed_lock:
                        COMPLETED_TASKS.add(f"{symbol} ({interval})")
                else:
                    if not is_retry:
                        retry_queue.put((symbol, interval, True))
        except Exception:
            if not is_retry:
                retry_queue.put((symbol, interval, True))
        finally:
            task_queue.task_done()

    fetcher.close()


def run_weekly_pipeline(
    node_index: int = NOTEBOOK_INDEX,
    restart_callback: Optional[Any] = None
):
    """Executes the fast weekly update pipeline for the given node using 100% Free accounts."""
    session_start_time = time.time()
    MAX_SESSION_SECONDS = float(os.environ.get("MAX_SESSION_SECONDS", 11.5 * 3600))  # 11 hours 30 mins

    print("=" * 80)
    print(f"🔄 TRADINGVIEW WEEKLY MAINTENANCE PIPELINE (NODE {node_index}/{TOTAL_NOTEBOOKS})")
    print("=" * 80)
    print(f"• Mode: 100% Free Accounts (Updating {WEEKLY_N_BARS} Recent Candlesticks & Actions)")
    print(f"• Workers: {FREE_WORKERS} Concurrent Threads")
    print(f"• Adjustment: '{ADJUSTMENT_MODE}' (Raw Unadjusted Prices)")
    print(f"• Destination HF Repo: {HF_REPO_ID}")
    print("=" * 80)

    # 1. Authenticate Free Cookie
    free_cookie = get_node_free_cookie(node_index)
    sessionid = free_cookie["sessionid"]
    sign = free_cookie.get("sessionid_sign", "")
    jwt_token = fetch_jwt_token(sessionid, sign)
    print(f"🔑 Node Free Account: sessionid={sessionid[:8]}... (sign: {bool(sign)})")
    print(f"✅ Auth Token Acquired.\n")

    # 2. Storage Setup & Pull Checkpoints
    storage = HuggingFaceStorageEngine()
    print("⏳ Synchronizing checkpoint logs from Hugging Face...")
    storage.download_checkpoint_logs(local_log_dir=LOG_DIR, notebook_id=NOTEBOOK_ID)

    if os.path.exists(LOG_DIR):
        for log_f in ["checkpoint.log", "failures.log"]:
            lp = os.path.join(LOG_DIR, log_f)
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = [p.strip() for p in line.split(", ")]
                        if len(parts) >= 3:
                            with completed_lock:
                                COMPLETED_TASKS.add(f"{parts[1]} ({parts[2]})")
    print(f"✅ Loaded {len(COMPLETED_TASKS)} completed tasks from checkpoint logs.\n")

    # 3. Workload Discovery
    chunks = get_node_workload(node_index=node_index, total_nodes=TOTAL_NOTEBOOKS, chunk_size=CHUNK_SIZE)
    if not chunks:
        print(f"❌ No workload chunks assigned to Node {node_index}. Exiting.")
        return

    # 4. Process Each Workload Chunk
    for chunk_idx, chunk in enumerate(chunks, 1):
        chunk_tag = chunk["chunk_tag"]
        csv_name = chunk["csv_name"]
        symbols = chunk["symbols"]
        is_futures = ("FUTURES" in csv_name.upper()) or any("1!" in s for s in symbols[:5])

        print("\n" + "-" * 75)
        print(f"📂 Chunk {chunk_idx}/{len(chunks)}: {chunk_tag} ({len(symbols)} symbols)")
        print(f"   CSV: {csv_name} | Futures: {is_futures}")
        print("-" * 75)

        with results_lock:
            results_dict.clear()
        with events_lock:
            symbol_events.clear()

        # Step A: 1D Resolution & Corporate Actions Probe (Concurrent across FREE_WORKERS)
        print(f"🔍 Probing 1D Timeframe & Corporate Actions across {FREE_WORKERS} workers...")
        p1_queue = queue.Queue()
        p1_retry_queue = queue.Queue()
        is_p1_done = threading.Event()

        queued_1d = 0
        for sym in symbols:
            if f"{sym} (1D)" not in COMPLETED_TASKS:
                p1_queue.put((sym, is_futures, False))
                queued_1d += 1

        if queued_1d > 0:
            p1_threads = []
            for w_id in range(FREE_WORKERS):
                t = threading.Thread(
                    target=weekly_probe_worker_loop,
                    args=(w_id, sessionid, sign, jwt_token, p1_queue, p1_retry_queue, is_p1_done),
                    daemon=True
                )
                t.start()
                p1_threads.append(t)
                time.sleep(0.02)

            wait_queues_periodic_log([(p1_queue, queued_1d)], desc="Weekly 1D & Corporate Actions Probe")

            # Retry transient drops
            retry_round = 1
            while not p1_retry_queue.empty() and retry_round <= MAX_NETWORK_RETRIES:
                retries = []
                while not p1_retry_queue.empty():
                    retries.append(p1_retry_queue.get())
                if not retries: break
                time.sleep(2.0 * retry_round)
                for r in retries:
                    p1_queue.put((r[0], r[1], True))
                wait_queues_periodic_log([(p1_queue, len(retries))], desc=f"Probe Retry Round {retry_round}")
                retry_round += 1

            # Log permanent failures
            while not p1_retry_queue.empty():
                f_sym, _, _ = p1_retry_queue.get()
                log_failure(f_sym, "1D", "server_network_error", f"Exceeded {MAX_NETWORK_RETRIES} retries", log_dir=LOG_DIR)
                with completed_lock:
                    for any_int in ALL_INTERVALS:
                        COMPLETED_TASKS.add(f"{f_sym} ({any_int})")

            is_p1_done.set()
            for t in p1_threads:
                t.join(timeout=3.0)

        # Identify valid symbols that actually have 1D candlestick data
        valid_symbols = []
        intraday_eligible = []
        for sym in symbols:
            res = results_dict.get(f"{sym}_1D")
            if res and res.get("bars"):
                valid_symbols.append(sym)
                sym_info = res.get("sym_info")
                has_intra = sym_info.get("has_intraday", True) if sym_info else True
                if has_intra:
                    intraday_eligible.append(sym)

        print(f"   -> {len(valid_symbols)}/{len(symbols)} symbols valid with 1D candlestick data.")
        print(f"   -> {len(intraday_eligible)}/{len(symbols)} symbols eligible for intraday updates.")

        # Step B: Queue Remaining Intervals (1W, 1M, and Intraday)
        weekly_queue = queue.Queue()
        weekly_retry_queue = queue.Queue()
        is_weekly_done = threading.Event()

        queued_tasks = 0

        # Queue Macro (1W, 1M) for valid symbols ONLY
        for m_int in ["1W", "1M"]:
            for sym in valid_symbols:
                if f"{sym} ({m_int})" not in COMPLETED_TASKS:
                    weekly_queue.put((sym, m_int, False))
                    queued_tasks += 1

        # Queue Intraday Intervals for eligible symbols
        for i_int in INTRADAY_INTERVALS:
            for sym in intraday_eligible:
                if f"{sym} ({i_int})" not in COMPLETED_TASKS:
                    weekly_queue.put((sym, i_int, False))
                    queued_tasks += 1

        if queued_tasks > 0:
            threads = []
            for w_id in range(FREE_WORKERS):
                t = threading.Thread(
                    target=weekly_worker_loop,
                    args=(w_id, sessionid, sign, jwt_token, weekly_queue, weekly_retry_queue, is_weekly_done, WEEKLY_N_BARS),
                    daemon=True
                )
                t.start()
                threads.append(t)
                time.sleep(0.02)

            wait_queues_periodic_log([(weekly_queue, queued_tasks)], desc="Weekly Recent Candles Fetch")

            # Retry dropped tasks
            retry_round = 1
            while not weekly_retry_queue.empty() and retry_round <= MAX_NETWORK_RETRIES:
                retries = []
                while not weekly_retry_queue.empty():
                    retries.append(weekly_retry_queue.get())
                if not retries: break
                print(f"🔄 Retrying {len(retries)} dropped tasks (Round {retry_round}/{MAX_NETWORK_RETRIES})...")
                time.sleep(2.0 * retry_round)
                for r in retries:
                    weekly_queue.put((r[0], r[1], True))
                wait_queues_periodic_log([(weekly_queue, len(retries))], desc=f"Weekly Retry Round {retry_round}")
                retry_round += 1

            # Log permanent failures
            while not weekly_retry_queue.empty():
                f_sym, f_int, _ = weekly_retry_queue.get()
                log_failure(f_sym, f_int, "server_network_error", f"Exceeded {MAX_NETWORK_RETRIES} retries", log_dir=LOG_DIR)
                with completed_lock:
                    COMPLETED_TASKS.add(f"{f_sym} ({f_int})")

            is_weekly_done.set()
            for t in threads:
                t.join(timeout=3.0)

        # Step C: Single-Shard Overwrite & Merge per interval
        # Downloads target chunk from HF, merges with keep='last', flattens, and overwrites
        with events_lock:
            local_events = dict(symbol_events)

        with results_lock:
            all_collected_keys = list(results_dict.keys())
            intervals_in_chunk = set(results_dict[k]["interval"] for k in all_collected_keys)

        for interval in intervals_in_chunk:
            with results_lock:
                keys_for_int = [k for k in all_collected_keys if results_dict.get(k, {}).get("interval") == interval]
                items = [results_dict[k] for k in keys_for_int]

            if not items:
                continue

            df_list = []
            for data in items:
                sym = data["symbol"]
                bars = data.get("bars", [])
                events = local_events.get(sym, {})
                if not bars:
                    continue

                arr = np.array(bars, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] < 5:
                    continue

                n_rows = arr.shape[0]
                vols = arr[:, 5] if arr.shape[1] >= 6 else np.zeros(n_rows, dtype=np.float64)

                df_single = pd.DataFrame({
                    "time": arr[:, 0].astype(np.int64),
                    "open": arr[:, 1],
                    "high": arr[:, 2],
                    "low": arr[:, 3],
                    "close": arr[:, 4],
                    "volume": vols,
                    "symbol": sym
                })

                df_single = attach_events_to_df(df=df_single, raw_events=events, interval=interval)
                df_list.append(df_single)

                chunk_rel_path = f"data/{csv_name}/{interval}/{chunk_tag}.parquet"
                record_checkpoint(sym, interval, len(df_single), chunk_rel_path, log_dir=LOG_DIR)
                with completed_lock:
                    COMPLETED_TASKS.add(f"{sym} ({interval})")

            if df_list:
                new_interval_df = pd.concat(df_list, ignore_index=True)
                del df_list
                chunk_rel_path = f"data/{csv_name}/{interval}/{chunk_tag}.parquet"

                # 1. Pull target existing chunk from Hugging Face
                existing_df = storage.pull_chunk(chunk_rel_path)

                # 2. Merge & Overwrite: keep='last' finalizes previous unclosed candles!
                merged_df = storage.merge_datasets(existing_df, new_interval_df)
                del existing_df
                del new_interval_df

                # 3. Save merged clean Parquet to local output folder
                storage.save_local_chunk(
                    df=merged_df,
                    chunk_rel_path=chunk_rel_path,
                    local_dir=OUTPUT_DIR
                )
                del merged_df
                gc.collect()

        # Step D: Single Atomic Commit to Hugging Face
        local_data_dir = os.path.join(OUTPUT_DIR, "data")
        if os.path.exists(local_data_dir):
            storage.push_data_folder(
                local_data_dir=local_data_dir,
                commit_message=f"Weekly update: {csv_name} {chunk_tag}"
            )

        storage.push_checkpoint_logs(
            local_log_dir=LOG_DIR,
            notebook_id=NOTEBOOK_ID,
            commit_message=f"Sync weekly logs {csv_name} {chunk_tag}"
        )

        # Cleanup local disk
        shutil.rmtree(os.path.join(OUTPUT_DIR, "data"), ignore_errors=True)
        print(f"✅ Chunk {chunk_tag} merged and pushed to Hugging Face!")

        # Check watchdog session elapsed time before starting next chunk
        session_elapsed = time.time() - session_start_time
        if session_elapsed > MAX_SESSION_SECONDS and chunk_idx < len(chunks):
            print(f"\n⏰ Session elapsed: {session_elapsed/3600:.2f}h exceeds limit ({MAX_SESSION_SECONDS/3600:.2f}h).")
            print(f"📦 Completed {chunk_idx}/{len(chunks)} chunks. {len(chunks) - chunk_idx} chunks remaining.")
            storage.squash_history()
            if restart_callback:
                print("🔁 Triggering next Kaggle session before timeout...")
                restart_callback()
            print("👋 Clean session exit.")
            return

    # Step E: Squash Git history to prune old overwritten blobs and enforce < 8TB limit
    storage.squash_history()

    print("=" * 80)
    print(f"🏁 WEEKLY MAINTENANCE COMPLETE FOR NODE {node_index}")
    print("=" * 80)
