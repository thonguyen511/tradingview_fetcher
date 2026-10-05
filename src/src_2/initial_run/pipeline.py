"""Initial run pipeline: Deep historical back-paging to Day 1 with dual-tier worker pools."""
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
    ULTIMATE_WORKERS,
    FREE_WORKERS,
    TOTAL_WORKERS,
    INTRADAY_INTERVALS,
    DAILY_INTERVALS,
    ALL_INTERVALS,
    ADJUSTMENT_MODE,
    CHUNK_SIZE,
    MAX_NETWORK_RETRIES,
    MAX_HISTORY_BATCHES,
    LOG_DIR,
    OUTPUT_DIR,
    HF_REPO_ID,
    ULTIMATE_SESSIONID,
    ULTIMATE_SIGN,
    get_node_free_cookie,
    fetch_jwt_token
)
from ..core.symbol_reader import get_node_workload
from ..core.tv_fetcher import TradingView1DProbeFetcher, TradingViewDeepFetcher
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

def flush_and_save_interval(
    interval: str,
    csv_name: str,
    chunk_tag: str,
    storage: HuggingFaceStorageEngine,
    output_dir: str = OUTPUT_DIR,
    log_dir: str = LOG_DIR
) -> int:
    """
    Serializes collected bars for an interval to Parquet on disk and immediately frees RAM.
    """
    with results_lock:
        keys_for_int = [k for k, v in results_dict.items() if v.get("interval") == interval]
        items_to_save = [results_dict.pop(k) for k in keys_for_int]

    if not items_to_save:
        return 0

    with events_lock:
        local_events = dict(symbol_events)

    df_list = []
    chunk_rel_path = f"data/{csv_name}/{interval}/{chunk_tag}.parquet"

    for data in items_to_save:
        sym = data["symbol"]
        bars = data.get("bars")
        events = local_events.get(sym, {})
        if not bars:
            continue

        arr = bars if isinstance(bars, np.ndarray) else np.array(bars, dtype=np.float64)
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
        del arr

        # Attach corporate actions (splits, dividends, earnings, continuous rolls)
        df_single = attach_events_to_df(df=df_single, raw_events=events, interval=interval)
        df_list.append(df_single)

        record_checkpoint(sym, interval, len(df_single), chunk_rel_path, log_dir=log_dir)
        with completed_lock:
            COMPLETED_TASKS.add(f"{sym} ({interval})")

    del items_to_save

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        del df_list
        combined_df.drop_duplicates(subset=["symbol", "time"], keep="last", inplace=True)
        combined_df.sort_values(by=["symbol", "time"], ascending=[True, True], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        storage.save_local_chunk(
            df=combined_df,
            chunk_rel_path=chunk_rel_path,
            local_dir=output_dir
        )
        del combined_df

    gc.collect()
    return len(keys_for_int)


# ------------------------------------------------------------------------------
# WORKER THREAD LOOPS
# ------------------------------------------------------------------------------
def phase1_probe_worker_loop(
    worker_id: int,
    sessionid: str,
    sign: str,
    jwt: str,
    server: str,
    task_queue: queue.Queue,
    retry_queue: queue.Queue,
    is_done: threading.Event
):
    """Phase 1 Worker: 100% Free cookie. Probes 1D bars, corporate actions, and has_intraday."""
    fetcher = TradingView1DProbeFetcher(sessionid, sign, jwt, server=server)
    global IS_STOPPED

    while not IS_STOPPED and not is_done.is_set():
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            fetcher.client.keep_alive()
            continue

        symbol, is_futures, is_retry = task
        task_key = f"{symbol}_1D"
        time.sleep(random.uniform(0.02, 0.05))

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
                if err_cat == "tradingview_message":
                    log_failure(symbol, "1D", err_cat, err_det, log_dir=LOG_DIR)
                    with completed_lock:
                        COMPLETED_TASKS.add(f"{symbol} (1D)")
                else:
                    if not is_retry:
                        retry_queue.put((symbol, is_futures, True))
        except Exception:
            if not is_retry:
                retry_queue.put((symbol, is_futures, True))
        finally:
            task_queue.task_done()

    fetcher.close()


def phase2_deep_worker_loop(
    worker_id: int,
    sessionid: str,
    sign: str,
    jwt: str,
    server: str,
    task_queue: queue.Queue,
    retry_queue: queue.Queue,
    is_done: threading.Event,
    max_batches: int = MAX_HISTORY_BATCHES
):
    """Phase 2 Deep Ingestion Worker: Loops backward using request_more_data until Day 1."""
    fetcher = TradingViewDeepFetcher(sessionid, sign, jwt, server=server)
    global IS_STOPPED

    while not IS_STOPPED and not is_done.is_set():
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            fetcher.keep_alive()
            continue

        symbol, interval, is_retry = task
        task_key = f"{symbol}_{interval}"
        time.sleep(random.uniform(0.02, 0.05))

        try:
            bars, sym_info, status, err_cat, err_det = fetcher.fetch_deep_history(
                symbol=symbol,
                interval=interval,
                max_batches=max_batches
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
                if err_cat == "tradingview_message":
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


def run_initial_pipeline(
    node_index: int = NOTEBOOK_INDEX,
    restart_callback: Optional[Any] = None
):
    """Executes the full initial deep historical ingestion for the given node."""
    session_start_time = time.time()
    MAX_SESSION_SECONDS = float(os.environ.get("MAX_SESSION_SECONDS", 11.5 * 3600))  # 11 hours 30 mins

    print("=" * 80)
    print(f"🚀 TRADINGVIEW INITIAL DEEP INGESTION PIPELINE (NODE {node_index}/{TOTAL_NOTEBOOKS})")
    print("=" * 80)
    print(f"• Workers: {TOTAL_WORKERS} Concurrent Threads ({ULTIMATE_WORKERS} Intraday + {FREE_WORKERS} Macro)")
    print(f"• Adjustment: '{ADJUSTMENT_MODE}' (100% Raw Unadjusted Prices)")
    print(f"• Destination HF Repo: {HF_REPO_ID}")
    print("=" * 80)

    # 1. Credentials Setup
    free_cookie = get_node_free_cookie(node_index)
    free_sess = free_cookie["sessionid"]
    free_sign = free_cookie.get("sessionid_sign", "")
    free_jwt = fetch_jwt_token(free_sess, free_sign)

    ult_sess = ULTIMATE_SESSIONID.strip()
    ult_sign = ULTIMATE_SIGN.strip()

    if ult_sess:
        print(f"💎 Premium/Ultimate Account Configured (sessionid: {ult_sess[:8]}...) -> prodata.tradingview.com")
        ult_jwt = fetch_jwt_token(ult_sess, ult_sign)
        ult_server = "prodata"
    else:
        print("ℹ️ TV_ULTIMATE_SESSIONID not configured. Operating in FREE TEST MODE across all workers.")
        ult_sess = free_sess
        ult_sign = free_sign
        ult_jwt = free_jwt
        ult_server = "data"

    print(f"🔑 Node Free Account: sessionid={free_sess[:8]}... (server: data.tradingview.com)")
    print(f"✅ Auth Tokens Initialized.\n")

    # 2. Storage & Checkpoints Setup
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
    print(f"✅ Loaded {len(COMPLETED_TASKS)} previously completed tasks from checkpoint logs.\n")

    # 3. Workload Discovery
    chunks = get_node_workload(node_index=node_index, total_nodes=TOTAL_NOTEBOOKS, chunk_size=CHUNK_SIZE)
    if not chunks:
        print(f"❌ No workload chunks assigned to Node {node_index}. Exiting.")
        return

    # 4. Process Each Chunk
    for chunk_idx, chunk in enumerate(chunks, 1):
        chunk_tag = chunk["chunk_tag"]
        csv_name = chunk["csv_name"]
        symbols = chunk["symbols"]
        is_futures = ("FUTURES" in csv_name.upper()) or any("1!" in s for s in symbols[:5])

        print("\n" + "=" * 78)
        print(f"📂 Chunk {chunk_idx}/{len(chunks)}: {chunk_tag} ({len(symbols)} symbols)")
        print(f"   CSV: {csv_name} | Futures: {is_futures}")
        print("=" * 78)

        with results_lock:
            results_dict.clear()
        with events_lock:
            symbol_events.clear()

        # ----------------------------------------------------------------------
        # PHASE 1: 1D PROBE & CORPORATE ACTIONS (100% Free Accounts)
        # ----------------------------------------------------------------------
        print(f"\n🔍 [PHASE 1/2] Probing 1D Timeframe & Corporate Actions across {TOTAL_WORKERS} Free Workers...")
        phase1_queue = queue.Queue()
        phase1_retry_queue = queue.Queue()
        is_phase1_done = threading.Event()

        queued_1d = 0
        for sym in symbols:
            if f"{sym} (1D)" not in COMPLETED_TASKS:
                phase1_queue.put((sym, is_futures, False))
                queued_1d += 1

        if queued_1d > 0:
            p1_threads = []
            for w_id in range(TOTAL_WORKERS):
                t = threading.Thread(
                    target=phase1_probe_worker_loop,
                    args=(w_id, free_sess, free_sign, free_jwt, "data", phase1_queue, phase1_retry_queue, is_phase1_done),
                    daemon=True
                )
                t.start()
                p1_threads.append(t)
                time.sleep(0.02)

            wait_queues_periodic_log([(phase1_queue, queued_1d)], desc="Phase 1 (1D Probe & Corporate Actions)")

            # Retry transient drops
            retry_round = 1
            while not phase1_retry_queue.empty() and retry_round <= MAX_NETWORK_RETRIES:
                retries = []
                while not phase1_retry_queue.empty():
                    retries.append(phase1_retry_queue.get())
                if not retries: break
                print(f"🔄 Retrying {len(retries)} Phase 1 dropped tasks (Round {retry_round}/{MAX_NETWORK_RETRIES})...")
                time.sleep(2.0 * retry_round)
                for r in retries:
                    phase1_queue.put((r[0], r[1], True))
                wait_queues_periodic_log([(phase1_queue, len(retries))], desc=f"Phase 1 Retry Round {retry_round}")
                retry_round += 1

            # Log permanent failures
            while not phase1_retry_queue.empty():
                f_sym, _, _ = phase1_retry_queue.get()
                log_failure(f_sym, "1D", "server_network_error", f"Exceeded {MAX_NETWORK_RETRIES} retries", log_dir=LOG_DIR)
                with completed_lock:
                    COMPLETED_TASKS.add(f"{f_sym} (1D)")

            is_phase1_done.set()
            for t in p1_threads:
                t.join(timeout=3.0)

        # Identify intraday-eligible symbols via has_intraday
        intraday_symbols = []
        for sym in symbols:
            res = results_dict.get(f"{sym}_1D")
            sym_info = res.get("sym_info") if res else None
            has_intra = sym_info.get("has_intraday", True) if sym_info else True
            if has_intra:
                intraday_symbols.append(sym)

        print(f"   -> {len(intraday_symbols)}/{len(symbols)} symbols support intraday intervals.")

        # Immediately flush 1D Parquet to disk and clear RAM
        flush_and_save_interval("1D", csv_name, chunk_tag, storage, output_dir=OUTPUT_DIR, log_dir=LOG_DIR)
        print("   💾 Phase 1: 1D Parquet serialized to disk. RAM purged.")

        # ----------------------------------------------------------------------
        # PHASE 2: CONCURRENT DEEP INGESTION (INTRADAY + MACRO)
        # ----------------------------------------------------------------------
        print(f"\n⚡ [PHASE 2/2] Launching Deep Historical Ingestion (Day 1 Back-Paging)...")
        macro_queue = queue.Queue()
        intraday_queue = queue.Queue()
        phase2_retry_queue = queue.Queue()
        is_phase2_done = threading.Event()

        # Queue Macro (1W, 1M) for Free workers
        queued_macro = 0
        for m_int in ["1W", "1M"]:
            for sym in symbols:
                if f"{sym} ({m_int})" not in COMPLETED_TASKS:
                    macro_queue.put((sym, m_int, False))
                    queued_macro += 1

        print(f"   • Macro Tasks: {queued_macro} (1W, 1M on {FREE_WORKERS} Free Workers)")

        macro_threads = []
        for w_id in range(ULTIMATE_WORKERS, TOTAL_WORKERS):
            t = threading.Thread(
                target=phase2_deep_worker_loop,
                args=(w_id, free_sess, free_sign, free_jwt, "data", macro_queue, phase2_retry_queue, is_phase2_done),
                daemon=True
            )
            t.start()
            macro_threads.append(t)
            time.sleep(0.02)

        # Launch Intraday Workers (Dedicated to Premium or fallback Free)
        intra_threads = []
        for w_id in range(ULTIMATE_WORKERS):
            t = threading.Thread(
                target=phase2_deep_worker_loop,
                args=(w_id, ult_sess, ult_sign, ult_jwt, ult_server, intraday_queue, phase2_retry_queue, is_phase2_done),
                daemon=True
            )
            t.start()
            intra_threads.append(t)
            time.sleep(0.02)

        # Ingest Intraday Intervals sequentially (240 -> 60 -> 15 -> 5 -> 1)
        # While Macro workers process 1W & 1M in parallel!
        for i_int in INTRADAY_INTERVALS:
            queued_i = 0
            for sym in intraday_symbols:
                if f"{sym} ({i_int})" not in COMPLETED_TASKS:
                    intraday_queue.put((sym, i_int, False))
                    queued_i += 1

            if queued_i > 0:
                print(f"\n⚡ Ingesting Intraday interval {i_int} ({queued_i} tasks)...")
                wait_queues_periodic_log([(intraday_queue, queued_i)], desc=f"Intraday {i_int} Fetch")

                # Retry dropped tasks for this interval
                retry_round = 1
                while not phase2_retry_queue.empty() and retry_round <= MAX_NETWORK_RETRIES:
                    all_retries = []
                    while not phase2_retry_queue.empty():
                        all_retries.append(phase2_retry_queue.get())
                    matching = [r for r in all_retries if r[1] == i_int]
                    others = [r for r in all_retries if r[1] != i_int]
                    for o in others: phase2_retry_queue.put(o)

                    if not matching: break
                    print(f"🔄 Retrying {len(matching)} dropped tasks for {i_int} (Round {retry_round}/{MAX_NETWORK_RETRIES})...")
                    time.sleep(2.0 * retry_round)
                    for r in matching:
                        intraday_queue.put((r[0], i_int, True))
                    wait_queues_periodic_log([(intraday_queue, len(matching))], desc=f"Retry {i_int} Round {retry_round}")
                    retry_round += 1

                # Flush this completed intraday interval to disk & free RAM
                flush_and_save_interval(i_int, csv_name, chunk_tag, storage, output_dir=OUTPUT_DIR, log_dir=LOG_DIR)
                print(f"   💾 Intraday {i_int} complete. Serialized to Parquet & RAM purged.")

        # Finalize Macro queue
        if macro_queue.unfinished_tasks > 0:
            print("\n⏳ Finalizing remaining Macro tasks (1W, 1M)...")
            wait_queues_periodic_log([(macro_queue, macro_queue.unfinished_tasks)], desc="Macro Final Drain")

        # Flush Macro intervals
        for m_int in ["1W", "1M"]:
            flush_and_save_interval(m_int, csv_name, chunk_tag, storage, output_dir=OUTPUT_DIR, log_dir=LOG_DIR)

        # Terminate Phase 2 workers
        is_phase2_done.set()
        for t in intra_threads + macro_threads:
            t.join(timeout=3.0)

        # ----------------------------------------------------------------------
        # SYNCHRONIZE CHUNK DATA & LOGS TO HUGGING FACE
        # ----------------------------------------------------------------------
        local_data_dir = os.path.join(OUTPUT_DIR, "data")
        if os.path.exists(local_data_dir):
            storage.push_data_folder(
                local_data_dir=local_data_dir,
                commit_message=f"Initial deep ingestion: {csv_name} {chunk_tag}"
            )

        storage.push_checkpoint_logs(
            local_log_dir=LOG_DIR,
            notebook_id=NOTEBOOK_ID,
            commit_message=f"Sync logs {csv_name} {chunk_tag}"
        )

        # Clean local output to keep runner disk at 0% usage
        shutil.rmtree(os.path.join(OUTPUT_DIR, "data"), ignore_errors=True)
        print(f"✅ Chunk {chunk_tag} fully ingested, serialized, and synchronized to Hugging Face!\n")

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

    # Final History Squash on Hugging Face to remove any intermediate commit bloat
    storage.squash_history()

    print("=" * 80)
    print(f"🏁 INITIAL RUN COMPLETE FOR NODE {node_index}")
    print("=" * 80)
