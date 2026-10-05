"""Structured progress tracking and rate-limited logging for long-running Kaggle runs."""
import os
import time
import queue
from typing import List, Tuple, Any

def log_failure(
    symbol: str,
    interval: str,
    category: str,
    detail: Any,
    log_dir: str = "log"
):
    """Logs structured failure to failures.log without crashing."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    clean_detail = str(detail).replace('\n', ' ').replace('\r', '').strip()
    msg = f"{timestamp}, {symbol}, {interval}, {category}, {clean_detail}"
    try:
        with open(os.path.join(log_dir, "failures.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def record_checkpoint(
    symbol: str,
    interval: str,
    bars_count: int,
    chunk_path: str,
    log_dir: str = "log"
):
    """Records successfully completed task to checkpoint.log."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{timestamp}, {symbol}, {interval}, checkpoint, Bars: {bars_count}, {chunk_path}"
    try:
        with open(os.path.join(log_dir, "checkpoint.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def wait_queues_periodic_log(
    queues_with_totals: List[Tuple[queue.Queue, int]],
    desc: str = "Processing",
    interval_sec: float = 60.0
):
    """
    Monitors queues and logs clean progress updates periodically (every interval_sec).
    Prevents Kaggle 100,000-line output buffer overflows and log truncation.
    """
    total = sum(t for _, t in queues_with_totals)
    if total <= 0:
        return

    start_time = time.time()
    last_log_time = start_time

    while True:
        current_unfinished = sum(q.unfinished_tasks for q, _ in queues_with_totals)
        completed = total - current_unfinished
        now = time.time()

        if current_unfinished <= 0:
            elapsed = now - start_time
            overall_speed = completed / elapsed if elapsed > 0 else 0
            print(f"   [{time.strftime('%H:%M:%S')}] ✅ {desc}: {completed:,}/{total:,} (100.0%) | "
                  f"Avg Speed: {overall_speed:.1f} tasks/s | Elapsed: {int(elapsed)}s")
            break

        if now - last_log_time >= interval_sec:
            elapsed = now - start_time
            overall_speed = completed / elapsed if elapsed > 0 else 0
            remaining = current_unfinished
            eta_sec = int(remaining / overall_speed) if overall_speed > 0 else 0
            eta_str = f"{eta_sec // 60}m {eta_sec % 60}s" if eta_sec >= 60 else f"{eta_sec}s"
            pct = (completed / total * 100.0) if total > 0 else 0.0

            print(f"   [{time.strftime('%H:%M:%S')}] ⏳ {desc}: {completed:,}/{total:,} ({pct:.1f}%) | "
                  f"Speed: {overall_speed:.1f} tasks/s | ETA: {eta_str} | Pending: {remaining:,}")
            last_log_time = now

        time.sleep(1.0)
