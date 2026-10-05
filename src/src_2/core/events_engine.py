"""Engine for aligning and serializing corporate actions to candlestick bars."""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

def attach_events_to_df(
    df: pd.DataFrame,
    raw_events: Optional[Dict[str, Any]],
    interval: str = "1D"
) -> pd.DataFrame:
    """
    Attaches corporate events (splits, dividends, earnings, rolls) to candlestick bars.
    Uses schema-safe scalar and JSON string representations to guarantee zero PyArrow crashes.
    """
    if df.empty:
        return df

    # Initialize event columns with strict types
    if "raw_split" not in df.columns:
        df["raw_split"] = np.nan
    if "raw_dividend" not in df.columns:
        df["raw_dividend"] = None
    if "raw_earnings" not in df.columns:
        df["raw_earnings"] = None
    if "raw_roll" not in df.columns:
        df["raw_roll"] = None

    if not raw_events:
        return df

    times = df["time"].values
    n_times = len(times)
    if n_times == 0:
        return df

    max_delta = 8 * 86400 if interval == "1W" else (35 * 86400 if interval == "1M" else 86400)

    def find_target_idx(event_ts_sec: float) -> Optional[int]:
        if interval in ["1D", "1", "5", "15", "60", "240"]:
            # Day-aligned match: candle on or after start of UTC day
            start_of_day = (int(event_ts_sec) // 86400) * 86400
            end_of_day = start_of_day + 86400
            idx = np.searchsorted(times, start_of_day)
            if idx < n_times and times[idx] < end_of_day:
                return int(idx)
            return None
        else:
            # Macro (1W, 1M): candle that opened on or before event timestamp
            cand_idx = np.searchsorted(times, event_ts_sec, side="right") - 1
            if 0 <= cand_idx < n_times and (event_ts_sec - times[cand_idx]) <= max_delta:
                return int(cand_idx)
            return None

    # 1. Splits: item = [ts_sec, split_factor]
    for sp in raw_events.get("splits", []):
        if len(sp) >= 2 and sp[0] is not None:
            sp_ts, sp_factor = float(sp[0]), float(sp[1])
            idx = find_target_idx(sp_ts)
            if idx is not None:
                df.iat[idx, df.columns.get_loc("raw_split")] = sp_factor

    # 2. Dividends: v = [bar_ts, gross_payout, ex_date_ms, adjusted_payout, payment_date_ms]
    for d in raw_events.get("dividends", []):
        v = d.get("v", [])
        if len(v) >= 3 and v[2]:
            ex_ts = float(v[2]) / 1000.0
            idx = find_target_idx(ex_ts)
            if idx is not None:
                div_payload = {
                    "gross": float(v[1]) if (len(v) > 1 and v[1] is not None and v[1] < 1e5) else None,
                    "ex_date_ms": int(v[2]),
                    "adj_payout": float(v[3]) if (len(v) > 3 and v[3] is not None and v[3] < 1e5) else None,
                    "payment_date_ms": int(v[4]) if (len(v) > 4 and v[4]) else None
                }
                df.iat[idx, df.columns.get_loc("raw_dividend")] = json.dumps(div_payload, separators=(',', ':'))

    # 3. Earnings: v = [bar_ts, std_eps, est_eps, period_s, ann_ms, rep_eps, rev_est_M, rev_act_M, flags]
    for e in raw_events.get("earnings", []):
        v = e.get("v", [])
        if len(v) >= 5 and v[4]:
            ann_ts = float(v[4]) / 1000.0
            idx = find_target_idx(ann_ts)
            if idx is not None:
                earn_payload = {
                    "std_eps": float(v[1]) if (len(v) > 1 and v[1] is not None and v[1] < 1e5) else None,
                    "est_eps": float(v[2]) if (len(v) > 2 and v[2] is not None and v[2] < 1e5) else None,
                    "period_s": int(v[3]) if (len(v) > 3 and v[3]) else None,
                    "ann_ms": int(v[4]),
                    "rep_eps": float(v[5]) if (len(v) > 5 and v[5] is not None and v[5] < 1e5) else None,
                    "rev_est_m": float(v[6]) if (len(v) > 6 and v[6] is not None and v[6] < 1e9) else None,
                    "rev_act_m": float(v[7]) if (len(v) > 7 and v[7] is not None and v[7] < 1e9) else None,
                    "timing": "AMC" if (len(v) > 8 and v[8] in [20, 22]) else ("BMO" if (len(v) > 8 and v[8] in [10, 11]) else "UNKNOWN")
                }
                df.iat[idx, df.columns.get_loc("raw_earnings")] = json.dumps(earn_payload, separators=(',', ':'))

    # 4. Continuous Rolls: v = [bar_ts, from_code, to_code, switch_date_yyyymmdd]
    for r in raw_events.get("rolls", []):
        v = r.get("v", [])
        if len(v) >= 4 and v[0] is not None:
            roll_ts = float(v[0])
            idx = find_target_idx(roll_ts)
            if idx is not None:
                roll_payload = {
                    "from_contract": str(int(v[1])) if v[1] is not None else "",
                    "to_contract": str(int(v[2])) if v[2] is not None else "",
                    "switch_date": int(v[3]) if v[3] is not None else 0
                }
                df.iat[idx, df.columns.get_loc("raw_roll")] = json.dumps(roll_payload, separators=(',', ':'))

    return df
