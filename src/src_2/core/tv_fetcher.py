"""TradingView high-level data fetchers for 1D probing, corporate actions, and deep historical back-paging."""
import time
import random
import string
import json
from typing import Dict, Any, List, Tuple, Optional
from .tv_client import TradingViewSocketClient

def generate_session_id(prefix: str = "cs") -> str:
    """Generates a random unique TradingView session ID."""
    rand_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{rand_suffix}"

class TradingView1DProbeFetcher:
    """
    Probes 1D timeframe for a symbol:
    1. Quote Session: Retrieves 100% of historical stock splits (last_splits back to IPO).
    2. Chart Session: Retrieves full 1D OHLCV bars and attaches studies:
       - Equities/ETFs: Dividends@tv-basicstudies-277 & Earnings@tv-basicstudies-277
       - Futures: BarSetContinuousRollDates@tv-corestudies-48 (100% of contract switch dates)
    3. Symbol Resolution: Extracts metadata, particularly `has_intraday`.
    """

    def __init__(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token",
        server: str = "data"
    ):
        self.client = TradingViewSocketClient(sessionid, sessionid_sign, jwt_token, server=server)

    def fetch_1d_and_events(
        self,
        symbol: str,
        is_futures: bool = False,
        timeout: float = 12.0
    ) -> Tuple[List[list], Dict[str, list], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        bars = []
        raw_events = {
            "splits": [],
            "dividends": [],
            "earnings": [],
            "rolls": []
        }
        sym_info = None
        error_cat = None
        error_det = None

        qs_id = generate_session_id("qs_probe")
        cs_id = generate_session_id("cs_probe")

        try:
            # 1. Quote Session for stock splits
            if not is_futures:
                self.client.send("quote_create_session", [qs_id])
                self.client.send("quote_set_fields", [qs_id, "base_name", "last_splits"])
                self.client.send("quote_add_symbols", [qs_id, symbol])

            # 2. Chart Session for 1D bars & corporate action studies
            self.client.send("chart_create_session", [cs_id, ""])

            # Raw unadjusted prices
            sym_spec = {"symbol": symbol, "adjustment": "none"}
            self.client.send("resolve_symbol", [cs_id, "sds_sym_1", f"={json.dumps(sym_spec)}"])
            self.client.send("create_series", [cs_id, "sds_1", "s1", "sds_sym_1", "1D", 2000000, ""])

            required_studies = set()
            completed_studies = set()

            if not is_futures:
                self.client.send("create_study", [cs_id, "st_div", "st1", "sds_1", "Dividends@tv-basicstudies-277", {}])
                self.client.send("create_study", [cs_id, "st_earn", "st1", "sds_1", "Earnings@tv-basicstudies-277", {}])
                required_studies.update(["st_div", "st_earn"])
            else:
                self.client.send("create_study", [cs_id, "st_roll", "st1", "sds_1", "BarSetContinuousRollDates@tv-corestudies-48", {"currenttime": "now"}])
                required_studies.add("st_roll")

            loop_start = time.time()
            series_done = False

            while time.time() - loop_start < timeout:
                try:
                    messages = self.client.recv_messages(timeout=2.0)
                except Exception:
                    continue

                for m in messages:
                    meth = m.get("m")
                    params = m.get("p", [])

                    # Quote session splits
                    if meth == "qsd" and params and params[0] == qs_id:
                        val = params[1].get("v", {})
                        if "last_splits" in val:
                            raw_events["splits"] = val["last_splits"]

                    # Symbol metadata
                    elif meth == "symbol_resolved" and params and params[0] == cs_id:
                        sym_info = params[2]

                    # Bars and studies
                    elif meth in ["timescale_update", "du"] and params and params[0] == cs_id:
                        p = params[1]
                        if "sds_1" in p:
                            for b in p["sds_1"].get("s", []):
                                v = b.get("v", [])
                                if len(v) >= 5:
                                    bars.append([v[0], v[1], v[2], v[3], v[4], v[5] if len(v) > 5 else 0.0])
                        if "st_div" in p:
                            raw_events["dividends"].extend(p["st_div"].get("st", []))
                        if "st_earn" in p:
                            raw_events["earnings"].extend(p["st_earn"].get("st", []))
                        if "st_roll" in p:
                            raw_events["rolls"].extend(p["st_roll"].get("st", []))

                    elif meth == "series_completed" and params and params[0] == cs_id:
                        series_done = True

                    elif meth == "study_completed" and params and params[0] == cs_id:
                        completed_studies.add(params[1])

                    elif meth in ["symbol_error", "series_error"] and params and params[0] == cs_id:
                        error_cat = "tradingview_message"
                        error_det = meth
                        break

                if error_cat:
                    break

                if series_done and (not required_studies or completed_studies >= required_studies):
                    break

            if not bars and not error_cat:
                if series_done:
                    error_cat = "tradingview_message"
                    error_det = "series_completed_but_empty"
                else:
                    error_cat = "server_network_error"
                    error_det = "socket_timeout"

        except Exception as e:
            error_cat = "server_network_error"
            error_det = f"{type(e).__name__}: {e}"
            self.client.close()

        finally:
            try:
                self.client.send("chart_delete_session", [cs_id])
                if not is_futures:
                    self.client.send("quote_delete_session", [qs_id])
            except Exception:
                self.client.close()

        return bars, raw_events, sym_info, error_cat, error_det

    def close(self):
        self.client.close()


class TradingViewDeepFetcher:
    """
    Dedicated fetcher for deep historical back-paging loops:
    - Uses unadjusted raw pricing ("adjustment": "none")
    - Backward pagination with request_more_data(2000000)
    - Loops backwards until Day 1 inception date or max_batches limit
    - Ideal for Intraday (1, 5, 15, 60, 240) and Macro (1W, 1M)
    """

    def __init__(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token",
        server: str = "data"
    ):
        self.client = TradingViewSocketClient(sessionid, sessionid_sign, jwt_token, server=server)

    def fetch_deep_history(
        self,
        symbol: str,
        interval: str,
        max_batches: int = 100,
        timeout_per_batch: float = 25.0
    ) -> Tuple[List[list], Optional[Dict[str, Any]], str, Optional[str], Optional[str]]:
        bars_dict: Dict[int, list] = {}
        sym_info = None
        status = "UNKNOWN"
        error_cat = None
        error_det = None

        cs_id = generate_session_id("cs_deep")

        try:
            self.client.send("chart_create_session", [cs_id, ""])

            sym_spec = {"symbol": symbol, "adjustment": "none"}
            self.client.send("resolve_symbol", [cs_id, "ser_1", f"={json.dumps(sym_spec)}"])
            self.client.send("create_series", [cs_id, "$prices", "s1", "ser_1", interval, 2000000])

            batch_count = 0
            new_bars_in_batch = 0
            loop_start = time.time()

            while time.time() - loop_start < timeout_per_batch:
                try:
                    messages = self.client.recv_messages(timeout=2.5)
                except Exception:
                    continue

                series_done = False

                for m in messages:
                    meth = m.get("m")
                    params = m.get("p", [])

                    if meth == "symbol_resolved" and params and params[0] == cs_id:
                        sym_info = params[2]

                    elif meth in ["timescale_update", "du"] and params and params[0] == cs_id:
                        payload = params[1]
                        series = payload.get("$prices", {}).get("s", []) or payload.get("s1", {}).get("s", [])
                        for bar in series:
                            v = bar.get("v", [])
                            if len(v) >= 5 and v[0] not in bars_dict:
                                bars_dict[v[0]] = [v[0], v[1], v[2], v[3], v[4], v[5] if len(v) > 5 else 0.0]
                                new_bars_in_batch += 1

                    elif meth == "series_completed" and params and params[0] == cs_id:
                        series_done = True
                        batch_count += 1

                    elif meth in ["symbol_error", "series_error"] and params and params[0] == cs_id:
                        error_cat = "tradingview_message"
                        error_det = meth
                        status = f"ERROR: {meth}"
                        break

                    elif meth in ["critical_error", "error"]:
                        error_cat = "server_network_error"
                        error_det = str(params)
                        status = f"CRITICAL: {params}"
                        break

                if error_cat:
                    break

                if series_done:
                    # If more bars arrived and batch cap not reached, request earlier history!
                    if new_bars_in_batch > 0 and batch_count < max_batches:
                        new_bars_in_batch = 0  # Crucial: reset for the next batch!
                        self.client.send("request_more_data", [cs_id, "$prices", 2000000])
                        loop_start = time.time()
                        continue
                    else:
                        # Reached Day 1 or maximum configured depth
                        status = "OK"
                        break

            if bars_dict:
                status = "OK"
            elif not error_cat:
                if status != "OK":
                    error_cat = "tradingview_timeout"
                    error_det = "batch_timeout"
                else:
                    error_cat = "tradingview_message"
                    error_det = "series_completed_but_empty"

        except Exception as e:
            error_cat = "server_network_error"
            error_det = f"{type(e).__name__}: {e}"
            status = f"EXCEPTION: {e}"
            self.client.close()

        finally:
            try:
                self.client.send("chart_delete_session", [cs_id])
            except Exception:
                self.client.close()

        sorted_bars = sorted(bars_dict.values(), key=lambda x: x[0]) if bars_dict else []
        return sorted_bars, sym_info, status, error_cat, error_det

    def update_credentials(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token",
        server: str = "data"
    ):
        self.client.update_credentials(sessionid, sessionid_sign, jwt_token, server=server)

    def keep_alive(self):
        self.client.keep_alive()

    def close(self):
        self.client.close()


class TradingViewWeeklyFetcher:
    """
    Dedicated fetcher for fast weekly maintenance:
    - Fetches recent candlesticks (e.g. last 300 bars)
    - Probes 1D for corporate action updates
    - 100% Free accounts
    """

    def __init__(
        self,
        sessionid: str,
        sessionid_sign: str = "",
        jwt_token: str = "unauthorized_user_token"
    ):
        self.client = TradingViewSocketClient(sessionid, sessionid_sign, jwt_token, server="data")

    def fetch_recent_bars(
        self,
        symbol: str,
        interval: str,
        n_bars: int = 300,
        timeout: float = 10.0
    ) -> Tuple[List[list], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        bars = []
        sym_info = None
        error_cat = None
        error_det = None

        cs_id = generate_session_id("cs_weekly")

        try:
            self.client.send("chart_create_session", [cs_id, ""])
            sym_spec = {"symbol": symbol, "adjustment": "none"}
            self.client.send("resolve_symbol", [cs_id, "ser_1", f"={json.dumps(sym_spec)}"])
            self.client.send("create_series", [cs_id, "$prices", "s1", "ser_1", interval, n_bars])

            loop_start = time.time()
            series_done = False

            while time.time() - loop_start < timeout:
                try:
                    messages = self.client.recv_messages(timeout=2.0)
                except Exception:
                    continue

                for m in messages:
                    meth = m.get("m")
                    params = m.get("p", [])

                    if meth == "symbol_resolved" and params and params[0] == cs_id:
                        sym_info = params[2]

                    elif meth in ["timescale_update", "du"] and params and params[0] == cs_id:
                        payload = params[1]
                        series = payload.get("$prices", {}).get("s", []) or payload.get("s1", {}).get("s", [])
                        for bar in series:
                            v = bar.get("v", [])
                            if len(v) >= 5:
                                bars.append([v[0], v[1], v[2], v[3], v[4], v[5] if len(v) > 5 else 0.0])

                    elif meth == "series_completed" and params and params[0] == cs_id:
                        series_done = True
                        break

                    elif meth in ["symbol_error", "series_error"] and params and params[0] == cs_id:
                        error_cat = "tradingview_message"
                        error_det = meth
                        break

                if error_cat or series_done:
                    break

            if not bars and not error_cat:
                if series_done:
                    error_cat = "tradingview_message"
                    error_det = "series_completed_but_empty"
                else:
                    error_cat = "server_network_error"
                    error_det = "socket_timeout"

        except Exception as e:
            error_cat = "server_network_error"
            error_det = f"{type(e).__name__}: {e}"
            self.client.close()

        finally:
            try:
                self.client.send("chart_delete_session", [cs_id])
            except Exception:
                self.client.close()

        bars_sorted = sorted(bars, key=lambda x: x[0]) if bars else []
        return bars_sorted, sym_info, error_cat, error_det

    def keep_alive(self):
        self.client.keep_alive()

    def close(self):
        self.client.close()
