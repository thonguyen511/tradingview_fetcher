import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

import json
import re
import requests
from typing import Dict, List, Optional

# Parse CLI arguments if provided
for i, arg in enumerate(sys.argv):
    if arg in ["--node", "-n", "--index"] and i + 1 < len(sys.argv):
        os.environ["NOTEBOOK_INDEX"] = sys.argv[i + 1]
    elif arg.isdigit() and i > 0 and (sys.argv[i - 1].endswith(".py") or sys.argv[i - 1] in ["-m", "main"]):
        os.environ["NOTEBOOK_INDEX"] = arg
    elif arg in ["--chunk-size", "-c"] and i + 1 < len(sys.argv):
        os.environ["CHUNK_SIZE"] = sys.argv[i + 1]
    elif arg in ["--encrypt", "-e"] and i + 1 < len(sys.argv):
        os.environ["ENCRYPTION_MODE"] = sys.argv[i + 1]

# ------------------------------------------------------------------------------
# 1. RUNTIME & CONCURRENCY
# ------------------------------------------------------------------------------
TOTAL_NOTEBOOKS = int(os.environ.get("TOTAL_NOTEBOOKS", 20))
NOTEBOOK_INDEX = int(os.environ.get("NOTEBOOK_INDEX", 0))  # 0 to 19
NOTEBOOK_ID = f"NODE_{NOTEBOOK_INDEX}"

# Concurrency per notebook
# Initial run: 5 paid intraday workers + 3 free macro workers (or 8 free workers in testing)
# Weekly run: 3 free workers per node
ULTIMATE_WORKERS = int(os.environ.get("ULTIMATE_WORKERS", 5))
FREE_WORKERS = int(os.environ.get("FREE_WORKERS", 3))
TOTAL_WORKERS = int(os.environ.get("TOTAL_WORKERS", ULTIMATE_WORKERS + FREE_WORKERS))

# Chunking & Retries
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 5000))
MAX_NETWORK_RETRIES = int(os.environ.get("MAX_NETWORK_RETRIES", 5))
MAX_HISTORY_BATCHES = int(os.environ.get("MAX_HISTORY_BATCHES", 1000))
WEEKLY_N_BARS = int(os.environ.get("WEEKLY_N_BARS", 300))

# Price Adjustment Mode: strictly "none" for 100% raw unadjusted exchange data
ADJUSTMENT_MODE = "none"

# Supported Timeframes
INTRADAY_INTERVALS = ["1", "5", "15", "60", "240"]
DAILY_INTERVALS = ["1D", "1W", "1M"]
ALL_INTERVALS = DAILY_INTERVALS + INTRADAY_INTERVALS

# ------------------------------------------------------------------------------
# 2. LOCAL & WORKSPACE PATHS
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
CSV_DIR = os.path.join(DATA_DIR, "AIO_CSV")
MAP_FILE = os.path.join(CSV_DIR, "map.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
LOG_DIR = os.environ.get("LOG_DIR", os.path.join("log", NOTEBOOK_ID))

# ------------------------------------------------------------------------------
# 3. KAGGLE ACCOUNTS (4 Accounts x 5 Notebooks = 20 Concurrent Workers)
# ------------------------------------------------------------------------------
KAGGLE_ACCOUNTS = [
    {"username": "thonguyen511", "token": "KGAT_9a29974782c1c24e1e9bccf69a51e21d"},
    {"username": "hero0511acc", "token": "KGAT_8aad194411dbee85815ec01a682f3047"},
    {"username": "tqunacc", "token": "KGAT_46cc7c9f3ebb3239e81b25d0f7f6ae97"},
    {"username": "acc3conheo", "token": "KGAT_9b980e55d4c06e72e229bc520ae1ac98"},
]

def get_kaggle_account_for_node(node_idx: int = NOTEBOOK_INDEX) -> dict:
    """Maps node index (0..19) to the corresponding Kaggle account."""
    acc_idx = (node_idx // 5) % len(KAGGLE_ACCOUNTS)
    return KAGGLE_ACCOUNTS[acc_idx]

# ------------------------------------------------------------------------------
# 4. HUGGING FACE SETTINGS
# ------------------------------------------------------------------------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "thonguyen511/temp")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ------------------------------------------------------------------------------
# 5. TRADINGVIEW CREDENTIALS (20 FREE ACCOUNTS + 1 SHARED PREMIUM/ULTIMATE)
# ------------------------------------------------------------------------------
FREE_COOKIES: List[Dict[str, str]] = [
    {"sessionid": "nyd2xhhge1eqe1nz7vlt56bjq7gii7rv", "sessionid_sign": "v3:HN1KRBXurQgBil6iUvZ/VFBTmRGD/X+2PZ9HHLn1sso="},
    {"sessionid": "6oju8gg72zx3inpzdam4hg1owv9b7b4v", "sessionid_sign": "v3:IlDG6eb3hNM1MeKfrhkD1SC59Yb3pyqf2DE9LVJ4QtY="},
    {"sessionid": "r68mmxpvvg6neoc5azk0zubo3q9aj058", "sessionid_sign": "v3:0kmDxH+xdl8DKWy+vhLcSUcw4fAqC6L47rtfJ3K9DNs="},
    {"sessionid": "2xx7a6j9cmfpuq5w5aff9eoum6kccui8", "sessionid_sign": "v3:dojiB43SFWBHWYQgBG7EtC44iLXOMqpfOcx5hKfPrE8="},
    {"sessionid": "8zdncz5k1bdxrkx4fc5i4aeng99xypgg", "sessionid_sign": "v3:gji0jmijT6/fvTy2QK7vWX+wJ9xMYtjGMKHi4Vk+370="},
    {"sessionid": "m56urgnz31kkubfu4zyrqi0wfhkwzyxh", "sessionid_sign": "v3:76wkQm2BMYNOD9ru/isHX2ZuqEOWaXibVwC7U+G3XBw="},
    {"sessionid": "36906qz62lf46hm5e88gm44qm7i3ooqu", "sessionid_sign": "v3:pFwesXZ8uWB4hh5asBEOCyfIp77OQ9CWCMn6yS6ZGy8="},
    {"sessionid": "mutvuw4vczaqmpvgaih4gw5350lns4kb", "sessionid_sign": "v3:2xeMpVLXD2lkDXKHl3eT2JYQzUQGTLVDoENt1+s+Nh4="},
    {"sessionid": "jn8enyc1lt2r656lh15u1qb4340al4r2", "sessionid_sign": "v3:2P5aycMG9Sf3VrvYdDtbMssQLJsrb9EzaOUGvoo0qns="},
    {"sessionid": "r2lauc0jxqri7c9kezdoy6t6e71opkay", "sessionid_sign": "v3:2b/0b4MomCx+76tmfF3uQItmh8MQK2GKymuXsKqh9dQ="},
    {"sessionid": "od5mt2i8lo5mg4k4jie4cxneu89kl3kr", "sessionid_sign": "v3:uhpG+CL/Z8eBuOVWlqqFYu3Yupq0j+h4wnjeQe6NlaE="},
    {"sessionid": "on8jljxtijcz08nxvtdsafnvpgpca6wu", "sessionid_sign": "v3:zdrSEQxqSJdqOcRxWD8xLajJcgXsyh1s9CgDXQsnZtc="},
    {"sessionid": "5m1xyo8gi0omogyef3roflrteexrt3zv", "sessionid_sign": "v3:pfgppWB0cO4xDYcI1Y6y/mY6rhFOklANzNSEtnUQXdI="},
    {"sessionid": "vxvdynua8osiyf0fwfn2x2yvy5q8n8cs", "sessionid_sign": "v3:eI4Mg4fXQGTAJ1SJ9rlxiskSr+ngc3on66Nd0IbmTr4="},
    {"sessionid": "08sqlp86hd3bduscbjv159tq0ds9adzt", "sessionid_sign": "v3:OqBdhTW3DSjVzJRjH0mSHUYdStCyrvZ3MoXJmtmD3I0="},
    {"sessionid": "qatorul4xe9ah6rzb1vkk5xtuwt7uygq", "sessionid_sign": "v3:FOhRQDa0O2Oo6IrnHcpizxSSlw4CgFdOtYs4kf/25wE="},
    {"sessionid": "c6e83lfw9hop1hld0qsft4634zah8ucp", "sessionid_sign": "v3:mE0TcSuxquhCXT7ysQaGtiq+vz2IGLwSj7NyKiWPk5k="},
    {"sessionid": "rdu9xyuda4c2sawsigzowj8xyynkh1oj", "sessionid_sign": "v3:1fhMVrppO2SEAihS7O/BB7UOE5Q+WUBwgwVa1AJP9Tk="},
    {"sessionid": "6b51nxhkk9lvl54o7lrtnz86yudctaej", "sessionid_sign": "v3:ZXq8jO1JJr3/m3IDN/8XizhFNhe+ytm6nBCLouDXagg="},
    {"sessionid": "xa5a7083pvwu57qrsioayf3mw4ofnevq", "sessionid_sign": "v3:gg1koNMDyACH4F4EDuNRQtwYNnNe8Ss6NrFkYObVpM0="}
]

ULTIMATE_SESSIONID = os.environ.get("TV_ULTIMATE_SESSIONID", "")
ULTIMATE_SIGN = os.environ.get("TV_ULTIMATE_SIGN", "")

# Auto-detect secrets inside Kaggle environments
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    if not HF_TOKEN:
        try: HF_TOKEN = _secrets.get_secret("HF_TOKEN")
        except Exception: pass
    if not ULTIMATE_SESSIONID:
        try: ULTIMATE_SESSIONID = _secrets.get_secret("TV_ULTIMATE_SESSIONID")
        except Exception: pass
    if not ULTIMATE_SIGN:
        try: ULTIMATE_SIGN = _secrets.get_secret("TV_ULTIMATE_SIGN")
        except Exception: pass
except Exception:
    pass

# Sanitize ultimate credentials
if ULTIMATE_SESSIONID.lower() in ["", "none", "false", "your_ultimate_sessionid_here"]:
    ULTIMATE_SESSIONID = ""

def get_node_free_cookie(node_idx: int = NOTEBOOK_INDEX) -> Dict[str, str]:
    """Returns the dedicated free cookie assigned to this node index."""
    idx = node_idx % len(FREE_COOKIES)
    return FREE_COOKIES[idx]

def fetch_jwt_token(sessionid: str, sessionid_sign: str = "") -> str:
    """Queries TradingView frontend to obtain a fresh JWT auth_token."""
    cookies = {"sessionid": sessionid}
    if sessionid_sign:
        cookies["sessionid_sign"] = sessionid_sign
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://www.tradingview.com"
    }
    try:
        resp = requests.get("https://www.tradingview.com/", cookies=cookies, headers=headers, timeout=12)
        match = re.search(r'"auth_token":"([^"]+)"', resp.text)
        return match.group(1) if match else "unauthorized_user_token"
    except Exception:
        return "unauthorized_user_token"

# Optional encryption settings
DATA_PASSWORD = os.environ.get("DATA_PASSWORD", "")
ENCRYPTION_MODE = os.environ.get("ENCRYPTION_MODE", "AUTO").strip().upper()
ENABLE_ENCRYPTION = bool(DATA_PASSWORD) and (ENCRYPTION_MODE not in ["FALSE", "NO", "0", "OFF"])
