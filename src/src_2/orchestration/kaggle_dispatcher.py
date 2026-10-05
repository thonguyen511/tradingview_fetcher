"""Kaggle Dispatcher: Controls and monitors 20 concurrent notebook runs across 4 Kaggle accounts."""
import os
import sys
import json
import time
import shutil
import subprocess
from typing import List, Dict, Optional
from ..config import KAGGLE_ACCOUNTS, TOTAL_NOTEBOOKS, HF_REPO_ID

def set_kaggle_env(account: dict):
    """Sets environment variables for Kaggle CLI authentication for a specific account."""
    os.environ['KAGGLE_USERNAME'] = account['username']
    os.environ['KAGGLE_KEY'] = account['token'].replace("KGAT_", "")
    os.environ['KAGGLE_API_TOKEN'] = account['token']

def generate_runner_script(
    node_index: int,
    mode: str = "weekly",
    symbols_repo: str = "https://github.com/thonguyen511/tradingview_aio_symbols.git"
) -> str:
    """Generates the self-contained Python script to run on Kaggle."""
    hf_token = os.environ.get("HF_TOKEN", "")
    gh_pat = os.environ.get("GH_PAT", "")
    data_password = os.environ.get("DATA_PASSWORD", "")
    tv_sessionid = os.environ.get("TV_ULTIMATE_SESSIONID", "")
    tv_sign = os.environ.get("TV_ULTIMATE_SIGN", "")

    return f'''# Auto-generated runner for Node {node_index} ({mode} run)
import os
import sys
import subprocess

print("=" * 80)
print("🚀 Starting Automated Node {node_index} ({mode}) on Kaggle")
print("=" * 80)

# 1. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "websocket-client", "huggingface_hub", "cryptography", "pandas", "pyarrow", "requests"], check=True)

# 2. Clone CSV symbols and map.json from tradingview_aio_symbols
work_dir = "/kaggle/working"
csv_target = os.path.join(work_dir, "data", "AIO_CSV")
os.makedirs(os.path.dirname(csv_target), exist_ok=True)

if not os.path.exists(csv_target):
    print("📥 Cloning CSV symbols and map.json from {symbols_repo}...")
    subprocess.run(["git", "clone", "--depth", "1", "{symbols_repo}", csv_target], check=True)
else:
    print("🔄 Pulling latest symbols updates...")
    subprocess.run(["git", "-C", csv_target, "pull"], check=False)

# Add working directory to pythonpath so bundled src is found immediately
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)

os.environ["NOTEBOOK_INDEX"] = "{node_index}"
os.environ["TOTAL_NOTEBOOKS"] = "{TOTAL_NOTEBOOKS}"
if "{hf_token}" and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "{hf_token}"
if "{gh_pat}" and not os.environ.get("GH_PAT"):
    os.environ["GH_PAT"] = "{gh_pat}"
if "{data_password}" and not os.environ.get("DATA_PASSWORD"):
    os.environ["DATA_PASSWORD"] = "{data_password}"
if "{tv_sessionid}" and not os.environ.get("TV_ULTIMATE_SESSIONID"):
    os.environ["TV_ULTIMATE_SESSIONID"] = "{tv_sessionid}"
if "{tv_sign}" and not os.environ.get("TV_ULTIMATE_SIGN"):
    os.environ["TV_ULTIMATE_SIGN"] = "{tv_sign}"

# Helper to re-trigger GitHub Actions workflow before 12h timeout
def trigger_github_restart(node_idx, mode):
    import requests
    gh_pat = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    gh_repo = os.environ.get("GH_REPO", "thonguyen511/tradingview_fetcher")

    if not gh_pat:
        try:
            from kaggle_secrets import UserSecretsClient
            gh_pat = UserSecretsClient().get_secret("GH_PAT")
        except Exception:
            pass

    if gh_pat:
        url = f"https://api.github.com/repos/{gh_repo}/actions/workflows/node_runner.yml/dispatches"
        headers = {
            "Authorization": f"Bearer {gh_pat}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {
            "ref": "main",
            "inputs": {{"node": str(node_idx), "mode": mode}}
        }
        try:
            res = requests.post(url, headers=headers, json=data, timeout=15)
            print(f"📡 Re-triggered GitHub Actions for Node {{node_idx}}: HTTP {{res.status_code}}")
        except Exception as e:
            print(f"⚠️ Failed to re-trigger GitHub Actions: {{e}}")
    else:
        print("⚠️ GH_PAT not found in Kaggle Secrets. Cannot trigger GitHub Actions automatically.")

try:
    if "{mode}" == "initial":
        from src.src_2.initial_run.pipeline import run_initial_pipeline
        run_initial_pipeline(node_index={node_index}, restart_callback=lambda: trigger_github_restart({node_index}, "{mode}"))
    else:
        from src.src_2.weekly_run.pipeline import run_weekly_pipeline
        run_weekly_pipeline(node_index={node_index}, restart_callback=lambda: trigger_github_restart({node_index}, "{mode}"))
except Exception as e:
    print(f"❌ Execution failed on Node {node_index}: {{e}}")
    raise
'''

def prepare_kernel_folder(
    node_index: int,
    mode: str,
    output_dir: str
) -> str:
    """Prepares directory containing kernel-metadata.json, main.py, and bundled src for kaggle kernels push."""
    acc_idx = (node_index // 5) % len(KAGGLE_ACCOUNTS)
    account = KAGGLE_ACCOUNTS[acc_idx]
    kernel_slug = f"tv-ingestion-node-{node_index}"

    folder = os.path.join(output_dir, f"kernel_node_{node_index}")
    os.makedirs(folder, exist_ok=True)

    metadata = {
        "id": f"{account['username']}/{kernel_slug}",
        "title": f"TV Ingestion Node {node_index} {mode.capitalize()}",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }

    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Bundle src/ directory directly into the kernel folder so Kaggle executes it immediately
    src_origin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_target = os.path.join(folder, "src")
    if os.path.exists(src_target):
        shutil.rmtree(src_target)
    shutil.copytree(src_origin, src_target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    script_content = generate_runner_script(node_index=node_index, mode=mode)
    with open(os.path.join(folder, "main.py"), "w", encoding="utf-8") as f:
        f.write(script_content)

    return folder

def dispatch_all_nodes(
    mode: str = "weekly",
    nodes: Optional[List[int]] = None,
    temp_dir: str = "temp/kaggle_kernels"
):
    """Pushes and launches the 20 notebooks across the 4 Kaggle accounts."""
    target_nodes = nodes if nodes is not None else list(range(TOTAL_NOTEBOOKS))
    print("=" * 80)
    print(f"🚀 DISPATCHING {len(target_nodes)} KAGGLE RUNNERS (Mode: {mode.upper()})")
    print("=" * 80)

    for node_idx in target_nodes:
        acc_idx = (node_idx // 5) % len(KAGGLE_ACCOUNTS)
        account = KAGGLE_ACCOUNTS[acc_idx]
        set_kaggle_env(account)

        kernel_dir = prepare_kernel_folder(node_idx, mode, temp_dir)
        print(f"\n[{node_idx + 1}/{len(target_nodes)}] Pushing Node {node_idx} to account '{account['username']}'...")

        res = subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"   ✅ Successfully pushed and started Node {node_idx}!")
        else:
            print(f"   ⚠️ Push failed: {res.stderr.strip() or res.stdout.strip()}")

        time.sleep(2)  # Avoid hitting Kaggle rate limit

    print("\n" + "=" * 80)
    print("🏁 ALL RUNNERS DISPATCHED")
    print("=" * 80)

def check_all_status(temp_dir: str = "temp/kaggle_kernels"):
    """Checks execution status for all 20 nodes across the 4 Kaggle accounts."""
    print("=" * 80)
    print("📊 KAGGLE EXECUTION STATUS DASHBOARD")
    print("=" * 80)

    for node_idx in range(TOTAL_NOTEBOOKS):
        acc_idx = (node_idx // 5) % len(KAGGLE_ACCOUNTS)
        account = KAGGLE_ACCOUNTS[acc_idx]
        set_kaggle_env(account)

        kernel_id = f"{account['username']}/tv-ingestion-node-{node_idx}"
        res = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
        status_line = res.stdout.strip() if res.returncode == 0 else res.stderr.strip()
        print(f"Node {node_idx:02d} ({account['username']}): {status_line}")

def stop_all_nodes():
    """Cancels/deletes all 20 notebooks across the 4 Kaggle accounts."""
    for node_idx in range(TOTAL_NOTEBOOKS):
        acc_idx = (node_idx // 5) % len(KAGGLE_ACCOUNTS)
        account = KAGGLE_ACCOUNTS[acc_idx]
        set_kaggle_env(account)

        kernel_id = f"{account['username']}/tv-ingestion-node-{node_idx}"
        print(f"Deleting {kernel_id}...")
        subprocess.run(["kaggle", "kernels", "delete", "-y", kernel_id], capture_output=True, text=True)

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    mode_arg = "weekly"
    node_arg = None

    for i, a in enumerate(sys.argv):
        if a in ["initial", "weekly"]:
            mode_arg = a
        elif a in ["--node", "-n"] and i + 1 < len(sys.argv):
            node_arg = int(sys.argv[i + 1])

    if action == "dispatch":
        nodes_list = [node_arg] if node_arg is not None else None
        dispatch_all_nodes(mode=mode_arg, nodes=nodes_list)
    elif action == "status":
        check_all_status()
    elif action == "stop":
        stop_all_nodes()
    else:
        print(f"Unknown action: {action}. Usage: python -m src.src_2.orchestration.kaggle_dispatcher [dispatch|status|stop] [weekly|initial] [--node X]")
