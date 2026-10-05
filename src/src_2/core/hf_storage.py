"""Hugging Face Storage Engine with chunked overwrite and Git history squashing.

Prevents exceeding Hugging Face 8TB limit by:
1. Downloading only target 500MB chunk shards on limited-disk runners (< 19GB disk).
2. Merging & deduplicating with keep='last' to finalize incomplete bars.
3. Overwriting the Parquet shard in a single commit.
4. Squashing Git history (super_squash_history) to prune old blobs.
"""
import os
import io
import time
import shutil
import pandas as pd
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download
from ..config import HF_REPO_ID, HF_TOKEN, ENABLE_ENCRYPTION, DATA_PASSWORD

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    def get_cipher(pwd: str):
        if not pwd: return None
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'tradingview',
            iterations=480000,
        )
        import base64
        return Fernet(base64.urlsafe_b64encode(kdf.derive(pwd.encode())))
except ImportError:
    def get_cipher(pwd: str): return None

class HuggingFaceStorageEngine:
    """Manages Parquet synchronization, single-shard merges, log persistence, and history squashing."""

    def __init__(self, repo_id: str = HF_REPO_ID, token: Optional[str] = None):
        self.repo_id = repo_id
        self.token = token or HF_TOKEN or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token) if self.token else HfApi()
        self.cipher = get_cipher(DATA_PASSWORD) if ENABLE_ENCRYPTION else None

    def pull_chunk(self, chunk_rel_path: str) -> pd.DataFrame:
        """
        Pulls a single Parquet shard from Hugging Face if present.
        Returns an empty DataFrame if the chunk does not exist yet.
        """
        remote_filename = chunk_rel_path + (".enc" if (ENABLE_ENCRYPTION and self.cipher) else "")
        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_filename,
                repo_type="dataset",
                token=self.token
            )
            if downloaded and os.path.exists(downloaded):
                with open(downloaded, "rb") as f:
                    raw_bytes = f.read()

                if ENABLE_ENCRYPTION and self.cipher:
                    raw_bytes = self.cipher.decrypt(raw_bytes)

                df = pd.read_parquet(io.BytesIO(raw_bytes))
                return df
        except Exception:
            pass

        return pd.DataFrame()

    def merge_datasets(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges existing historical chunk with new data:
        1. Appends new candlesticks.
        2. keep='last' overwrites incomplete/unclosed candles with finalized figures.
        3. Sorts strictly by [symbol, time].
        """
        if existing_df is None or existing_df.empty:
            return new_df.copy() if (new_df is not None and not new_df.empty) else pd.DataFrame()

        if new_df is None or new_df.empty:
            return existing_df.copy()

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Deduplicate: replace unclosed prior week candle with finalized candle
        combined.drop_duplicates(subset=["symbol", "time"], keep="last", inplace=True)
        combined.sort_values(by=["symbol", "time"], ascending=[True, True], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    def save_local_chunk(
        self,
        df: pd.DataFrame,
        chunk_rel_path: str,
        local_dir: str = "output"
    ) -> str:
        """Saves DataFrame as Parquet (optionally Fernet encrypted) to local output folder."""
        if df.empty:
            return ""

        target_file = os.path.join(local_dir, chunk_rel_path)
        os.makedirs(os.path.dirname(target_file), exist_ok=True)

        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        parquet_bytes = buf.getvalue()

        if ENABLE_ENCRYPTION and self.cipher:
            encrypted_bytes = self.cipher.encrypt(parquet_bytes)
            target_file += ".enc"
            with open(target_file, "wb") as f:
                f.write(encrypted_bytes)
        else:
            with open(target_file, "wb") as f:
                f.write(parquet_bytes)

        return target_file

    def push_data_folder(
        self,
        local_data_dir: str,
        commit_message: str = "Sync chunk data",
        max_retries: int = 5
    ) -> bool:
        """Uploads entire data folder to Hugging Face in a single atomic commit."""
        if not os.path.exists(local_data_dir):
            return False

        for attempt in range(1, max_retries + 1):
            try:
                self.api.upload_folder(
                    folder_path=local_data_dir,
                    path_in_repo="data",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message
                )
                print(f"🚀 Synced Parquet data to HF: {commit_message}")
                return True
            except Exception as e:
                print(f"⚠️ Push data attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(3.0 * attempt)
        return False

    def squash_history(self):
        """
        Runs super_squash_history on Hugging Face Hub to permanently clear overwritten Git blobs.
        Guarantees that the repository never balloons beyond the 8TB limit.
        """
        try:
            print(f"🧹 Squashing Git history on HF Hub ({self.repo_id})...")
            self.api.super_squash_history(repo_id=self.repo_id, repo_type="dataset")
            print("✅ Git history squashed. Pruned hidden blobs.")
        except Exception as e:
            print(f"⚠️ Could not squash history: {e}")

    def download_checkpoint_logs(self, local_log_dir: str, notebook_id: str):
        """Downloads existing checkpoint.log and failures.log to resume interrupted multi-day runs."""
        os.makedirs(local_log_dir, exist_ok=True)
        for log_name in ["checkpoint.log", "failures.log"]:
            try:
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"log/{notebook_id}/{log_name}",
                    repo_type="dataset",
                    token=self.token
                )
                if downloaded and os.path.exists(downloaded):
                    shutil.copy(downloaded, os.path.join(local_log_dir, log_name))
            except Exception:
                pass

    def push_checkpoint_logs(
        self,
        local_log_dir: str,
        notebook_id: str,
        commit_message: str = "Update checkpoint logs"
    ) -> bool:
        """Pushes checkpoint.log and failures.log to Hugging Face under log/{notebook_id}."""
        if not os.path.exists(local_log_dir):
            return False

        for attempt in range(1, 4):
            try:
                self.api.upload_folder(
                    folder_path=local_log_dir,
                    path_in_repo=f"log/{notebook_id}",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=commit_message
                )
                return True
            except Exception:
                time.sleep(2.0)
        return False
