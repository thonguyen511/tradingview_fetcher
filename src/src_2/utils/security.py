"""Decryption helpers for Fernet-encrypted parquet files."""
import os
import io
import glob
import base64
import pandas as pd
from typing import Optional

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    Fernet = None

def get_cipher_from_password(password: str):
    if not Fernet or not password:
        return None
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'tradingview',
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return Fernet(key)

def decrypt_dataset_directory(folder_path: str, password: str) -> int:
    """Recursively decrypts all .parquet.enc files in folder_path into standard .parquet."""
    cipher = get_cipher_from_password(password)
    if not cipher:
        raise ValueError("Cryptography library missing or empty password.")

    pattern = os.path.join(folder_path, "**", "*.parquet.enc")
    enc_files = glob.glob(pattern, recursive=True)
    if not enc_files:
        print(f"No encrypted files found in {folder_path}")
        return 0

    decrypted_count = 0
    for enc_file in enc_files:
        out_file = enc_file[:-4]
        try:
            with open(enc_file, "rb") as f:
                data = f.read()
            dec_bytes = cipher.decrypt(data)
            df = pd.read_parquet(io.BytesIO(dec_bytes))
            df.to_parquet(out_file, index=False)
            os.remove(enc_file)
            decrypted_count += 1
        except Exception as e:
            print(f"Failed to decrypt {enc_file}: {e}")

    print(f"Decrypted {decrypted_count} files successfully.")
    return decrypted_count
