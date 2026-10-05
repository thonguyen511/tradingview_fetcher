"""Robust CSV symbol reader and workload partitioner."""
import os
import csv
import ast
import glob
import json
from typing import List, Dict, Any, Optional

csv.field_size_limit(10**9)

def extract_symbols_from_csv(csv_path: str) -> List[str]:
    """
    Extracts all TradingView symbol strings in 'EXCHANGE:SYMBOL' format from a CSV file.
    Correctly unpacks 'contracts' JSON/Python arrays and applies prefixes.
    """
    symbols = []
    if not os.path.exists(csv_path):
        return []

    try:
        with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []

            try:
                ex_idx = header.index("exchange_code")
                sym_idx = header.index("symbol")
            except ValueError:
                return []

            prefix_idx = header.index("prefix") if "prefix" in header else -1
            contracts_idx = header.index("contracts") if "contracts" in header else -1

            for row in reader:
                if len(row) <= max(ex_idx, sym_idx):
                    continue

                base_sym = row[sym_idx].strip()
                if not base_sym:
                    continue

                row_ex = row[ex_idx].strip()
                row_prefix = row[prefix_idx].strip() if (prefix_idx != -1 and len(row) > prefix_idx) else ""
                default_prefix = row_prefix if (row_prefix and row_prefix.lower() != "none") else row_ex

                extracted = []
                # Check for contracts array (futures, derivatives, continuous)
                if contracts_idx != -1 and len(row) > contracts_idx:
                    c_str = row[contracts_idx].strip()
                    if c_str.startswith("[") and c_str.endswith("]"):
                        try:
                            c_list = ast.literal_eval(c_str)
                            if isinstance(c_list, list):
                                for c in c_list:
                                    if isinstance(c, dict) and c.get("symbol"):
                                        c_sym = str(c["symbol"]).strip()
                                        c_pref = str(c.get("prefix") or "").strip()
                                        pref = c_pref if (c_pref and c_pref.lower() != "none") else default_prefix
                                        extracted.append(f"{pref}:{c_sym}")
                        except Exception:
                            pass

                if not extracted:
                    extracted.append(f"{default_prefix}:{base_sym}")

                for s in extracted:
                    symbols.append(s)

    except Exception as e:
        print(f"⚠️ Error reading symbols from {csv_path}: {e}")

    # Deduplicate while strictly preserving chronological/file row order
    return list(dict.fromkeys(symbols))

def slice_into_chunks(symbols: List[str], chunk_size: int = 5000) -> List[Dict[str, Any]]:
    """
    Slices symbols list into fixed chunks of chunk_size (default 5,000).
    Enables scalable appending when new rows are added to the end of a CSV.
    """
    chunks = []
    total = len(symbols)
    if total == 0:
        return []

    for start_idx in range(0, total, chunk_size):
        end_idx = min(start_idx + chunk_size, total)
        chunk_symbols = symbols[start_idx:end_idx]
        chunk_id = (start_idx // chunk_size) + 1
        chunks.append({
            "chunk_id": chunk_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "count": len(chunk_symbols),
            "is_tail": end_idx == total,
            "symbols": chunk_symbols
        })
    return chunks

def discover_csv_files(search_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Discovers all symbol CSV files across local and Kaggle paths.
    Returns mapping of filename -> absolute path.
    """
    default_patterns = [
        "data/AIO_CSV/*.csv",
        "../data/AIO_CSV/*.csv",
        "../../data/AIO_CSV/*.csv",
        "/kaggle/input/**/AIO_CSV/*.csv",
        "/kaggle/input/**/*.csv",
        "/kaggle/working/**/AIO_CSV/*.csv",
    ]
    patterns = search_dirs or default_patterns
    found = {}
    for p in patterns:
        for f in glob.glob(p, recursive=True):
            fname = os.path.basename(f)
            if fname.endswith(".csv") and not fname.startswith("."):
                if fname not in found:
                    found[fname] = os.path.abspath(f)
    return found

def get_node_workload(
    node_index: int,
    total_nodes: int = 20,
    chunk_size: int = 5000,
    map_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Computes workload chunks for a specific node using map.json.
    Each chunk is a unit of work (<= 5,000 symbols) that produces a Parquet shard.
    """
    csv_map = discover_csv_files()
    if not csv_map:
        print("❌ No CSV files discovered in workspace or Kaggle input directories.")
        return []

    # Locate map.json
    candidate_maps = [map_path] if map_path else [
        "data/AIO_CSV/map.json",
        "../data/AIO_CSV/map.json",
        "/kaggle/working/data/AIO_CSV/map.json",
        "/kaggle/input/**/map.json"
    ]
    mapping_data = None
    for cand in candidate_maps:
        if not cand: continue
        matches = glob.glob(cand, recursive=True) if "*" in cand else ([cand] if os.path.exists(cand) else [])
        if matches:
            try:
                with open(matches[0], "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, list) and len(d) == total_nodes:
                        mapping_data = d
                        print(f"📖 Loaded greedy allocation map from: {matches[0]}")
                        break
            except Exception:
                pass

    assigned_files = []
    if mapping_data and 0 <= node_index < len(mapping_data):
        node_spec = mapping_data[node_index]
        for f_name in node_spec.get("files", []):
            base = os.path.basename(f_name)
            if base in csv_map:
                assigned_files.append((base, csv_map[base]))
    else:
        # Fallback to deterministic modulo allocation if map.json is not found
        sorted_files = sorted(csv_map.keys())
        for idx, base in enumerate(sorted_files):
            if idx % total_nodes == node_index:
                assigned_files.append((base, csv_map[base]))

    workload_chunks = []
    total_syms = 0

    for fname, fpath in assigned_files:
        symbols = extract_symbols_from_csv(fpath)
        total_syms += len(symbols)
        csv_name = os.path.splitext(fname)[0]
        chunks = slice_into_chunks(symbols, chunk_size=chunk_size)

        for c in chunks:
            chunk_tag = f"{csv_name}_chunk_{c['chunk_id']}"
            workload_chunks.append({
                "csv_name": csv_name,
                "csv_path": fpath,
                "chunk_id": c["chunk_id"],
                "chunk_tag": chunk_tag,
                "symbols": c["symbols"],
                "count": c["count"],
                "is_tail": c["is_tail"]
            })

    print(f"📋 Node {node_index}: Assigned {len(assigned_files)} files ({total_syms:,} symbols) -> {len(workload_chunks)} chunks")
    return workload_chunks
