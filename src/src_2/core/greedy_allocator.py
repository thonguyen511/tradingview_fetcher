import os
import sys

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
from typing import Dict, List, Any
from .symbol_reader import extract_symbols_from_csv

def build_greedy_map(
    csv_dir: str = "data/AIO_CSV",
    map_file: str = "data/AIO_CSV/map.json",
    total_nodes: int = 20
) -> List[Dict[str, Any]]:
    """
    Builds a fresh greedy split map across total_nodes.
    Sorts CSV files descending by symbol count and assigns greedily to the lightest node.
    """
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and not f.startswith(".")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    print(f"⚖️ Scanning {len(csv_files)} CSV files for greedy allocation across {total_nodes} nodes...")
    file_stats = []
    for f in csv_files:
        full_p = os.path.join(csv_dir, f)
        count = len(extract_symbols_from_csv(full_p))
        file_stats.append({"filename": f, "symbols": count})

    # Sort descending by count, then by filename for deterministic ordering
    file_stats.sort(key=lambda x: (-x["symbols"], x["filename"]))

    nodes = [{"node_index": i, "total_symbols": 0, "files": []} for i in range(total_nodes)]

    for f_stat in file_stats:
        lightest = min(nodes, key=lambda n: n["total_symbols"])
        lightest["files"].append(f_stat["filename"])
        lightest["total_symbols"] += f_stat["symbols"]

    os.makedirs(os.path.dirname(os.path.abspath(map_file)), exist_ok=True)
    with open(map_file, "w", encoding="utf-8") as out:
        json.dump(nodes, out, indent=4)

    print(f"✅ Saved greedy split map to: {map_file}")
    for n in nodes:
        print(f"   Node {n['node_index']:02d}: {n['total_symbols']:,} symbols across {len(n['files'])} files")

    return nodes

def update_map_incremental(
    csv_dir: str = "data/AIO_CSV",
    map_file: str = "data/AIO_CSV/map.json",
    total_nodes: int = 20
) -> List[Dict[str, Any]]:
    """
    Incrementally updates map.json without disrupting existing assignments:
    1. Preserves existing file-to-node assignments.
    2. Recalculates current symbol counts for all files (captures newly appended rows).
    3. Detects brand-new CSV files and assigns them greedily to the currently lightest nodes.
    """
    if not os.path.exists(map_file):
        return build_greedy_map(csv_dir, map_file, total_nodes)

    with open(map_file, "r", encoding="utf-8") as f:
        try:
            nodes = json.load(f)
            if not isinstance(nodes, list) or len(nodes) != total_nodes:
                return build_greedy_map(csv_dir, map_file, total_nodes)
        except Exception:
            return build_greedy_map(csv_dir, map_file, total_nodes)

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv") and not f.startswith(".")]
    existing_assigned = set()
    for n in nodes:
        existing_assigned.update(n.get("files", []))

    new_files = [f for f in csv_files if f not in existing_assigned]

    # Recalculate symbol counts for currently assigned files to account for row appends
    print(f"🔄 Recalculating symbol counts for {len(existing_assigned)} mapped files...")
    for n in nodes:
        n["total_symbols"] = 0
        for f_name in n.get("files", []):
            f_path = os.path.join(csv_dir, f_name)
            if os.path.exists(f_path):
                cnt = len(extract_symbols_from_csv(f_path))
                n["total_symbols"] += cnt

    if new_files:
        print(f"📦 Found {len(new_files)} new CSV files to incrementally allocate...")
        new_stats = []
        for nf in new_files:
            full_p = os.path.join(csv_dir, nf)
            cnt = len(extract_symbols_from_csv(full_p))
            new_stats.append({"filename": nf, "symbols": cnt})

        new_stats.sort(key=lambda x: (-x["symbols"], x["filename"]))

        for f_stat in new_stats:
            lightest = min(nodes, key=lambda n: n["total_symbols"])
            lightest["files"].append(f_stat["filename"])
            lightest["total_symbols"] += f_stat["symbols"]
            print(f"   + Assigned {f_stat['filename']} ({f_stat['symbols']:,} symbols) -> Node {lightest['node_index']}")

    with open(map_file, "w", encoding="utf-8") as out:
        json.dump(nodes, out, indent=4)

    print(f"✅ Incremental update complete. Map written to {map_file}")
    return nodes
