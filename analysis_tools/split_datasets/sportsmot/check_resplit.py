#!/usr/bin/env python3

import argparse
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def load_manifest(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def list_archive_members(pklz_path: Path) -> list[str]:
    with zipfile.ZipFile(pklz_path, "r") as zf:
        return sorted(zf.namelist())


def infer_num_samples(members: list[str]) -> int:
    ids = []
    for name in members:
        if name.endswith(".pkl") and not name.endswith("_image.pkl"):
            stem = name[:-4]
            if stem.isdigit():
                ids.append(int(stem))
    return max(ids) if ids else 0


def unpickle_member(pklz_path: Path, member_name: str):
    with zipfile.ZipFile(pklz_path, "r") as zf:
        return pickle.loads(zf.read(member_name))


# ============================================================
# Old indexing rule: alphabetical folder order
# ============================================================

def build_old_index_map(old_root: Path, split: str) -> dict[str, int]:
    split_dir = old_root / split
    if not split_dir.is_dir():
        raise RuntimeError(f"Old split directory not found: {split_dir}")
    seqs = sorted([p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    return {seq: i for i, seq in enumerate(seqs, start=1)}


# ============================================================
# Renumber sample_* keys inside pickles
# ============================================================

def _renumber_samples(obj, old_id: int, new_id: int):
    old_plain = f"sample_{old_id}"
    old_pkl = f"sample_{old_id}.pkl"
    new_pkl = f"sample_{new_id}.pkl"

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str) and (k == old_plain or k == old_pkl):
                nk = new_pkl
            out[nk] = _renumber_samples(v, old_id, new_id)
        return out

    if isinstance(obj, list):
        return [_renumber_samples(x, old_id, new_id) for x in obj]

    if isinstance(obj, tuple):
        return tuple(_renumber_samples(x, old_id, new_id) for x in obj)

    return obj


# ============================================================
# Deep compare
# ============================================================

def _deep_equal(a: Any, b: Any, path: str = "") -> Tuple[bool, str]:
    if type(a) is not type(b):
        return False, f"type differs at {path or '<root>'}: {type(a)} vs {type(b)}"

    if isinstance(a, np.ndarray):
        ok = np.array_equal(a, b)
        return ok, ("match" if ok else f"np.ndarray differs at {path or '<root>'}")

    if isinstance(a, pd.DataFrame):
        ok = a.equals(b)
        return ok, ("match" if ok else f"DataFrame differs at {path or '<root>'}")

    if isinstance(a, pd.Series):
        ok = a.equals(b)
        return ok, ("match" if ok else f"Series differs at {path or '<root>'}")

    if isinstance(a, dict):
        ak, bk = set(a.keys()), set(b.keys())
        if ak != bk:
            missing = sorted(list(ak - bk), key=str)
            extra = sorted(list(bk - ak), key=str)
            return False, (
                f"dict keys differ at {path or '<root>'}: "
                f"missing_in_other={missing}, extra_in_other={extra}"
            )
        for k in sorted(ak, key=str):
            ok, reason = _deep_equal(a[k], b[k], path=f"{path}.{k}" if path else str(k))
            if not ok:
                return False, reason
        return True, "match"

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f"length differs at {path or '<root>'}: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, reason = _deep_equal(x, y, path=f"{path}[{i}]")
            if not ok:
                return False, reason
        return True, "match"

    if isinstance(a, (bytes, str, int, float, bool)) or a is None:
        ok = (a == b)
        return ok, ("match" if ok else f"value differs at {path or '<root>'}: {a!r} vs {b!r}")

    try:
        eq = (a == b)
        if isinstance(eq, np.ndarray):
            ok = bool(np.all(eq))
            return ok, ("match" if ok else f"equality-array differs at {path or '<root>'}")
        return bool(eq), ("match" if eq else f"objects differ at {path or '<root>'} using ==")
    except Exception as e:
        return False, f"could not compare at {path or '<root>'}: {e}"


# ============================================================
# Per-split runner
# ============================================================

def run_split(new_root: Path, old_root: Path, manifest: dict, split: str) -> Tuple[int, int]:
    new_pklz = new_root / "states" / f"{split}.pklz"
    if not new_pklz.exists():
        print(f"\n# Split {split}: new archive not found: {new_pklz}\n")
        return 0, 0

    if split not in manifest or not isinstance(manifest[split], dict):
        print(f"\n# Split {split}: not found in manifest or wrong format.\n")
        return 0, 0

    members = list_archive_members(new_pklz)
    member_set = set(members)
    n_samples = infer_num_samples(members)
    videos = list(manifest[split].keys())

    print("\n" + "=" * 70)
    print(f"# Split: {split}")
    print(f"# Archive: {new_pklz}")
    print(f"# Members (as list):")
    print(members)
    print(f"\n# Videos in manifest[{split}]: {len(videos)}")
    print(f"# Samples inferred from archive: {n_samples}\n")

    if n_samples != len(videos):
        print(f"# WARNING: samples({n_samples}) != manifest videos({len(videos)})\n")

    # Mapping + presence check
    n = min(n_samples, len(videos))
    for i in range(1, n + 1):
        det = f"{i}.pkl"
        img = f"{i}_image.pkl"
        ok = (det in member_set) and (img in member_set)
        status = "OK" if ok else "MISSING"
        print(f"{i:4d} -> {videos[i-1]}  [{status}]")

    print("\n# Comparing content against original archives...\n")

    total = 0
    matches = 0

    # cache old index maps per origin split
    old_index_maps = {}

    for idx in range(1, len(videos) + 1):
        video_name = videos[idx - 1]
        origin_split = manifest[split].get(video_name)

        if origin_split not in ("train", "val"):
            print(f"{idx:4d} -> {video_name} [ERROR origin_split={origin_split}]")
            total += 1
            continue

        if origin_split not in old_index_maps:
            old_index_maps[origin_split] = build_old_index_map(old_root, origin_split)

        old_idx = old_index_maps[origin_split].get(video_name)
        if old_idx is None:
            print(f"{idx:4d} -> {video_name} [ERROR not found in old {origin_split}]")
            total += 1
            continue

        new_member = f"{idx}.pkl"
        old_member = f"{old_idx}.pkl"
        old_pklz = old_root / "states" / f"{origin_split}.pklz"

        try:
            new_obj = unpickle_member(new_pklz, new_member)
        except Exception as e:
            print(f"{idx:4d} -> {video_name} [ERROR unpickle new {new_member}: {e}]")
            total += 1
            continue

        try:
            old_obj = unpickle_member(old_pklz, old_member)
        except Exception as e:
            print(f"{idx:4d} -> {video_name} [ERROR unpickle old {origin_split}:{old_member}: {e}]")
            total += 1
            continue

        old_obj_norm = _renumber_samples(old_obj, old_id=old_idx, new_id=idx)
        ok, reason = _deep_equal(old_obj_norm, new_obj)

        total += 1
        if ok:
            matches += 1
            print(f"{idx:4d} -> {video_name} [MATCH]")
        else:
            print(f"{idx:4d} -> {video_name} [DIFF] {reason}")

    print(f"\n# Split summary: {matches}/{total} match exactly\n")
    return matches, total


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_root", type=Path, required=True,
                    help="New resplit dataset root containing split_manifest.json and states/")
    ap.add_argument("--old_root", type=Path, required=True,
                    help="Original SportsMOT root containing train/, val/, and states/train.pklz + states/val.pklz")
    args = ap.parse_args()

    manifest = load_manifest(args.new_root / "manifest.json")

    global_matches = 0
    global_total = 0

    for split in ["train", "val", "test"]:
        m, t = run_split(args.new_root, args.old_root, manifest, split)
        global_matches += m
        global_total += t

    print("\n" + "=" * 70)
    print(f"# GLOBAL SUMMARY: {global_matches}/{global_total} samples match exactly")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()