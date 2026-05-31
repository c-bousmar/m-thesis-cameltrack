#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd


# ============================================================
# Utilities
# ============================================================

def rel_symlink(target: Path, link_path: Path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel)


# ============================================================
# Manifest reading
# ============================================================

def read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split in ["train", "val", "test"]:
        if split not in manifest:
            manifest[split] = []

    return manifest


def normalize_manifest_entries(entries: List[Any], split_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for item in entries:
        if not isinstance(item, dict):
            raise TypeError(
                f"All entries in manifest['{split_name}'] must be dicts, got {type(item)}"
            )

        required = ["new_index", "sequence_name", "orig_split", "orig_index"]
        missing = [k for k in required if k not in item]
        if missing:
            raise KeyError(
                f"Manifest entry in split '{split_name}' missing required keys {missing}: {item}"
            )

        entry = dict(item)
        entry["new_index"] = int(entry["new_index"])
        entry["orig_index"] = int(entry["orig_index"])

        if entry["orig_split"] not in {"train", "val"}:
            raise ValueError(
                f"orig_split must be 'train' or 'val', got {entry['orig_split']!r} "
                f"for sequence {entry['sequence_name']!r}"
            )

        out.append(entry)

    return out


def validate_manifest(manifest: dict):
    seen_sequences = set()

    for split in ["train", "val", "test"]:
        entries = normalize_manifest_entries(manifest.get(split, []), split)

        seen_new_indices = set()
        for entry in entries:
            seq = entry["sequence_name"]
            new_index = entry["new_index"]

            if seq in seen_sequences:
                raise RuntimeError(f"Duplicate sequence across manifest splits: {seq}")
            seen_sequences.add(seq)

            if new_index in seen_new_indices:
                raise RuntimeError(
                    f"Duplicate new_index={new_index} inside manifest split '{split}'"
                )
            seen_new_indices.add(new_index)

    if "counts" in manifest and isinstance(manifest["counts"], dict):
        for split in ["train", "val", "test"]:
            expected = manifest["counts"].get(split)
            if expected is not None:
                actual = len(manifest.get(split, []))
                if int(expected) != actual:
                    raise RuntimeError(
                        f"Manifest counts mismatch for '{split}': expected {expected}, got {actual}"
                    )


# ============================================================
# Load PKLZ archive
# ============================================================

def load_pklz(path: Path) -> Tuple[zipfile.ZipFile, dict]:
    z = zipfile.ZipFile(path, "r")
    summary = json.loads(z.read("summary.json").decode("utf-8"))
    return z, summary


# ============================================================
# Read/write pickled members
# ============================================================

def _read_pickle_from_zip(z: zipfile.ZipFile, member: str):
    return pickle.loads(z.read(member))


def _write_pickle_to_zip(zout: zipfile.ZipFile, member: str, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    zout.writestr(member, data)


def _ensure_pandas_df(obj, what: str) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected {what} to be a pandas.DataFrame, got {type(obj)}")
    return obj


# ============================================================
# Path rewriting
# ============================================================

def _rewrite_file_path(old_path: str, seq: str, dst_root: Path, new_split: str) -> str:
    needle = f"/{seq}/"
    idx = old_path.find(needle)
    if idx == -1:
        return old_path
    suffix = old_path[idx + 1:]  # "seq/..."
    return str((dst_root / new_split / Path(suffix)).resolve())


# ============================================================
# X_image.pkl rewrite
# ============================================================

def _rewrite_X_image_pkl_content(
    img_df: pd.DataFrame,
    x_image_frame_offset: int,
    *,
    seq: str,
    dst_root: Path,
    split_name: str,
    new_video_id: int,
) -> tuple[pd.DataFrame, int]:
    required = ["frame", "file_path", "is_labeled"]
    for col in required:
        if col not in img_df.columns:
            raise KeyError(f"X_image.pkl: missing required column '{col}'")

    frame = pd.to_numeric(img_df["frame"], errors="raise").astype("int64")

    # KEEP PREVIOUS BEHAVIOR:
    # rewritten image id is frame + split-wise offset
    new_id = (frame + x_image_frame_offset).astype("int64")

    seq_span = int(frame.max()) + 1
    x_image_frame_offset += seq_span

    new_file_path = img_df["file_path"].astype(str).map(
        lambda p: _rewrite_file_path(p, seq=seq, dst_root=dst_root, new_split=split_name)
    )

    # Preserve previous duplicated-id behavior if the input has duplicated "id" columns
    id_positions = [i for i, c in enumerate(img_df.columns) if c == "id"]

    if len(id_positions) >= 2:
        first_id = pd.to_numeric(
            img_df.iloc[:, id_positions[0]], errors="raise"
        ).astype("int64")

        out = pd.DataFrame(
            {
                "id": first_id.values,
                "id": new_id.values,
                "frame": frame.values,
                "file_path": new_file_path.values,
                "video_id": new_video_id,
                "is_labeled": img_df["is_labeled"].values,
            }
        )
        # The dict form above collapses duplicate keys, so rebuild explicitly:
        out = pd.DataFrame(
            list(
                zip(
                    first_id.values,
                    new_id.values,
                    frame.values,
                    new_file_path.values,
                    [new_video_id] * len(img_df),
                    img_df["is_labeled"].values,
                )
            ),
            columns=["id", "id", "frame", "file_path", "video_id", "is_labeled"],
        )
    else:
        out = pd.DataFrame(
            {
                "id": new_id.values,
                "frame": frame.values,
                "file_path": new_file_path.values,
                "video_id": new_video_id,
                "is_labeled": img_df["is_labeled"].values,
            }
        )

    # KEEP PREVIOUS BEHAVIOR:
    out.index = pd.Index(new_id.values, name="id")

    return out, x_image_frame_offset


# ============================================================
# X.pkl rewrite (detections)
# ============================================================

def _rewrite_X_pkl_content(
    det_df: pd.DataFrame,
    *,
    new_video_id: int,
    global_image_id_offset: int,
    global_det_id_offset: int,
    det_src_name: str,
) -> tuple[pd.DataFrame, int, int]:

    for col in ["video_id", "image_id"]:
        if col not in det_df.columns:
            raise KeyError(f"{det_src_name}: missing required column '{col}'")

    det_df = det_df.copy()
    det_df["video_id"] = new_video_id

    det_img = pd.to_numeric(det_df["image_id"], errors="raise").astype("int64")
    det_img0 = int(det_img.iloc[0])
    det_df["image_id"] = (det_img - det_img0 + global_image_id_offset).astype("int64")

    det_df.index = pd.RangeIndex(
        start=global_det_id_offset,
        stop=global_det_id_offset + len(det_df),
        step=1,
    )

    global_det_id_offset += len(det_df)

    return det_df, global_image_id_offset, global_det_id_offset


# ============================================================
# Rebuild PKLZ
# ============================================================

def rebuild_pklz_with_reindex(
    output_path: Path,
    split_entries: List[Dict[str, Any]],
    train_zip: zipfile.ZipFile,
    val_zip: zipfile.ZipFile,
    train_summary: dict,
    val_summary: dict,
    *,
    dst_root: Path,
    split_name: str,
):
    print(f"Creating {output_path} ({len(split_entries)} sequences)")

    source_summary = train_summary if train_summary is not None else val_summary
    frozen_summary = deepcopy(source_summary)

    global_image_id_offset = 0
    global_det_id_offset = 0
    x_image_frame_offset = 0

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("summary.json", json.dumps(frozen_summary, indent=2))

        for entry in split_entries:
            new_video_id = entry["new_index"]
            seq = entry["sequence_name"]
            old_id = entry["orig_index"]
            orig_split = entry["orig_split"]

            src_zip = train_zip if orig_split == "train" else val_zip

            det_src = f"{old_id}.pkl"
            img_src = f"{old_id}_image.pkl"

            det_df = _ensure_pandas_df(_read_pickle_from_zip(src_zip, det_src), "detection pickle")
            img_df = _ensure_pandas_df(_read_pickle_from_zip(src_zip, img_src), "image pickle")

            img_df, x_image_frame_offset = _rewrite_X_image_pkl_content(
                img_df,
                x_image_frame_offset,
                seq=seq,
                dst_root=dst_root,
                split_name=split_name,
                new_video_id=new_video_id,
            )

            det_df, global_image_id_offset, global_det_id_offset = _rewrite_X_pkl_content(
                det_df,
                new_video_id=new_video_id,
                global_image_id_offset=global_image_id_offset,
                global_det_id_offset=global_det_id_offset,
                det_src_name=det_src,
            )

            det_dst = f"{new_video_id}.pkl"
            img_dst = f"{new_video_id}_image.pkl"
            _write_pickle_to_zip(zout, det_dst, det_df)
            _write_pickle_to_zip(zout, img_dst, img_df)

            frame_series = pd.to_numeric(img_df["frame"], errors="raise").astype("int64")
            n_images_span = int(frame_series.max()) + 1
            global_image_id_offset += n_images_span


# ============================================================
# Create symlink dataset
# ============================================================

def create_symlink_split(src_root: Path, dst_root: Path, split_name: str, split_entries: List[Dict[str, Any]]):
    dst_split = dst_root / split_name
    dst_split.mkdir(parents=True, exist_ok=True)

    for entry in split_entries:
        seq = entry["sequence_name"]
        orig_split = entry["orig_split"]

        src_path = src_root / orig_split / seq
        if not src_path.exists():
            raise RuntimeError(f"Sequence not found in src {orig_split}: {src_path}")

        rel_symlink(src_path, dst_split / seq)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="Dataset root (with train/ val/ states/); manifest must exist at <src>/manifest.json")
    parser.add_argument("--dst", required=True, type=Path, help="New dataset root to create")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    if src == dst:
        raise RuntimeError("--dst must be different from --src")
    if dst in [src, src / "train", src / "val", src / "test", src / "states"]:
        raise RuntimeError("--dst must not overlap source dataset directories")
    if str(dst).startswith(str(src) + os.sep):
        raise RuntimeError("--dst must not be inside --src")

    manifest_path = src / "manifest.json"
    print(f"Reading manifest from: {manifest_path}")
    manifest = read_manifest(manifest_path)
    validate_manifest(manifest)

    train_entries = normalize_manifest_entries(manifest.get("train", []), "train")
    val_entries = normalize_manifest_entries(manifest.get("val", []), "val")
    test_entries = normalize_manifest_entries(manifest.get("test", []), "test")

    print(
        "Split sizes:",
        {
            "train": len(train_entries),
            "val": len(val_entries),
            "test": len(test_entries),
        },
    )

    print("\nCreating symlink dataset...")
    create_symlink_split(src, dst, "train", train_entries)
    create_symlink_split(src, dst, "val", val_entries)
    create_symlink_split(src, dst, "test", test_entries)

    print("\nLoading original PKLZ archives...")
    train_zip, train_summary = load_pklz(src / "states/train.pklz")
    val_zip, val_summary = load_pklz(src / "states/val.pklz")

    states_out = dst / "states"
    states_out.mkdir(parents=True, exist_ok=True)

    rebuild_pklz_with_reindex(
        states_out / "train.pklz",
        train_entries,
        train_zip,
        val_zip,
        train_summary,
        val_summary,
        dst_root=dst,
        split_name="train",
    )
    rebuild_pklz_with_reindex(
        states_out / "val.pklz",
        val_entries,
        train_zip,
        val_zip,
        train_summary,
        val_summary,
        dst_root=dst,
        split_name="val",
    )
    rebuild_pklz_with_reindex(
        states_out / "test.pklz",
        test_entries,
        train_zip,
        val_zip,
        train_summary,
        val_summary,
        dst_root=dst,
        split_name="test",
    )

    train_zip.close()
    val_zip.close()

    print("\nDone.")
    print("Dataset ready at:", dst)


if __name__ == "__main__":
    main()