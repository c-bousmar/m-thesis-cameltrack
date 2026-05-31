#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Helpers
# ============================================================

def read_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
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

        required = ["new_index", "sequence_name"]
        missing = [k for k in required if k not in item]
        if missing:
            raise KeyError(
                f"Manifest entry in split '{split_name}' missing required keys {missing}: {item}"
            )

        entry = dict(item)
        entry["new_index"] = int(entry["new_index"])
        out.append(entry)
    return out


def load_pklz(path: Path) -> zipfile.ZipFile:
    if not path.exists():
        raise FileNotFoundError(f"PKLZ not found: {path}")
    return zipfile.ZipFile(path, "r")


def read_pickle_from_zip(z: zipfile.ZipFile, member: str):
    if member not in z.namelist():
        raise KeyError(f"Missing member in archive: {member}")
    return pickle.loads(z.read(member))


def ensure_df(obj, what: str) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected {what} to be a pandas.DataFrame, got {type(obj)}")
    return obj


def infer_sequence_names_from_file_paths(file_paths: pd.Series) -> List[str]:
    """
    Extract the sequence folder name from paths like:
      /.../<split>/<sequence_name>/img1/000001.jpg
    We take the parent of 'img1' when possible.
    """
    names = set()

    for p in file_paths.astype(str):
        path = Path(p)
        parts = path.parts

        if "img1" in parts:
            img1_idx = parts.index("img1")
            if img1_idx >= 1:
                names.add(parts[img1_idx - 1])
                continue

        # fallback: parent.parent is often the sequence folder
        try:
            names.add(path.parent.parent.name)
        except Exception:
            pass

    return sorted(names)


def pick_image_id_series(img_df: pd.DataFrame) -> pd.Series:
    """
    In X_image.pkl there may be duplicated 'id' columns.
    We want the rewritten image id, which in your previous convention is:
      - the dataframe index, if named id-like and meaningful
      - otherwise the LAST 'id' column
      - otherwise fail
    """
    # safest: use dataframe index if it is integer-like and non-empty
    if len(img_df.index) > 0:
        try:
            idx_series = pd.Series(img_df.index, index=img_df.index)
            pd.to_numeric(idx_series, errors="raise")
            return idx_series.astype("int64")
        except Exception:
            pass

    id_positions = [i for i, c in enumerate(img_df.columns) if c == "id"]
    if id_positions:
        # use the last duplicated 'id' column = rewritten id convention
        col = img_df.iloc[:, id_positions[-1]]
        return pd.to_numeric(col, errors="raise").astype("int64")

    raise KeyError("Could not determine rewritten image id in X_image.pkl")


def maybe_get_video_id(df: pd.DataFrame) -> Optional[int]:
    if "video_id" not in df.columns or len(df) == 0:
        return None
    vals = pd.to_numeric(df["video_id"], errors="raise").astype("int64")
    uniq = vals.drop_duplicates().tolist()
    if len(uniq) == 1:
        return int(uniq[0])
    return None


def maybe_get_first_last(series: pd.Series) -> tuple[Optional[int], Optional[int]]:
    if len(series) == 0:
        return None, None
    vals = pd.to_numeric(series, errors="raise").astype("int64")
    return int(vals.iloc[0]), int(vals.iloc[-1])


def maybe_get_min_max(series: pd.Series) -> tuple[Optional[int], Optional[int]]:
    if len(series) == 0:
        return None, None
    vals = pd.to_numeric(series, errors="raise").astype("int64")
    return int(vals.min()), int(vals.max())


def count_distinct_track_ids(det_df: pd.DataFrame) -> Optional[int]:
    for candidate in ["track_id", "id"]:
        if candidate in det_df.columns:
            s = pd.to_numeric(det_df[candidate], errors="coerce")
            return int(s.dropna().nunique())
    return None


def get_split_sequence_names(dataset_root: Path, split_name: str) -> List[str]:
    split_dir = dataset_root / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    seq_names = [p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    seq_names.sort()  # mimic CAMELTrack / basic Python behavior exactly
    return seq_names


# ============================================================
# Verification core
# ============================================================
def verify_split(dataset_root: Path, split_name: str) -> List[Dict[str, Any]]:
    seq_names = get_split_sequence_names(dataset_root, split_name)
    pklz_path = dataset_root / "states" / f"{split_name}.pklz"
    z = load_pklz(pklz_path)

    rows: List[Dict[str, Any]] = []

    try:
        for idx, seq_name in enumerate(seq_names, start=1):
            img_member = f"{idx}_image.pkl"
            det_member = f"{idx}.pkl"

            row: Dict[str, Any] = {
                "split": split_name,
                "sequence_name_dir": seq_name,
                "index": idx,
                "img_member": img_member,
                "det_member": det_member,
                "sequence_name_from_file_path": None,
                "file_path_match": None,
                "x_image_video_id": None,
                "x_image_first_frame": None,
                "x_image_last_frame": None,
                "x_image_first_id": None,
                "x_image_last_id": None,
                "x_det_video_id": None,
                "x_det_num_distinct_track_id": None,
                "x_det_first_image_id": None,
                "x_det_last_image_id": None,
                "status": "OK",
                "notes": "",
            }

            notes: List[str] = []

            # ---------- X_image.pkl ----------
            try:
                img_df = ensure_df(read_pickle_from_zip(z, img_member), img_member)

                if "file_path" not in img_df.columns:
                    notes.append("X_image missing column 'file_path'")
                if "frame" not in img_df.columns:
                    notes.append("X_image missing column 'frame'")

                inferred_names = infer_sequence_names_from_file_paths(
                    img_df["file_path"] if "file_path" in img_df.columns else pd.Series(dtype=object)
                )

                if len(inferred_names) == 1:
                    inferred_seq = inferred_names[0]
                elif len(inferred_names) == 0:
                    inferred_seq = None
                    notes.append("Could not infer sequence name from any file_path")
                else:
                    inferred_seq = " | ".join(inferred_names)
                    notes.append(f"Multiple sequence names inferred from file_path: {inferred_names}")

                row["sequence_name_from_file_path"] = inferred_seq
                row["file_path_match"] = (inferred_seq == seq_name)

                if row["file_path_match"] is False:
                    notes.append(
                        f"Directory sequence_name '{seq_name}' != X_image file_path sequence '{inferred_seq}'"
                    )

                if "video_id" in img_df.columns:
                    row["x_image_video_id"] = maybe_get_video_id(img_df)

                if "frame" in img_df.columns:
                    first_frame, last_frame = maybe_get_first_last(img_df["frame"])
                    row["x_image_first_frame"] = first_frame
                    row["x_image_last_frame"] = last_frame

                try:
                    img_id = pick_image_id_series(img_df)
                    first_id, last_id = maybe_get_first_last(img_id)
                    row["x_image_first_id"] = first_id
                    row["x_image_last_id"] = last_id
                except Exception as e:
                    notes.append(f"Could not determine rewritten X_image id: {e}")

            except Exception as e:
                row["status"] = "ERROR"
                notes.append(f"Failed reading {img_member}: {e}")

            # ---------- X.pkl ----------
            try:
                det_df = ensure_df(read_pickle_from_zip(z, det_member), det_member)

                if "video_id" in det_df.columns:
                    row["x_det_video_id"] = maybe_get_video_id(det_df)
                else:
                    notes.append("X.pkl missing column 'video_id'")

                if "image_id" in det_df.columns:
                    first_img_id, last_img_id = maybe_get_first_last(det_df["image_id"])
                    row["x_det_first_image_id"] = first_img_id
                    row["x_det_last_image_id"] = last_img_id
                else:
                    notes.append("X.pkl missing column 'image_id'")

                n_track = count_distinct_track_ids(det_df)
                row["x_det_num_distinct_track_id"] = n_track
                if n_track is None:
                    notes.append("X.pkl missing both 'track_id' and fallback 'id'")

            except Exception as e:
                row["status"] = "ERROR"
                notes.append(f"Failed reading {det_member}: {e}")

            if row["status"] != "ERROR" and notes:
                row["status"] = "WARN"

            row["notes"] = " ; ".join(notes)
            rows.append(row)

        # Also detect extra archive members beyond number of directories
        expected_img = {f"{i}_image.pkl" for i in range(1, len(seq_names) + 1)}
        expected_det = {f"{i}.pkl" for i in range(1, len(seq_names) + 1)}
        archive_names = set(z.namelist())

        extra_img = sorted(n for n in archive_names if n.endswith("_image.pkl") and n not in expected_img)
        extra_det = sorted(
            n for n in archive_names
            if n.endswith(".pkl") and not n.endswith("_image.pkl") and n != "summary.json" and n not in expected_det
        )

        for name in extra_img + extra_det:
            rows.append(
                {
                    "split": split_name,
                    "sequence_name_dir": None,
                    "index": None,
                    "img_member": name if name.endswith("_image.pkl") else None,
                    "det_member": name if name.endswith(".pkl") and not name.endswith("_image.pkl") else None,
                    "sequence_name_from_file_path": None,
                    "file_path_match": None,
                    "x_image_video_id": None,
                    "x_image_first_frame": None,
                    "x_image_last_frame": None,
                    "x_image_first_id": None,
                    "x_image_last_id": None,
                    "x_det_video_id": None,
                    "x_det_num_distinct_track_id": None,
                    "x_det_first_image_id": None,
                    "x_det_last_image_id": None,
                    "status": "WARN",
                    "notes": f"Archive member has no corresponding sequence directory by index: {name}",
                }
            )

    finally:
        z.close()

    return rows

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify state archives directly against split directories for one or more splits."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=Path,
        help="Dataset root containing <split>/ directories and states/<split>.pklz",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to verify",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for split in args.splits:
        rows = verify_split(args.dataset_root, split)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print("No rows found.")
        return

    # nice terminal display
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 120,
    ):
        print(df.to_string(index=False))

    # compact summary
    print("\nSummary:")
    print(df["status"].value_counts(dropna=False).to_string())

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved CSV to: {args.out_csv}")

    # non-zero exit if any warnings/errors
    if (df["status"] != "OK").any():
        sys.exit(1)


if __name__ == "__main__":
    main()