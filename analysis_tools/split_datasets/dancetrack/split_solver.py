#!/usr/bin/env python3
"""
Create new DanceTrack train/val/test splits from the original train+val folders.

The script builds a manifest.json describing the repartition. It does not copy or
move any data. Each DanceTrack sequence is treated as an atomic assignable unit.

Default objective:
  - target split ratio: 50/25/25
  - balance number of sequences first
  - balance number of frames second

Expected dataset structure:
  DATASET_ROOT/
    train/
      dancetrack00XX/
        seqinfo.ini or sequinfo.ini
        img1/ or img/
        gt/
    val/
      dancetrack00YY/
        seqinfo.ini or sequinfo.ini
        img1/ or img/
        gt/

Example:
  python create_dancetrack_splits.py /path/to/DanceTrack
  python create_dancetrack_splits.py /path/to/DanceTrack --output /tmp/manifest.json
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ortools. Install it with `pip install ortools`."
    ) from exc


SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.50, 0.25, 0.25)
SEQ_RE = re.compile(r"^dancetrack(\d+)$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SequenceInfo:
    sequence_name: str
    orig_split: str
    orig_index: int
    frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced DanceTrack split manifest using CP-SAT."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the original DanceTrack root containing train/ and val/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path. Defaults to DATASET_ROOT/manifest.json.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=DEFAULT_RATIOS,
        help="Target split ratios. Default: 0.50 0.25 0.25.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=20.0,
        help="CP-SAT solving budget in seconds. Default: 20.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CP-SAT search workers. Default: 8.",
    )
    parser.add_argument(
        "--sequence-weight",
        type=int,
        default=100,
        help="Weight for sequence-count deviations. Default: 100.",
    )
    parser.add_argument(
        "--frame-weight",
        type=int,
        default=1,
        help="Weight for frame-count deviations. Default: 1.",
    )
    parser.add_argument(
        "--hard-sequence-counts",
        action="store_true",
        help=(
            "Force exact target sequence counts. For the default 50/25/25 ratio "
            "and 65 sequences this gives 33/16/16."
        ),
    )
    return parser.parse_args()


def natural_key(name: str) -> Tuple[int, str]:
    """Sort dancetrackXXXX by numeric suffix when possible."""
    match = SEQ_RE.match(name)
    if match:
        return int(match.group(1)), name
    return math.inf, name


def read_seq_length(seq_dir: Path) -> int:
    """Read sequence length from seqinfo.ini, falling back to image counting."""
    for ini_name in ("seqinfo.ini", "sequinfo.ini"):
        ini_path = seq_dir / ini_name
        if ini_path.exists():
            parser = configparser.ConfigParser()
            parser.read(ini_path)
            if parser.has_section("Sequence") and parser.has_option("Sequence", "seqLength"):
                return parser.getint("Sequence", "seqLength")

            # Robust fallback for non-standard formatting.
            text = ini_path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.strip().lower().startswith("seqlength"):
                    _, value = line.split("=", 1)
                    return int(value.strip())

    for img_dir_name in ("img1", "img"):
        img_dir = seq_dir / img_dir_name
        if img_dir.exists() and img_dir.is_dir():
            return sum(
                1
                for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

    raise FileNotFoundError(
        f"Could not determine frame count for {seq_dir}. Expected seqinfo.ini "
        "or an img1/img directory."
    )


def collect_sequences(dataset_root: Path) -> List[SequenceInfo]:
    sequences: List[SequenceInfo] = []

    for orig_split in ("train", "val"):
        split_dir = dataset_root / orig_split
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(f"Missing expected directory: {split_dir}")

        seq_dirs = sorted(
            [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("dancetrack")],
            key=lambda p: natural_key(p.name),
        )

        for idx, seq_dir in enumerate(seq_dirs, start=1):
            sequences.append(
                SequenceInfo(
                    sequence_name=seq_dir.name,
                    orig_split=orig_split,
                    orig_index=idx,
                    frames=read_seq_length(seq_dir),
                )
            )

    if not sequences:
        raise RuntimeError(f"No DanceTrack sequences found under {dataset_root}")

    names = [s.sequence_name for s in sequences]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate sequence names found: {duplicates}")

    return sorted(sequences, key=lambda s: natural_key(s.sequence_name))


def normalize_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("Expected exactly three ratios: train val test.")
    if any(r < 0 for r in ratios):
        raise ValueError("Ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one ratio must be positive.")
    return tuple(r / total for r in ratios)  # type: ignore[return-value]


def integer_targets(total: int, ratios: Sequence[float]) -> Dict[str, int]:
    """Largest-remainder rounding so targets sum exactly to total."""
    raw = [total * r for r in ratios]
    floors = [math.floor(x) for x in raw]
    remaining = total - sum(floors)
    order = sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remaining]:
        floors[i] += 1
    return dict(zip(SPLITS, floors))


def solve_assignment(
    sequences: List[SequenceInfo],
    ratios: Tuple[float, float, float],
    max_time: float,
    workers: int,
    sequence_weight: int,
    frame_weight: int,
    hard_sequence_counts: bool,
) -> Tuple[Dict[str, List[SequenceInfo]], Dict[str, int], Dict[str, int], float]:
    model = cp_model.CpModel()

    target_seq = integer_targets(len(sequences), ratios)
    total_frames = sum(s.frames for s in sequences)
    target_frames = integer_targets(total_frames, ratios)

    x = {}
    for i, seq in enumerate(sequences):
        for split in SPLITS:
            x[i, split] = model.NewBoolVar(f"x_{seq.sequence_name}_{split}")

    # Each sequence/video is assigned to exactly one new split.
    for i in range(len(sequences)):
        model.Add(sum(x[i, split] for split in SPLITS) == 1)

    dev_seq = {}
    dev_frames = {}

    for split in SPLITS:
        seq_expr = sum(x[i, split] for i in range(len(sequences)))
        frame_expr = sum(sequences[i].frames * x[i, split] for i in range(len(sequences)))

        if hard_sequence_counts:
            model.Add(seq_expr == target_seq[split])
            d_seq = model.NewIntVar(0, 0, f"dev_seq_{split}")
        else:
            d_seq = model.NewIntVar(0, len(sequences), f"dev_seq_{split}")
            model.AddAbsEquality(d_seq, seq_expr - target_seq[split])

        d_frames = model.NewIntVar(0, total_frames, f"dev_frames_{split}")
        model.AddAbsEquality(d_frames, frame_expr - target_frames[split])

        dev_seq[split] = d_seq
        dev_frames[split] = d_frames

    model.Minimize(
        sequence_weight * sum(dev_seq[s] for s in SPLITS)
        + frame_weight * sum(dev_frames[s] for s in SPLITS)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = workers

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible split assignment found.")

    assignments: Dict[str, List[SequenceInfo]] = {split: [] for split in SPLITS}
    for i, seq in enumerate(sequences):
        for split in SPLITS:
            if solver.Value(x[i, split]):
                assignments[split].append(seq)
                break

    for split in SPLITS:
        assignments[split].sort(key=lambda s: natural_key(s.sequence_name))

    return assignments, target_seq, target_frames, solver.ObjectiveValue()


def build_manifest(assignments: Dict[str, List[SequenceInfo]]) -> Dict[str, object]:
    manifest: Dict[str, object] = {}

    for split in SPLITS:
        manifest[split] = [
            {
                "new_index": new_idx,
                "sequence_name": seq.sequence_name,
                "orig_split": seq.orig_split,
                "orig_index": seq.orig_index,
            }
            for new_idx, seq in enumerate(assignments[split], start=1)
        ]

    manifest["counts"] = {split: len(assignments[split]) for split in SPLITS}
    return manifest


def print_summary(
    assignments: Dict[str, List[SequenceInfo]],
    target_seq: Dict[str, int],
    target_frames: Dict[str, int],
    objective_value: float,
) -> None:
    print("\nNew DanceTrack split summary")
    print("=" * 34)
    for split in SPLITS:
        seqs = assignments[split]
        n_seq = len(seqs)
        n_frames = sum(s.frames for s in seqs)
        print(f"{split:>5}: seq={n_seq:>2} target={target_seq[split]:>2} | "
              f"frames={n_frames:>6} target={target_frames[split]:>6}")
    print(f"Objective value: {objective_value}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else dataset_root / "manifest.json"

    ratios = normalize_ratios(args.ratios)
    sequences = collect_sequences(dataset_root)

    assignments, target_seq, target_frames, objective_value = solve_assignment(
        sequences=sequences,
        ratios=ratios,
        max_time=args.max_time,
        workers=args.workers,
        sequence_weight=args.sequence_weight,
        frame_weight=args.frame_weight,
        hard_sequence_counts=args.hard_sequence_counts,
    )

    manifest = build_manifest(assignments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_summary(assignments, target_seq, target_frames, objective_value)
    print(f"\nWrote manifest: {output_path}")


if __name__ == "__main__":
    main()
