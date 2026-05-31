#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path
import pandas as pd


def build_targets(seq: list[int]) -> list[str]:
    targets = []
    for n in seq:
        targets.append(f"{n}.pkl")
        targets.append(f"{n}_image.pkl")
    return targets


def write_listing(zf: zipfile.ZipFile, out_dir: Path) -> None:
    (out_dir / "pklz_listing.txt").write_text(
        "\n".join(sorted(zf.namelist())) + "\n",
        encoding="utf-8",
    )


def dump_summary_json(zf: zipfile.ZipFile, out_dir: Path) -> None:
    if "summary.json" in zf.namelist():
        summary = json.loads(zf.read("summary.json").decode("utf-8"))
        (out_dir / "summary.json.txt").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


def dump_targets(zf: zipfile.ZipFile, out_dir: Path, targets: list[str]) -> None:
    # map basename -> archive members (handles folders in zip)
    by_basename = {}
    for name in zf.namelist():
        by_basename.setdefault(Path(name).name, []).append(name)

    for target in targets:
        for member in by_basename.get(target, []):
            raw = zf.read(member)
            obj = pickle.loads(raw)

            safe_name = member.replace("/", "__").replace(".pkl", ".csv")
            out_path = out_dir / safe_name

            if isinstance(obj, pd.Series):
                obj.to_csv(out_path)
            else:  # assume DataFrame
                obj.to_csv(out_path, index=True)

            print(f"Saved: {out_path}")


def parse_seq(value: str) -> list[int]:
    """Parse a comma-separated list of integers, e.g. '1' or '1,2,6,34'."""
    try:
        return [int(v.strip()) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid --seq value '{value}': expected integers separated by commas (e.g. '1' or '1,2,6,34')"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pklz-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--seq",
        required=True,
        type=parse_seq,
        metavar="N[,N...]",
        help="Sequence numbers to extract, e.g. '1' or '1,2,6,34'",
    )
    args = parser.parse_args()

    targets = build_targets(args.seq)
    print(f"Targets: {targets}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.pklz_file.exists():
        raise RuntimeError(f"File not found: {args.pklz_file}")

    with zipfile.ZipFile(args.pklz_file, "r") as zf:
        write_listing(zf, out_dir)
        dump_summary_json(zf, out_dir)
        dump_targets(zf, out_dir, targets)


if __name__ == "__main__":
    main()