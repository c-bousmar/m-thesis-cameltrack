#!/usr/bin/env python3

import argparse
import json
import zipfile
from pathlib import Path


def print_summary(pklz_path: Path):
    if not pklz_path.exists():
        raise RuntimeError(f"File not found: {pklz_path}")

    with zipfile.ZipFile(pklz_path, "r") as zf:

        if "summary.json" not in zf.namelist():
            raise RuntimeError("summary.json not found inside archive.")

        summary = json.loads(zf.read("summary.json").decode("utf-8"))

    print("\n=== RAW summary.json content ===\n")
    print(json.dumps(summary, indent=2))

    print("\n=== Structure ===\n")
    print("Keys:", list(summary.keys()))

    if "num_sequences" in summary:
        print("num_sequences:", summary["num_sequences"])

    if "sequences" in summary:
        seqs = summary["sequences"]
        print("Number of sequences listed:", len(seqs))
        print("First 5 sequences:", seqs[:5])

    if "columns" in summary:
        print("Columns:", summary["columns"])

    print("\nDone.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args()

    print_summary(args.path)


if __name__ == "__main__":
    main()