import re
from pathlib import Path
from collections import defaultdict

LINE_RE = re.compile(
    r"""^(?:├──|└──)\s+
        (?P<seq>\S+)\s+
        (?P<tag>[bfv]\d+)\s+
        (?P<frames>\d+)\s*$
    """,
    re.VERBOSE,
)

def parse_tree_txt(txt_path: Path):
    splits = defaultdict(list)
    current_split = None

    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if line in {"train", "val", "test"}:
            current_split = line
            continue

        m = LINE_RE.match(line)
        if not m or current_split is None:
            continue

        tag = m.group("tag")
        splits[current_split].append(
            {
                "seq": m.group("seq"),
                "type": tag[0],      # b/f/v
                "video": tag,        # b01, f02, ...
                "frames": int(m.group("frames")),
            }
        )

    return splits


def summarize(split_items):
    out = defaultdict(lambda: {"seq": 0, "frames": 0})
    for it in split_items:
        t = it["type"]
        out[t]["seq"] += 1
        out[t]["frames"] += it["frames"]
    return out


def totals(stats_by_type):
    s = sum(v["seq"] for v in stats_by_type.values())
    f = sum(v["frames"] for v in stats_by_type.values())
    return s, f


def print_repartition(splits):
    global_stats = defaultdict(lambda: {"seq": 0, "frames": 0})

    print("\n# Repartition per split (by type)\n")

    for split in ["train", "val", "test"]:
        items = splits.get(split, [])
        st = summarize(items)
        split_seq, split_frames = totals(st)

        print(f"== {split.upper()} ==")
        if split_seq == 0:
            print("  (empty)\n")
            continue

        for t in sorted(st.keys()):
            s = st[t]["seq"]
            f = st[t]["frames"]
            print(f"  {t}: {s:3d} seq | {f:6d} frames | "
                  f"{s/split_seq:6.2%} of seq | {f/split_frames:6.2%} of frames")

            global_stats[t]["seq"] += s
            global_stats[t]["frames"] += f

        print(f"  TOTAL: {split_seq} seq | {split_frames} frames\n")

    g_seq, g_frames = totals(global_stats)

    print("== GLOBAL ==")
    for t in sorted(global_stats.keys()):
        s = global_stats[t]["seq"]
        f = global_stats[t]["frames"]
        print(f"  {t}: {s:3d} seq ({s/g_seq:6.2%}) | {f:6d} frames ({f/g_frames:6.2%})")
    print(f"  TOTAL: {g_seq} seq | {g_frames} frames\n")

    return global_stats


def suggest_new_split_targets(splits):
    """
    Suggest target sizes for new (train_new, val_new, test_new) built from (train+val) only,
    keeping the original split ratios (based on frames and sequences).
    """
    # Original split totals
    orig_split_totals = {}
    for split in ["train", "val", "test"]:
        st = summarize(splits.get(split, []))
        orig_split_totals[split] = totals(st)  # (seq, frames)

    orig_total_seq = sum(v[0] for v in orig_split_totals.values())
    orig_total_frames = sum(v[1] for v in orig_split_totals.values())

    # Available pool = train+val only
    pool = splits.get("train", []) + splits.get("val", [])
    pool_stats = summarize(pool)
    pool_seq, pool_frames = totals(pool_stats)

    # Ratios from original dataset
    ratios = {}
    for split, (s, f) in orig_split_totals.items():
        ratios[split] = {
            "seq_ratio": (s / orig_total_seq) if orig_total_seq else 0.0,
            "frame_ratio": (f / orig_total_frames) if orig_total_frames else 0.0,
        }

    # Targets inside the pool
    print("# Suggested targets for NEW split built from TRAIN+VAL only\n")
    print(f"Pool (train+val): {pool_seq} seq | {pool_frames} frames\n")

    for split in ["train", "val", "test"]:
        tgt_seq = round(pool_seq * ratios[split]["seq_ratio"])
        tgt_frames = round(pool_frames * ratios[split]["frame_ratio"])
        print(f"{split}_new targets:")
        print(f"  ~{tgt_seq} sequences (using original {ratios[split]['seq_ratio']:.2%} seq ratio)")
        print(f"  ~{tgt_frames} frames    (using original {ratios[split]['frame_ratio']:.2%} frame ratio)\n")

def main(txt_path: str):
    splits = parse_tree_txt(Path(txt_path))
    print_repartition(splits)
    suggest_new_split_targets(splits)

if __name__ == "__main__":
    # Example:
    # python repartition.py /path/to/dataset_tree.txt
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python repartition.py /path/to/dataset_tree.txt")
    main(sys.argv[1])