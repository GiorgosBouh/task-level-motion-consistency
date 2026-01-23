#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
import json
import sys


# ----------------------------
# Parsing
# ----------------------------

LABEL_CANONICAL = {
    # normalize variants -> canonical
    "stand": "stand",
    "chair": "chair",
    "wheelchair": "wheelchair",
    "sit": "sit",
    "stand-frame": "stand-frame",   # keep as-is (or change to "stand_frame" if you prefer)
    "standframe": "stand-frame",
    "stand_frame": "stand-frame",
}

ALLOWED_LABELS = {"stand", "chair", "wheelchair", "sit", "stand-frame"}


@dataclass(frozen=True)
class SampleInfo:
    path: Path
    subject_id: int
    col2: str
    col3: str
    col4: str
    col5: str
    label_raw: str
    label: str  # canonical


FILENAME_RE = re.compile(
    r"""
    ^(?P<id>\d+)_            # subject id (numeric)
    (?P<c2>[^_]+)_
    (?P<c3>[^_]+)_
    (?P<c4>[^_]+)_
    (?P<c5>[^_]+)_
    (?P<label>[^_]+)         # label is last underscore token
    \.txt$
    """,
    re.VERBOSE,
)


def canonicalize_label(label_raw: str) -> str:
    k = label_raw.strip()
    k_low = k.lower()
    # Keep hyphen variations consistent
    k_low = k_low.replace("–", "-").replace("—", "-")
    return LABEL_CANONICAL.get(k_low, k)  # if unknown, keep original (so you can detect it)


def parse_filename(p: Path) -> SampleInfo:
    m = FILENAME_RE.match(p.name)
    if not m:
        raise ValueError(f"Filename does not match expected 6-part pattern: {p.name}")

    subject_id = int(m.group("id"))
    c2 = m.group("c2")
    c3 = m.group("c3")
    c4 = m.group("c4")
    c5 = m.group("c5")
    label_raw = m.group("label")
    label = canonicalize_label(label_raw)

    return SampleInfo(
        path=p,
        subject_id=subject_id,
        col2=c2,
        col3=c3,
        col4=c4,
        col5=c5,
        label_raw=label_raw,
        label=label,
    )


# ----------------------------
# Main dataset scan + stats
# ----------------------------

def scan_dataset(data_dir: Path) -> list[SampleInfo]:
    txt_files = sorted(data_dir.glob("*.txt"))
    samples: list[SampleInfo] = []
    errors: list[str] = []

    for p in txt_files:
        try:
            samples.append(parse_filename(p))
        except Exception as e:
            errors.append(f"{p.name}: {e}")

    if errors:
        print("⚠️ Some files could not be parsed:", file=sys.stderr)
        for err in errors[:50]:
            print("  -", err, file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors)-50} more", file=sys.stderr)

    return samples


def compute_stats(samples: list[SampleInfo]) -> dict:
    total = len(samples)

    label_counts = Counter(s.label for s in samples)
    raw_label_counts = Counter(s.label_raw for s in samples)

    subject_counts = Counter(s.subject_id for s in samples)

    # (subject, label) counts like your awk '{print $1, $(NF-1)}'
    subj_label_counts = Counter((s.subject_id, s.label) for s in samples)

    unknown_labels = sorted({s.label for s in samples if s.label.lower() not in ALLOWED_LABELS})

    # Extra sanity: all filenames should have exactly 6 underscore-separated tokens before .txt
    # Example: 101_18_0_1_1_stand.txt -> 6 tokens
    bad_nf = []
    for s in samples:
        stem_tokens = s.path.stem.split("_")  # without .txt
        if len(stem_tokens) != 6:
            bad_nf.append(s.path.name)

    return {
        "total_files": total,
        "label_counts": dict(label_counts),
        "raw_label_counts": dict(raw_label_counts),
        "subject_counts_top10": subject_counts.most_common(10),
        "subject_counts_all": dict(subject_counts),
        "subject_label_counts_top50": [((sid, lab), c) for ((sid, lab), c) in subj_label_counts.most_common(50)],
        "unknown_labels": unknown_labels,
        "bad_nf_files": bad_nf,
    }


def print_human(stats: dict) -> None:
    print(f"Total .txt files parsed: {stats['total_files']}\n")

    print("Counts per label (canonical):")
    for lab, c in sorted(stats["label_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c:5d}  {lab}")
    print()

    # Show any raw labels that differ (useful to confirm Stand-frame situation)
    print("Counts per label (raw, as in filename):")
    for lab, c in sorted(stats["raw_label_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {c:5d}  {lab}")
    print()

    print("Top subjects by #files:")
    for sid, c in stats["subject_counts_top10"]:
        print(f"  {c:5d}  {sid}")
    print()

    print("Top (subject, label) pairs:")
    for (sid, lab), c in stats["subject_label_counts_top50"]:
        print(f"  {c:5d}  {sid}  {lab}")
    print()

    if stats["unknown_labels"]:
        print("⚠️ Unknown labels detected (after canonicalization):")
        for lab in stats["unknown_labels"]:
            print("  -", lab)
        print()

    if stats["bad_nf_files"]:
        print("⚠️ Filenames with NF!=6 detected (should be empty in your case):")
        for name in stats["bad_nf_files"][:20]:
            print("  -", name)
        if len(stats["bad_nf_files"]) > 20:
            print(f"  ... and {len(stats['bad_nf_files'])-20} more")
        print()


# ----------------------------
# Optional: filter by task + export filelists
# ----------------------------

def export_filelists(samples: list[SampleInfo], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list[Path]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s.path)

    for label, paths in by_label.items():
        # safe filename for label (keep hyphen)
        out_path = out_dir / f"files_{label}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for p in sorted(paths):
                f.write(str(p) + "\n")

    print(f"✅ Wrote per-label filelists into: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Parse task-level-motion-consistency filenames and compute stats.")
    ap.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing *.txt files")
    ap.add_argument("--json-out", type=Path, default=None, help="Write stats JSON to this path")
    ap.add_argument("--export-filelists", type=Path, default=None, help="Write per-label file lists into this directory")
    args = ap.parse_args()

    samples = scan_dataset(args.data_dir)
    stats = compute_stats(samples)

    print_human(stats)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Wrote stats JSON: {args.json_out}")

    if args.export_filelists:
        export_filelists(samples, args.export_filelists)


if __name__ == "__main__":
    main()