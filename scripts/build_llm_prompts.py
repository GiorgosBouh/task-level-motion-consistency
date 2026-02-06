#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any

import pandas as pd

FAMILIES = ["speed", "accel", "jerk", "smoothness"]

SYSTEM_INSTRUCTIONS = """You are an assistant that produces STRICT JSON ONLY.
No prose, no markdown, no extra keys.

You will be given an evidence object for one motion trial. The evidence contains:
- task context
- anomaly severity percentiles (score01_pct, mahal2_pct) within the same task
- normalized family contributions (speed_c, accel_c, jerk_c, smoothness_c) that sum to ~1
- percentiles of those contributions within the same task (speed_pct, accel_pct, jerk_pct, smoothness_pct)

Your job is NOT clinical interpretation. Your job is to report which feature families are supported by the evidence.
You MUST use ONLY the evidence fields. Do not invent causes (e.g., sensor noise) and do not mention segmentation or labels.

Return JSON with this schema:
{
  "path": str,
  "task": str,
  "severity": {"score01_pct": float, "mahal2_pct": float},
  "supported_families": [
     {"family": "speed|accel|jerk|smoothness", "support": "high|medium|low", "evidence": {"pct": float, "contribution": float}}
  ],
  "dominant_family": {"family": "...", "contribution": float, "pct": float},
  "abstain": bool,
  "notes": str
}

Rules:
- Use support="high" if family_pct >= 0.90 OR contribution >= 0.70.
- support="medium" if 0.75 <= family_pct < 0.90 OR 0.45 <= contribution < 0.70.
- support="low" otherwise.
- abstain=true if severity.score01_pct < 0.90 AND no family has support="high".
- notes must be <= 25 words, neutral, and refer only to evidence (e.g., "Anomaly dominated by jerk contribution with high percentile.").
"""

def build_prompt(e: Dict[str, Any]) -> Dict[str, Any]:
    # We keep everything needed inside one prompt string so it’s portable.
    evidence = {
        "path": e["path"],
        "task": e["task"],
        "score01_pct": float(e["score01_pct"]),
        "mahal2_pct": float(e["mahal2_pct"]),
        "dominant_family": e["dominant_family"],
        "dominant_mass": float(e["dominant_mass"]),
        "family_entropy": float(e["family_entropy"]),
    }
    for f in FAMILIES:
        evidence[f"{f}_c"] = float(e[f"{f}_c"])
        evidence[f"{f}_pct"] = float(e[f"{f}_pct"])

    prompt = SYSTEM_INSTRUCTIONS + "\n\nEVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False)
    return {
        "path": e["path"],
        "task": e["task"],
        "prompt": prompt,
        "evidence": evidence
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0, help="If >0, write only first N rows (debug).")
    args = ap.parse_args()

    df = pd.read_csv(args.summaries_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = build_prompt(r.to_dict())
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ wrote prompts: {args.out_jsonl} (rows={len(df)})")
    print("Example keys:", ["path","task","prompt","evidence"])

if __name__ == "__main__":
    main()
