#!/usr/bin/env python3
import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FAMILIES = ["speed", "accel", "jerk", "smoothness"]

META_COLS = [
    "rank","path","subject_id","date_id","gesture","repetition","correctness",
    "posture","label","task","mahalanobis_sq","score01",
]

def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _percentile_rank(s: pd.Series) -> pd.Series:
    # Percentile rank in [0,1]. If constant, returns 0.5.
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True, method="average")

def _renorm_row_families(row: pd.Series) -> Tuple[Dict[str, float], float]:
    vals = {f: _safe_float(row.get(f, 0.0), 0.0) for f in FAMILIES}
    total = float(sum(max(0.0, v) for v in vals.values()))
    if total <= 0:
        return {f: 0.0 for f in FAMILIES}, 0.0
    vals = {f: max(0.0, float(v))/total for f, v in vals.items()}
    return vals, 1.0

def build_summaries(scores_glob: str, out_csv: str, out_jsonl: str) -> None:
    files = sorted(glob.glob(scores_glob))
    if not files:
        raise SystemExit(f"No files matched: {scores_glob}")

    all_rows = []
    for fp in files:
        df = pd.read_csv(fp)

        # sanity: must contain families
        missing = [f for f in FAMILIES if f not in df.columns]
        if missing:
            raise SystemExit(f"[{fp}] Missing family cols: {missing}. Found: {df.columns.tolist()}")

        # coerce numeric
        for c in ["mahalanobis_sq","score01"] + FAMILIES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop rows without essentials
        df = df.dropna(subset=["path","task","mahalanobis_sq","score01"]).copy()

        # renormalize contributions to sum to 1 per row (defensive)
        fam_norm = []
        for _, r in df.iterrows():
            v, ok = _renorm_row_families(r)
            fam_norm.append(v)
        fam_norm_df = pd.DataFrame(fam_norm)
        for f in FAMILIES:
            df[f"{f}_c"] = fam_norm_df[f].astype(float)

        # within-task percentiles for score and each family contribution
        df["score01_pct"] = _percentile_rank(df["score01"])
        df["mahal2_pct"] = _percentile_rank(df["mahalanobis_sq"])
        for f in FAMILIES:
            df[f"{f}_pct"] = _percentile_rank(df[f"{f}_c"])

        # dominance / clarity proxies (objective)
        df["dominant_family"] = df[[f"{f}_c" for f in FAMILIES]].idxmax(axis=1).str.replace("_c","")
        df["dominant_mass"] = df[[f"{f}_c" for f in FAMILIES]].max(axis=1)
        df["family_entropy"] = -(df[[f"{f}_c" for f in FAMILIES]]
                                .replace(0.0, np.nan)
                                .apply(lambda row: np.nansum(row*np.log(row)), axis=1)
                                .fillna(0.0))

        # Create compact deterministic “evidence summary”
        out = pd.DataFrame({
            "task": df["task"].astype(str),
            "rank": df.get("rank", pd.Series([np.nan]*len(df))).astype("Int64"),
            "path": df["path"].astype(str),
            "score01": df["score01"].astype(float),
            "score01_pct": df["score01_pct"].astype(float),
            "mahalanobis_sq": df["mahalanobis_sq"].astype(float),
            "mahal2_pct": df["mahal2_pct"].astype(float),
            "dominant_family": df["dominant_family"].astype(str),
            "dominant_mass": df["dominant_mass"].astype(float),
            "family_entropy": df["family_entropy"].astype(float),
        })

        for f in FAMILIES:
            out[f"{f}_c"] = df[f"{f}_c"].astype(float)
            out[f"{f}_pct"] = df[f"{f}_pct"].astype(float)

        # keep minimal metadata if exists (optional; NOT used by Layer A rules)
        for c in ["subject_id","date_id","gesture","repetition","correctness","posture","label"]:
            if c in df.columns:
                out[c] = df[c]

        all_rows.append(out)

    big = pd.concat(all_rows, ignore_index=True)

    # stable ordering
    big = big.sort_values(["task","score01"], ascending=[True, False]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    big.to_csv(out_csv, index=False)

    # JSONL (one object per trial) for LLM input / eval
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, r in big.iterrows():
            obj = r.to_dict()
            # make numpy/pandas types JSON safe
            for k, v in list(obj.items()):
                if isinstance(v, (np.integer,)):
                    obj[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    obj[k] = float(v)
                elif pd.isna(v):
                    obj[k] = None
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ wrote summaries CSV:  {out_csv}  (rows={len(big)})")
    print(f"✅ wrote summaries JSONL: {out_jsonl}  (rows={len(big)})")
    print("Columns:", big.columns.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-glob", required=True, help='e.g. "outputs_unfiltered/*_anomaly_scores.csv"')
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()
    build_summaries(args.scores_glob, args.out_csv, args.out_jsonl)

if __name__ == "__main__":
    main()
