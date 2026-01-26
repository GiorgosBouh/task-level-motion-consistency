#!/usr/bin/env python3
import argparse
import glob
import os
import re
from typing import List, Tuple, Optional

import pandas as pd


META_COLS = {
    "subject_id", "subject", "subj", "id",
    "path", "file", "trial", "trial_path",
    "rank",
    "score01", "score", "score_01",
    "mahalanobis_sq", "mahalanobis2", "mahalanobis",
    "label", "explanation", "reason", "summary",
    "task",
}


def _guess_task_from_filename(fp: str) -> str:
    base = os.path.basename(fp)
    # expected: <task>_anomaly_scores.csv
    if base.endswith("_anomaly_scores.csv"):
        return base.replace("_anomaly_scores.csv", "")
    # fallback: take before first "_"
    return base.split("_")[0]


def _pick_sort_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["mahalanobis_sq", "mahalanobis2", "mahalanobis", "score01", "score", "score_01"]:
        if c in df.columns:
            return c
    return None


def _find_subject_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["subject_id", "subject", "subj", "id"]:
        if c in df.columns:
            return c
    return None


def _numeric_family_columns(df: pd.DataFrame) -> List[str]:
    fam_cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        # Heuristic: keep columns that look like family names (letters/underscore/hyphen),
        # and are numeric or can be coerced to numeric.
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(c)):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        # if it's basically numeric -> treat as family
        fam_cols.append(c)
    return fam_cols


def build_profiles_from_scores(scores_files: List[str], topk: int) -> pd.DataFrame:
    out_rows = []

    for fp in sorted(scores_files):
        task = _guess_task_from_filename(fp)
        df = pd.read_csv(fp)

        subj_col = _find_subject_col(df)
        if subj_col is None:
            raise ValueError(f"[{fp}] Could not find subject column (expected one of subject_id/subject/subj/id).")

        sort_col = _pick_sort_col(df)
        if sort_col is None:
            raise ValueError(f"[{fp}] Could not find sort column (expected mahalanobis_sq or score01).")

        fam_cols = _numeric_family_columns(df)
        if not fam_cols:
            # If there are truly no family columns, we cannot build profiles meaningfully.
            # We'll skip this task with a warning-like print.
            print(f"⚠️  [{fp}] No numeric family columns found. Skipping task={task}.")
            continue

        # Coerce needed numeric columns
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
        for c in fam_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop rows without subject or without sort score
        df = df.dropna(subset=[subj_col, sort_col]).copy()
        df[subj_col] = df[subj_col].astype(int)

        # Per subject: take topK rows by sort_col desc
        df = df.sort_values([subj_col, sort_col], ascending=[True, False])

        for sid, g in df.groupby(subj_col, sort=False):
            top = g.head(topk).copy()
            if top.empty:
                continue

            # Sum families across topK
            fam_sum = top[fam_cols].sum(axis=0, skipna=True)

            # If everything zero/NaN -> skip
            total = float(fam_sum.sum())
            if not pd.notna(total) or total <= 0:
                continue

            fam_norm = fam_sum / total

            for fam, sc in fam_norm.items():
                if pd.isna(sc):
                    continue
                sc = float(sc)
                if sc <= 0:
                    continue
                out_rows.append(
                    {"subject_id": int(sid), "task": task, "family": str(fam), "score": sc}
                )

    out_df = pd.DataFrame(out_rows, columns=["subject_id", "task", "family", "score"])
    return out_df


def main():
    ap = argparse.ArgumentParser(description="Build cross-task family profiles from per-task anomaly_scores.csv files.")
    ap.add_argument("--topk", type=int, required=True, help="Top-K anomalies per subject per task to aggregate.")
    ap.add_argument(
        "--scores-glob",
        type=str,
        default="outputs/*_anomaly_scores.csv",
        help="Glob pattern for per-task anomaly score files.",
    )
    ap.add_argument("--out-csv", type=str, required=True, help="Output CSV path (subject_id,task,family,score).")
    args = ap.parse_args()

    files = glob.glob(args.scores_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.scores_glob}")

    out_df = build_profiles_from_scores(files, args.topk)
    if out_df.empty:
        raise SystemExit("No profiles were produced. Check that anomaly_scores.csv contain numeric family columns.")

    # Stable ordering
    out_df = out_df.sort_values(["subject_id", "task", "score"], ascending=[True, True, False])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote: {args.out_csv}  (rows={len(out_df)})")


if __name__ == "__main__":
    main()
