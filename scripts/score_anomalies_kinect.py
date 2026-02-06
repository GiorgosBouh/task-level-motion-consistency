#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
except Exception:
    LedoitWolf = None

FAMILIES = {
    "speed": ["speed_mean", "speed_std", "speed_range"],
    "accel": ["accel_mean", "accel_std", "accel_range"],
    "jerk":  ["jerk_mean",  "jerk_std",  "jerk_range"],
    "smoothness": ["smoothness_mean", "smoothness_std", "smoothness_range"],
}

META = ["path","subject_id","date_id","gesture","repetition","correctness","posture"]

def _fit_precision(X: np.ndarray):
    if LedoitWolf is None:
        raise RuntimeError("scikit-learn is required (LedoitWolf). Install: pip install scikit-learn")
    lw = LedoitWolf().fit(X)
    mu = lw.location_
    P = lw.precision_
    return mu, P

def _mahal_and_contribs(X: np.ndarray, mu: np.ndarray, P: np.ndarray):
    """
    mahal^2 = d^T P d
    per-feature additive decomposition:
      mahal^2 = sum_i d_i * (P d)_i
    Then family contribution = sum over features in that family.
    """
    d = X - mu
    Pd = d @ P
    contrib_feat = d * Pd                       # (n, p)
    mahal2 = np.sum(contrib_feat, axis=1)       # (n,)
    # numerical safety
    mahal2 = np.maximum(mahal2, 0.0)
    return mahal2, contrib_feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--groupby", choices=["posture","posture_gesture"], default="posture")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)

    # group key
    if args.groupby == "posture":
        df["task"] = df["posture"].astype(str)
    else:
        df["task"] = df["posture"].astype(str) + "__" + df["gesture"].astype(str)

    # ensure columns exist
    feat_cols = []
    for fam, cols in FAMILIES.items():
        for c in cols:
            if c not in df.columns:
                raise SystemExit(f"Missing feature column: {c}")
            feat_cols.append(c)

    # coerce numeric
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing feats
    df = df.dropna(subset=feat_cols + ["subject_id","task","path"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)

    os.makedirs(args.out_dir, exist_ok=True)

    total_rows = 0
    for task, g in df.groupby("task", sort=True):
        X = g[feat_cols].to_numpy(dtype=float)
        mu, P = _fit_precision(X)
        mahal2, contrib_feat = _mahal_and_contribs(X, mu, P)

        out = g[META].copy()
        out["label"] = g["posture"].astype(str)  # keep compatibility with older naming
        out["task"] = task
        out["mahalanobis_sq"] = mahal2

        # convert to 0..1 style score01 (monotonic)
        # use robust scaling per task: score01 = rank-based percentile
        ranks = pd.Series(mahal2).rank(method="average", pct=True).to_numpy()
        out["score01"] = ranks

        # family contributions as FRACTION of mahal^2 (so summable and comparable)
        eps = 1e-12
        denom = (mahal2 + eps).reshape(-1, 1)

        # per-family = sum of per-feature additive terms for features in family
        for fam, cols in FAMILIES.items():
            idx = [feat_cols.index(c) for c in cols]
            fam_raw = np.sum(contrib_feat[:, idx], axis=1)
            fam_raw = np.maximum(fam_raw, 0.0)
            out[fam] = fam_raw / (mahal2 + eps)

        # sort: highest anomaly first
        out = out.sort_values("mahalanobis_sq", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))

        # write
        safe = task.replace("/", "_")
        fp = os.path.join(args.out_dir, f"{safe}_anomaly_scores.csv")
        out.to_csv(fp, index=False)

        total_rows += len(out)
        print(f"✅ wrote {fp} | rows={len(out)} | task={task}")

    print(f"\nDONE. Total scored rows: {total_rows} | tasks: {df['task'].nunique()}")
    print("Output dir:", args.out_dir)

if __name__ == "__main__":
    main()
