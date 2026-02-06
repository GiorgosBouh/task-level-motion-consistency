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
    "jerk": ["jerk_mean", "jerk_std", "jerk_range"],
    "smoothness": ["smoothness_mean", "smoothness_std", "smoothness_range"],
}


def fit_gaussian_shrinkage(X: np.ndarray):
    """
    Returns mean, precision matrix (inverse covariance) using Ledoit-Wolf shrinkage.
    """
    if LedoitWolf is None:
        raise RuntimeError("scikit-learn is required for LedoitWolf shrinkage. Install scikit-learn.")
    mu = X.mean(axis=0)
    lw = LedoitWolf().fit(X)
    prec = lw.precision_
    return mu, prec


def mahalanobis_sq(X: np.ndarray, mu: np.ndarray, prec: np.ndarray):
    D = X - mu
    # (x-mu)^T P (x-mu)
    return np.einsum("ij,jk,ik->i", D, prec, D)


def per_feature_contrib(X: np.ndarray, mu: np.ndarray, prec: np.ndarray):
    """
    Decompose Mahalanobis: d^2 = sum_i (x-mu)_i * (P*(x-mu))_i
    This yields signed contributions; we use absolute value for "importance".
    """
    D = X - mu
    PD = D @ prec.T
    c = D * PD   # signed
    return c


def minmax01(x: np.ndarray):
    x = np.asarray(x, float)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True, help="Output of extract_features.py")
    ap.add_argument("--groupby", default="posture", choices=["posture", "context"], help="Task grouping")
    ap.add_argument("--out-dir", default="outputs", help="Output directory")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    os.makedirs(args.out_dir, exist_ok=True)

    # feature columns used in model
    feat_cols = []
    for fam, cols in FAMILIES.items():
        feat_cols.extend(cols)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}. Did extract_features.py run correctly?")

    for task, g in df.groupby(args.groupby):
        g = g.copy()
        X = g[feat_cols].to_numpy(dtype=float)

        mu, prec = fit_gaussian_shrinkage(X)
        d2 = mahalanobis_sq(X, mu, prec)
        score01 = minmax01(d2)

        contrib = per_feature_contrib(X, mu, prec)  # [n, p]
        contrib_abs = np.abs(contrib)
        # normalize per row so families sum to 1
        row_sum = contrib_abs.sum(axis=1, keepdims=True) + 1e-12
        contrib_norm = contrib_abs / row_sum

        # aggregate into families
        fam_scores = {}
        for fam, cols in FAMILIES.items():
            idx = [feat_cols.index(c) for c in cols]
            fam_scores[fam] = contrib_norm[:, idx].sum(axis=1)

        out = pd.DataFrame({
            "path": g["path"].values,
            "subject_id": g["subject_id"].values,
            "posture": g["posture"].values,
            "gesture": g["gesture"].values,
            "context": g["context"].values,
            "task": task,
            "mahalanobis_sq": d2,
            "score01": score01,
            **{fam: fam_scores[fam] for fam in fam_scores}
        })

        out = out.sort_values("score01", ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))

        safe = str(task).replace("/", "_")
        fp = os.path.join(args.out_dir, f"{safe}_anomaly_scores.csv")
        out.to_csv(fp, index=False)
        print(f"✅ Wrote {fp} rows={len(out)}")

if __name__ == "__main__":
    main()
