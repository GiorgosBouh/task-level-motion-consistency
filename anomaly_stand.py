#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anomaly_stand.py

Compute anomaly scores for STAND trials only, using simple, explainable features
and a robust Mahalanobis distance model.

Input:
  - filelists/files_stand.txt  (one relative/absolute path per line to a .txt trial)

Assumptions about each .txt file:
  - Plain text numeric matrix with either:
      * 75 columns (25 joints * 3 coords), or
      * 3 columns (already a single 3D point over time), or
      * Any N columns where N is divisible by 3 (interpreted as K points * 3 coords)
  - Rows are time steps (T). T can vary across trials.

Outputs:
  - outputs/stand_anomaly_scores.csv
  - outputs/stand_feature_matrix.csv
  - outputs/stand_anomaly_topN.txt
  - (optional) a JSON with configuration

Usage examples:
  python anomaly_stand.py --filelist filelists/files_stand.txt --data-root ./ --outdir outputs
  python anomaly_stand.py --filelist filelists/files_stand.txt --outdir outputs --topn 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Config / utilities
# ----------------------------

@dataclass
class TrialMeta:
    path: str
    subject_id: Optional[str] = None
    label: Optional[str] = None


def parse_subject_and_label_from_filename(p: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expected filename pattern (based on your dataset):
      id_col2_col3_col4_col5_label.txt

    Example:
      101_18_0_1_1_stand.txt  -> subject_id=101, label=stand
    """
    name = Path(p).name
    m = re.match(r"^(\d+)_.*_([^_]+)\.txt$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def robust_center_scale(X: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust z-scoring per feature using median and MAD.
    Returns (Xz, median, scale)
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad  # approx std if normal
    scale = np.where(scale < eps, 1.0, scale)
    Xz = (X - med) / scale
    return Xz, med, scale


def safe_covariance(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    Covariance with ridge regularization to ensure invertibility.
    """
    C = np.cov(X, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    C = C + ridge * np.eye(C.shape[0])
    return C


# ----------------------------
# Loading / shaping
# ----------------------------

def load_trial_txt(path: Path) -> np.ndarray:
    """
    Loads a trial from a txt file into float32 array of shape (T, D).
    """
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

    # If the file has a single row, np.loadtxt returns shape (D,)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def to_points(trial: np.ndarray) -> np.ndarray:
    """
    Converts raw trial (T, D) into (T, K, 3) if D divisible by 3.
    If D == 3 => K=1.
    """
    T, D = trial.shape
    if D % 3 != 0:
        raise ValueError(f"Trial dimension D={D} is not divisible by 3 (cannot interpret as 3D points).")
    K = D // 3
    pts = trial.reshape(T, K, 3)
    return pts


def resample_time(pts: np.ndarray, n: int = 100) -> np.ndarray:
    """
    Linear resampling in time to a fixed length n.
    Input: (T, K, 3)
    Output: (n, K, 3)
    """
    T, K, C = pts.shape
    if T == n:
        return pts
    if T < 2:
        # Not enough samples; just tile
        return np.repeat(pts, repeats=n, axis=0)[:n]

    x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float32)

    out = np.empty((n, K, 3), dtype=np.float32)
    for k in range(K):
        for c in range(3):
            out[:, k, c] = np.interp(x_new, x_old, pts[:, k, c]).astype(np.float32)
    return out


# ----------------------------
# Feature extraction (explainable)
# ----------------------------

def extract_features_from_pts(pts: np.ndarray) -> Dict[str, float]:
    """
    pts: (n, K, 3) after resampling
    We compute a compact set of explainable kinematic summary features:
      - energy of velocity/acceleration (mean & std across points)
      - jerk energy
      - path length per point (mean & std)
      - spatial spread (mean std of position)
      - dominant axis ratio (variance ratio)
      - zero-motion fraction (near-zero speed)
      - correlation between axes (avg abs corr)
    """
    n, K, _ = pts.shape

    # Position centered per point to reduce subject/global offset effects
    pos = pts - pts.mean(axis=0, keepdims=True)  # (n,K,3)

    # Velocity / accel / jerk (finite differences)
    vel = np.diff(pos, axis=0)  # (n-1,K,3)
    acc = np.diff(vel, axis=0)  # (n-2,K,3)
    jerk = np.diff(acc, axis=0)  # (n-3,K,3)

    speed = np.linalg.norm(vel, axis=2)  # (n-1,K)
    accel_mag = np.linalg.norm(acc, axis=2)  # (n-2,K)
    jerk_mag = np.linalg.norm(jerk, axis=2)  # (n-3,K)

    # Path length per point
    path_len = speed.sum(axis=0)  # (K,)

    # Spatial spread: std of positions per axis, then mean across points
    pos_std = pos.std(axis=0)  # (K,3)
    spread_mean = float(np.mean(pos_std))
    spread_std = float(np.std(pos_std))

    # Velocity energy stats across points
    speed_mean_per_point = speed.mean(axis=0)  # (K,)
    speed_std_per_point = speed.std(axis=0)  # (K,)
    accel_mean_per_point = accel_mag.mean(axis=0) if acc.shape[0] > 0 else np.zeros((K,), dtype=np.float32)
    jerk_mean_per_point = jerk_mag.mean(axis=0) if jerk.shape[0] > 0 else np.zeros((K,), dtype=np.float32)

    # Dominant axis ratio: variance of x,y,z, averaged across points
    var_xyz = pos.var(axis=0)  # (K,3)
    var_sum = var_xyz.sum(axis=1) + 1e-9
    dom_ratio = (np.max(var_xyz, axis=1) / var_sum)  # (K,)
    dom_ratio_mean = float(dom_ratio.mean())
    dom_ratio_std = float(dom_ratio.std())

    # Zero-motion fraction: fraction of frames with near-zero speed (tuned threshold)
    zero_frac = float((speed < 1e-3).mean())

    # Axis correlations (avg abs corr) using flattened time series for each axis across all points
    # shape: (n, K, 3) -> (n, 3K) then corr among grouped axes
    X = pos.reshape(n, 3 * K)  # (n,3K)
    # Split into X/Y/Z stacks
    xs = X[:, 0::3]
    ys = X[:, 1::3]
    zs = X[:, 2::3]
    # Correlation for each point axis pair, then average abs
    def avg_abs_corr(A: np.ndarray, B: np.ndarray) -> float:
        # compute per-point corr quickly
        A0 = A - A.mean(axis=0, keepdims=True)
        B0 = B - B.mean(axis=0, keepdims=True)
        num = (A0 * B0).sum(axis=0)
        den = np.sqrt((A0**2).sum(axis=0) * (B0**2).sum(axis=0) + 1e-12)
        corr = num / den
        return float(np.mean(np.abs(corr)))

    corr_xy = avg_abs_corr(xs, ys)
    corr_xz = avg_abs_corr(xs, zs)
    corr_yz = avg_abs_corr(ys, zs)
    corr_mean = float(np.mean([corr_xy, corr_xz, corr_yz]))

    feats = {
        # Path length
        "path_len_mean": float(path_len.mean()),
        "path_len_std": float(path_len.std()),

        # Speed stats
        "speed_mean_mean": float(speed_mean_per_point.mean()),
        "speed_mean_std": float(speed_mean_per_point.std()),
        "speed_std_mean": float(speed_std_per_point.mean()),
        "speed_std_std": float(speed_std_per_point.std()),

        # Accel / jerk
        "accel_mean_mean": float(accel_mean_per_point.mean()),
        "accel_mean_std": float(accel_mean_per_point.std()),
        "jerk_mean_mean": float(jerk_mean_per_point.mean()),
        "jerk_mean_std": float(jerk_mean_per_point.std()),

        # Spatial spread
        "spread_mean": spread_mean,
        "spread_std": spread_std,

        # Dominant axis ratio
        "dom_ratio_mean": dom_ratio_mean,
        "dom_ratio_std": dom_ratio_std,

        # Other
        "zero_speed_frac": zero_frac,
        "axis_corr_mean_abs": corr_mean,
    }
    return feats


# ----------------------------
# Mahalanobis anomaly scoring
# ----------------------------

def mahalanobis_scores(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - scores: squared Mahalanobis distance (higher => more anomalous)
      - mu: mean in z-scored space
      - inv_cov: inverse covariance
    """
    mu = X.mean(axis=0)
    C = safe_covariance(X, ridge=1e-3)
    invC = np.linalg.inv(C)
    dif = X - mu
    # vectorized quadratic form
    scores = np.einsum("ij,jk,ik->i", dif, invC, dif)
    return scores, mu, invC


# ----------------------------
# Main
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", type=str, default="filelists/files_stand.txt",
                   help="Path to file containing one trial path per line (STAND).")
    p.add_argument("--data-root", type=str, default=".",
                   help="Optional root to prepend to relative paths in filelist.")
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Output directory.")
    p.add_argument("--resample-n", type=int, default=100,
                   help="Resample each trial to this number of time steps.")
    p.add_argument("--topn", type=int, default=20,
                   help="How many top anomalies to write in a text file.")
    p.add_argument("--min-frames", type=int, default=10,
                   help="Skip trials with fewer than this many frames (rows).")
    p.add_argument("--save-config", action="store_true",
                   help="Save the run configuration as JSON.")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    filelist_path = Path(args.filelist)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not filelist_path.exists():
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")

    raw_lines = [ln.strip() for ln in filelist_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        print("No files in filelist.")
        return 0

    metas: List[TrialMeta] = []
    feature_dicts: List[Dict[str, float]] = []
    kept_paths: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for rel in raw_lines:
        trial_path = Path(rel)
        if not trial_path.is_absolute():
            trial_path = data_root / trial_path
        if not trial_path.exists():
            skipped.append((str(trial_path), "missing"))
            continue

        try:
            trial = load_trial_txt(trial_path)
            if trial.shape[0] < args.min_frames:
                skipped.append((str(trial_path), f"too_few_frames(T={trial.shape[0]})"))
                continue

            pts = to_points(trial)
            pts = resample_time(pts, n=args.resample_n)
            feats = extract_features_from_pts(pts)

            sid, lab = parse_subject_and_label_from_filename(str(trial_path))
            metas.append(TrialMeta(path=str(trial_path), subject_id=sid, label=lab))
            feature_dicts.append(feats)
            kept_paths.append(str(trial_path))

        except Exception as e:
            skipped.append((str(trial_path), f"error:{e}"))
            continue

    if not feature_dicts:
        print("No trials were successfully processed.")
        if skipped:
            print(f"Skipped: {len(skipped)} (showing first 10)")
            for p, why in skipped[:10]:
                print("  ", p, "->", why)
        return 0

    # Build feature matrix in stable column order
    feat_names = sorted(feature_dicts[0].keys())
    X = np.array([[fd[n] for n in feat_names] for fd in feature_dicts], dtype=np.float64)

    # Robust z-score, then Mahalanobis
    Xz, med, scale = robust_center_scale(X)
    scores, mu, invC = mahalanobis_scores(Xz)

    # Normalize scores to [0,1] for convenience (rank-based)
    ranks = scores.argsort().argsort().astype(np.float64)
    score01 = ranks / max(1.0, (len(scores) - 1))

    # Write feature matrix CSV
    feat_csv = outdir / "stand_feature_matrix.csv"
    with feat_csv.open("w", encoding="utf-8") as f:
        header = ["path", "subject_id", "label"] + feat_names
        f.write(",".join(header) + "\n")
        for meta, row in zip(metas, X):
            row_str = [meta.path, meta.subject_id or "", meta.label or ""] + [f"{v:.10g}" for v in row]
            f.write(",".join(row_str) + "\n")

    # Write anomaly scores CSV
    scores_csv = outdir / "stand_anomaly_scores.csv"
    with scores_csv.open("w", encoding="utf-8") as f:
        header = ["path", "subject_id", "label", "mahalanobis_sq", "score01"]
        f.write(",".join(header) + "\n")
        for meta, s, s01 in zip(metas, scores, score01):
            f.write(",".join([
                meta.path,
                meta.subject_id or "",
                meta.label or "",
                f"{float(s):.10g}",
                f"{float(s01):.10g}",
            ]) + "\n")

    # Top-N anomalies (highest score)
    topn = min(args.topn, len(scores))
    top_idx = np.argsort(scores)[::-1][:topn]
    top_txt = outdir / "stand_anomaly_topN.txt"
    with top_txt.open("w", encoding="utf-8") as f:
        f.write(f"Top {topn} anomalies (stand) by squared Mahalanobis distance\n")
        f.write("rank,mahalanobis_sq,score01,subject_id,label,path\n")
        for r, i in enumerate(top_idx, start=1):
            meta = metas[i]
            f.write(",".join([
                str(r),
                f"{float(scores[i]):.10g}",
                f"{float(score01[i]):.10g}",
                meta.subject_id or "",
                meta.label or "",
                meta.path
            ]) + "\n")

    # Save config/model params for reproducibility
    model_json = outdir / "stand_anomaly_model.json"
    model_payload = {
        "filelist": str(filelist_path),
        "data_root": str(data_root),
        "resample_n": int(args.resample_n),
        "min_frames": int(args.min_frames),
        "feature_names": feat_names,
        "robust_median": med.tolist(),
        "robust_scale": scale.tolist(),
        "mahalanobis_mu": mu.tolist(),
        "mahalanobis_inv_cov": invC.tolist(),
        "n_trials_used": int(len(scores)),
        "n_trials_skipped": int(len(skipped)),
        "skipped_examples": skipped[:20],
    }
    model_json.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    if args.save_config:
        cfg_json = outdir / "run_config.json"
        cfg_json.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Console summary
    print(f"✅ Processed trials: {len(scores)}")
    print(f"⚠️ Skipped trials:   {len(skipped)}")
    print(f"📁 Wrote features:   {feat_csv}")
    print(f"📁 Wrote scores:     {scores_csv}")
    print(f"📁 Wrote top-N:      {top_txt}")
    print(f"📁 Wrote model:      {model_json}")

    # Show quick peek
    print("\nTop-10 anomalies (by mahalanobis_sq):")
    for i in top_idx[: min(10, len(top_idx))]:
        meta = metas[i]
        print(f"  score={scores[i]:.3f} (s01={score01[i]:.3f}) | subj={meta.subject_id} | {meta.path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())