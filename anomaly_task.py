#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class TrialMeta:
    path: str
    subject_id: Optional[str] = None
    label: Optional[str] = None

def parse_subject_and_label_from_filename(p: str) -> Tuple[Optional[str], Optional[str]]:
    name = Path(p).name
    m = re.match(r"^(\d+)_.*_([^_]+)\.txt$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def robust_center_scale(X: np.ndarray, eps: float = 1e-9):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, 1.0, scale)
    return (X - med) / scale, med, scale

def safe_covariance(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    C = np.cov(X, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    return C + ridge * np.eye(C.shape[0])

def mahalanobis_scores(X: np.ndarray):
    mu = X.mean(axis=0)
    C = safe_covariance(X, ridge=1e-3)
    invC = np.linalg.inv(C)
    dif = X - mu
    scores = np.einsum("ij,jk,ik->i", dif, invC, dif)
    return scores, mu, invC

def load_trial_txt(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float32, delimiter=",")
        if arr.size > 0:
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr
    except Exception:
        pass
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def to_points(trial: np.ndarray) -> np.ndarray:
    T, D = trial.shape
    if D % 3 != 0:
        raise ValueError(f"D={D} not divisible by 3")
    return trial.reshape(T, D // 3, 3)

def resample_time(pts: np.ndarray, n: int = 100) -> np.ndarray:
    T, K, _ = pts.shape
    if T == n:
        return pts
    if T < 2:
        return np.repeat(pts, repeats=n, axis=0)[:n]
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.empty((n, K, 3), dtype=np.float32)
    for k in range(K):
        for c in range(3):
            out[:, k, c] = np.interp(x_new, x_old, pts[:, k, c]).astype(np.float32)
    return out

def extract_features_from_pts(pts: np.ndarray) -> Dict[str, float]:
    n, K, _ = pts.shape
    pos = pts - pts.mean(axis=0, keepdims=True)
    vel = np.diff(pos, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)

    speed = np.linalg.norm(vel, axis=2)
    accel_mag = np.linalg.norm(acc, axis=2) if acc.shape[0] > 0 else np.zeros((max(n-2, 0), K), dtype=np.float32)
    jerk_mag = np.linalg.norm(jerk, axis=2) if jerk.shape[0] > 0 else np.zeros((max(n-3, 0), K), dtype=np.float32)

    path_len = speed.sum(axis=0)

    pos_std = pos.std(axis=0)
    spread_mean = float(np.mean(pos_std))
    spread_std = float(np.std(pos_std))

    speed_mean_per_point = speed.mean(axis=0)
    speed_std_per_point = speed.std(axis=0)

    accel_mean_per_point = accel_mag.mean(axis=0) if accel_mag.size else np.zeros((K,), dtype=np.float32)
    jerk_mean_per_point = jerk_mag.mean(axis=0) if jerk_mag.size else np.zeros((K,), dtype=np.float32)

    var_xyz = pos.var(axis=0)
    var_sum = var_xyz.sum(axis=1) + 1e-9
    dom_ratio = np.max(var_xyz, axis=1) / var_sum

    zero_frac = float((speed < 1e-3).mean())

    X = pos.reshape(n, 3 * K)
    xs, ys, zs = X[:, 0::3], X[:, 1::3], X[:, 2::3]

    def avg_abs_corr(A: np.ndarray, B: np.ndarray) -> float:
        A0 = A - A.mean(axis=0, keepdims=True)
        B0 = B - B.mean(axis=0, keepdims=True)
        num = (A0 * B0).sum(axis=0)
        den = np.sqrt((A0**2).sum(axis=0) * (B0**2).sum(axis=0) + 1e-12)
        return float(np.mean(np.abs(num / den)))

    corr_mean = float(np.mean([avg_abs_corr(xs, ys), avg_abs_corr(xs, zs), avg_abs_corr(ys, zs)]))

    return {
        "path_len_mean": float(path_len.mean()),
        "path_len_std": float(path_len.std()),
        "speed_mean_mean": float(speed_mean_per_point.mean()),
        "speed_mean_std": float(speed_mean_per_point.std()),
        "speed_std_mean": float(speed_std_per_point.mean()),
        "speed_std_std": float(speed_std_per_point.std()),
        "accel_mean_mean": float(accel_mean_per_point.mean()),
        "accel_mean_std": float(accel_mean_per_point.std()),
        "jerk_mean_mean": float(jerk_mean_per_point.mean()),
        "jerk_mean_std": float(jerk_mean_per_point.std()),
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "dom_ratio_mean": float(dom_ratio.mean()),
        "dom_ratio_std": float(dom_ratio.std()),
        "zero_speed_frac": zero_frac,
        "axis_corr_mean_abs": corr_mean,
    }

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--filelist", type=str, required=True)
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--task", type=str, required=True, help="Prefix for output files, e.g. stand/chair/sit/wheelchair/stand-frame")
    p.add_argument("--resample-n", type=int, default=100)
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--min-frames", type=int, default=10)
    return p

def main():
    args = build_argparser().parse_args()
    task = args.task.strip()

    filelist_path = Path(args.filelist)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in filelist_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    metas: List[TrialMeta] = []
    feat_rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for rel in lines:
        p = Path(rel)
        if not p.is_absolute():
            p = data_root / p
        if not p.exists():
            skipped.append((str(p), "missing"))
            continue
        try:
            trial = load_trial_txt(p)
            if trial.shape[0] < args.min_frames:
                skipped.append((str(p), f"too_few_frames(T={trial.shape[0]})"))
                continue
            pts = resample_time(to_points(trial), n=args.resample_n)
            feats = extract_features_from_pts(pts)
            sid, lab = parse_subject_and_label_from_filename(str(p))
            metas.append(TrialMeta(path=str(p), subject_id=sid, label=lab))
            feat_rows.append(feats)
        except Exception as e:
            skipped.append((str(p), f"error:{e}"))

    if not feat_rows:
        print("No trials were successfully processed.")
        print(f"Skipped: {len(skipped)} (showing first 10)")
        for p, why in skipped[:10]:
            print("  ", p, "->", why)
        return 0

    feat_names = sorted(feat_rows[0].keys())
    X = np.array([[fr[n] for n in feat_names] for fr in feat_rows], dtype=np.float64)
    Xz, med, scale = robust_center_scale(X)
    scores, mu, invC = mahalanobis_scores(Xz)

    ranks = scores.argsort().argsort().astype(np.float64)
    score01 = ranks / max(1.0, (len(scores) - 1))

    feat_csv = outdir / f"{task}_feature_matrix.csv"
    scores_csv = outdir / f"{task}_anomaly_scores.csv"
    top_txt = outdir / f"{task}_anomaly_topN.txt"
    model_json = outdir / f"{task}_anomaly_model.json"

    with feat_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(["path", "subject_id", "label"] + feat_names) + "\n")
        for meta, row in zip(metas, X):
            f.write(",".join([meta.path, meta.subject_id or "", meta.label or ""] + [f"{v:.10g}" for v in row]) + "\n")

    with scores_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(["path", "subject_id", "label", "mahalanobis_sq", "score01"]) + "\n")
        for meta, s, s01 in zip(metas, scores, score01):
            f.write(",".join([meta.path, meta.subject_id or "", meta.label or "", f"{float(s):.10g}", f"{float(s01):.10g}"]) + "\n")

    topn = min(args.topn, len(scores))
    top_idx = np.argsort(scores)[::-1][:topn]
    with top_txt.open("w", encoding="utf-8") as f:
        f.write(f"Top {topn} anomalies ({task}) by squared Mahalanobis distance\n")
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

    model_payload = {
        "task": task,
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

    print(f"✅ Task={task} | Processed trials: {len(scores)}")
    print(f"⚠️ Skipped trials:   {len(skipped)}")
    print(f"📁 Features:         {feat_csv}")
    print(f"📁 Scores:           {scores_csv}")
    print(f"📁 Top-N:            {top_txt}")
    print(f"📁 Model:            {model_json}")

    print("\nTop-10 anomalies:")
    for i in top_idx[: min(10, len(top_idx))]:
        meta = metas[i]
        print(f"  score={scores[i]:.3f} (s01={score01[i]:.3f}) | subj={meta.subject_id} | {meta.path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())