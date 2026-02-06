#!/usr/bin/env python3
import argparse
import os
import re
import glob
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
except Exception:
    butter = filtfilt = None


def parse_meta_from_path(p: str):
    """
    Expected filename:
      SubjectID_DateID_GestureLabel_Repetition_Correctness_Posture.txt
    Example:
      data/202_18_3_2_3_stand.txt
    Returns: subject_id(int), date_id(str), gesture(str), rep(str), correctness(str), posture(str)
    """
    base = os.path.basename(p)
    m = re.match(r"(\d+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)\.txt$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {base}")
    sid, date_id, gesture, rep, corr, posture = m.groups()
    return int(sid), date_id, gesture, rep, corr, posture


def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int):
    if butter is None or filtfilt is None:
        raise RuntimeError("scipy is required for Butterworth filtering. Install scipy.")
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    # filtfilt along time axis (axis=0)
    return filtfilt(b, a, x, axis=0)


def finite_diff(x: np.ndarray, dt: float):
    # x: [T, D]
    # central diff for interior, forward/backward for ends via np.gradient
    return np.gradient(x, dt, axis=0)


def load_skeleton_txt(path: str) -> np.ndarray:
    """
    Load IRDS Kinect skeleton file.
    Assumption: each row is a frame; columns are 75 values (25 joints * xyz).
    If your files contain extra columns (timestamps etc.), adapt here.
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Keep only first 75 columns if more exist
    if arr.shape[1] < 75:
        raise ValueError(f"{path}: expected >=75 columns, got {arr.shape[1]}")
    arr = arr[:, :75]
    return arr


def aggregate_magnitude(vec: np.ndarray) -> np.ndarray:
    """
    vec: [T, 75] where xyz triplets per joint.
    Convert to per-joint magnitude then aggregate across joints -> [T]
    """
    T = vec.shape[0]
    xyz = vec.reshape(T, 25, 3)
    mag = np.linalg.norm(xyz, axis=2)  # [T, 25]
    return mag.mean(axis=1)            # [T]


def summarize_signal(sig: np.ndarray, prefix: str):
    # returns dict of mean/std/range for scalar signal
    sig = np.asarray(sig)
    return {
        f"{prefix}_mean": float(np.mean(sig)),
        f"{prefix}_std": float(np.std(sig, ddof=0)),
        f"{prefix}_range": float(np.max(sig) - np.min(sig)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default="data/*.txt", help="Glob for IRDS trial files")
    ap.add_argument("--fs", type=float, default=30.0, help="Sampling rate (Hz)")
    ap.add_argument("--cutoff", type=float, default=6.0, help="Butterworth cutoff (Hz)")
    ap.add_argument("--order", type=int, default=4, help="Butterworth order")
    ap.add_argument("--no-filter", action="store_true", help="Disable filtering (for sensitivity analysis)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No files matched {args.data_glob}")

    rows = []
    dt = 1.0 / args.fs

    for p in paths:
        sid, date_id, gesture, rep, corr, posture = parse_meta_from_path(p)

        X = load_skeleton_txt(p)  # [T,75]

        if (not args.no_filter) and args.cutoff > 0:
            Xf = butter_lowpass_filter(X, fs=args.fs, cutoff_hz=args.cutoff, order=args.order)
        else:
            Xf = X

        # Represent movement magnitude (simple, robust) and compute derivatives
        pos = aggregate_magnitude(Xf)                    # [T]
        vel = finite_diff(pos[:, None], dt)[:, 0]        # [T]
        acc = finite_diff(vel[:, None], dt)[:, 0]        # [T]
        jerk = finite_diff(acc[:, None], dt)[:, 0]       # [T]

        feat = {}
        feat.update(summarize_signal(np.abs(vel), "speed"))
        feat.update(summarize_signal(np.abs(acc), "accel"))
        feat.update(summarize_signal(np.abs(jerk), "jerk"))
        # Smoothness proxy (consistent with Kinect pipeline): 1 / (1 + RMS(jerk))
        jerk_rms = float(np.sqrt(np.mean(jerk**2)))    
        smoothness = 1.0 / (1.0 + jerk_rms)
        feat["smoothness_mean"] = float(smoothness)
        feat["smoothness_std"] = 0.0
        feat["smoothness_range"] = 0.0       
        rows.append({
            "path": p,
            "subject_id": sid,
            "posture": posture,
            "gesture": gesture,
            "context": f"{posture}::{gesture}",  # IMPORTANT for cross-context
            "correctness": corr,
            **feat
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {args.out} rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
