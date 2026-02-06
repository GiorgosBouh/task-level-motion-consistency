#!/usr/bin/env python3
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
except Exception:
    butter = filtfilt = None

JOINTS = [
    "SpineBase","SpineMid","Neck","Head",
    "ShoulderLeft","ElbowLeft","WristLeft","HandLeft",
    "ShoulderRight","ElbowRight","WristRight","HandRight",
    "HipLeft","KneeLeft","AnkleLeft","FootLeft",
    "HipRight","KneeRight","AnkleRight","FootRight",
    "SpineShoulder","HandTipLeft","ThumbLeft","HandTipRight","ThumbRight"
]

# Use a stable trunk/COM proxy for global motion
COM_JOINTS = ["SpineBase","SpineMid","SpineShoulder","HipLeft","HipRight","ShoulderLeft","ShoulderRight"]

def parse_meta_from_filename(fp: str):
    """
    Expected:
      <subject>_<date>_<gesture>_<rep>_<correctness>_<posture>.txt
    Example:
      307_18_8_9_1_stand.txt
      101_18_4_11_2_stand-frame.txt  (posture can include hyphen)
    """
    base = os.path.basename(fp)
    stem = base[:-4] if base.lower().endswith(".txt") else base
    parts = stem.split("_")
    if len(parts) < 6:
        # fallback
        return dict(subject_id=None, date_id=None, gesture=None, repetition=None, correctness=None, posture=None)

    subject_id = int(parts[0]) if parts[0].isdigit() else None
    date_id = parts[1]
    gesture = parts[2]
    repetition = parts[3]
    correctness = parts[4]
    posture = "_".join(parts[5:])  # keep hyphens etc. (e.g., stand-frame)
    return dict(
        subject_id=subject_id,
        date_id=str(date_id),
        gesture=str(gesture),
        repetition=str(repetition),
        correctness=str(correctness),
        posture=str(posture),
    )

def _butter_lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4):
    if butter is None or filtfilt is None:
        raise RuntimeError("scipy is required for Butterworth filtering but not available.")
    nyq = 0.5 * fs
    w = cutoff_hz / nyq
    b, a = butter(order, w, btype="low")
    return filtfilt(b, a, x, axis=0)

def parse_kinect_file(fp: str):
    """
    Returns:
      t: (T,) timestamps if present else None
      xyz: dict joint -> (T,3) float
    """
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        if not header.lower().startswith("version"):
            # sometimes files may not have the header; rewind
            f.seek(0)

        frames = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(line)

    if len(frames) < 5:
        return None, None

    # each frame: frame_id, 5, timestamp, (Joint,Tracked,x,y,z,px,py)...
    xyz = {j: [] for j in JOINTS}
    ts = []

    joint_pat = re.compile(r"\(([A-Za-z0-9_]+),Tracked,([^,]+),([^,]+),([^,]+),")
    for line in frames:
        # split first 3 comma fields
        # safer: take first 3 tokens before first "(..."
        head = line.split("(", 1)[0].strip().strip(",")
        head_parts = head.split(",")
        if len(head_parts) >= 3:
            try:
                ts.append(float(head_parts[2]))
            except Exception:
                ts.append(np.nan)
        else:
            ts.append(np.nan)

        matches = joint_pat.findall(line)
        if not matches:
            continue
        # map joints in this line
        found = {}
        for name, xs, ys, zs in matches:
            if name in xyz:
                try:
                    found[name] = (float(xs), float(ys), float(zs))
                except Exception:
                    found[name] = (np.nan, np.nan, np.nan)

        # append in JOINT order; if missing -> nan
        for j in JOINTS:
            if j in found:
                xyz[j].append(found[j])
            else:
                xyz[j].append((np.nan, np.nan, np.nan))

    # to arrays
    for j in JOINTS:
        xyz[j] = np.asarray(xyz[j], dtype=float)

    ts = np.asarray(ts, dtype=float)
    if np.all(np.isnan(ts)):
        ts = None
    return ts, xyz

def compute_features_from_xyz(xyz: dict, fs: float, do_filter: str, cutoff_hz: float):
    """
    Compute global motion features on COM proxy:
      speed, accel, jerk magnitudes
      smoothness proxy = 1 / (1 + jerk_rms)  (simple, monotonic, no jargon)
    Aggregate with mean/std/range for each.
    """
    # Build COM proxy
    pts = []
    for j in COM_JOINTS:
        if j in xyz:
            pts.append(xyz[j])
    if not pts:
        return None
    com = np.nanmean(np.stack(pts, axis=0), axis=0)  # (T,3)

    # handle NaNs by linear interpolation per dim (simple and deterministic)
    for d in range(3):
        v = com[:, d]
        n = len(v)
        idx = np.where(~np.isnan(v))[0]
        if len(idx) < max(5, n // 5):
            # too many NaNs
            return None
        com[:, d] = np.interp(np.arange(n), idx, v[idx])

    if do_filter == "butter":
        com = _butter_lowpass_filter(com, fs=fs, cutoff_hz=cutoff_hz, order=4)

    dt = 1.0 / fs
    vel = np.gradient(com, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    speed = np.linalg.norm(vel, axis=1)
    accel = np.linalg.norm(acc, axis=1)
    jerk_m = np.linalg.norm(jerk, axis=1)

    # smoothness proxy (simple, no fancy definitions)
    jerk_rms = float(np.sqrt(np.mean(jerk_m**2)))
    smoothness = 1.0 / (1.0 + jerk_rms)

    def agg(x):
        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x, ddof=1)) if len(x) > 2 else float(np.std(x)),
            range=float(np.max(x) - np.min(x)),
        )

    out = {}
    for name, arr in [("speed", speed), ("accel", accel), ("jerk", jerk_m)]:
        a = agg(arr)
        out[f"{name}_mean"] = a["mean"]
        out[f"{name}_std"] = a["std"]
        out[f"{name}_range"] = a["range"]

    # smoothness is scalar in this proxy; also add a "std/range" as 0 for schema stability
    out["smoothness_mean"] = float(smoothness)
    out["smoothness_std"] = 0.0
    out["smoothness_range"] = 0.0

    out["n_frames"] = int(len(com))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True, help='e.g. "data/RawData/*.txt"')
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--filter", choices=["none","butter"], default="none")
    ap.add_argument("--cutoff-hz", type=float, default=6.0, help="Butterworth cutoff (Hz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")

    rows = []
    bad = 0
    for fp in files:
        meta = parse_meta_from_filename(fp)
        ts, xyz = parse_kinect_file(fp)
        if xyz is None:
            bad += 1
            continue
        feats = compute_features_from_xyz(xyz, fs=args.fs, do_filter=args.filter, cutoff_hz=args.cutoff_hz)
        if feats is None:
            bad += 1
            continue
        row = {"path": fp, **meta, **feats}
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} | rows={len(df)} | bad={bad} | cols={len(df.columns)}")
    print("columns:", list(df.columns))

if __name__ == "__main__":
    main()
