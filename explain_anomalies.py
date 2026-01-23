#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def read_topN_paths(topn_txt: str, topn: int) -> List[str]:
    """
    Parses the repository's topN text like:
    rank,mahalanobis_sq,score01,subject_id,label,path
    1,409.53,1,202,stand,data/202_..._stand.txt
    Returns the 'path' column for first topn rows.
    """
    paths: List[str] = []
    with open(topn_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # find header line that contains ",path"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().endswith(",path") or ("," in ln and "path" in ln.lower().split(",")):
            header_idx = i
            break
    if header_idx is None:
        # fallback: try to extract "data/..." occurrences
        for ln in lines:
            m = re.search(r"(data/[^,\s]+\.txt)", ln)
            if m:
                paths.append(m.group(1))
            if len(paths) >= topn:
                break
        return paths

    header = [h.strip().lower() for h in lines[header_idx].split(",")]
    try:
        path_col = header.index("path")
    except ValueError:
        path_col = len(header) - 1

    for ln in lines[header_idx + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) <= path_col:
            continue
        paths.append(parts[path_col])
        if len(paths) >= topn:
            break
    return paths


def load_feature_matrix_csv(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Loads outputs/<task>_feature_matrix.csv

    Expected columns (at minimum):
      path,subject_id,label,<feature1>,<feature2>,...
    Returns (feature_names, rows) where rows are dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "path" not in fieldnames:
            raise ValueError(f"Feature matrix CSV missing 'path' column: {path}")

        meta_cols = {"path", "subject_id", "label"}
        feature_names = [c for c in fieldnames if c not in meta_cols]

        rows = []
        for r in reader:
            rows.append(r)
    return feature_names, rows


def to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return float("nan")
    return float(s)


def robust_baseline(feature_names: List[str], rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline = median, scale = IQR (p75-p25). If IQR==0, fallback to std or 1.
    """
    X = np.array([[to_float(r.get(fn)) for fn in feature_names] for r in rows], dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    p25 = np.nanpercentile(X, 25, axis=0)
    p75 = np.nanpercentile(X, 75, axis=0)
    iqr = p75 - p25

    # fallback: use std where iqr is zero
    std = np.nanstd(X, axis=0)
    scale = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.0))
    return med, scale


def find_row_by_path(rows: List[Dict[str, Any]], target_path: str) -> Optional[Dict[str, Any]]:
    # try exact match first
    for r in rows:
        if r.get("path") == target_path:
            return r
    # fallback: normalize
    t = target_path.replace("./", "")
    for r in rows:
        p = (r.get("path") or "").replace("./", "")
        if p == t:
            return r
    return None


def build_feature_diff_summary(
    feature_names: List[str],
    row: Dict[str, Any],
    med: np.ndarray,
    scale: np.ndarray,
    max_feats: int = 8,
) -> List[Tuple[str, float, float]]:
    """
    Returns list of (feature_name, value, robust_z) sorted by |robust_z| desc
    robust_z = (x - median)/scale
    """
    x = np.array([to_float(row.get(fn)) for fn in feature_names], dtype=np.float64)
    rz = (x - med) / scale
    order = np.argsort(-np.abs(rz))
    out = []
    for idx in order[:max_feats]:
        out.append((feature_names[idx], float(x[idx]), float(rz[idx])))
    return out


def ollama_generate(
    url: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    num_predict: int = 220,
) -> str:
    """
    Calls Ollama /api/generate (non-stream).
    """
    import urllib.request

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "stop": ["\n\n\n", "\u0000"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    obj = json.loads(raw)
    return (obj.get("response") or "").strip()


def make_prompt(task: str, meta: Dict[str, Any], diffs: List[Tuple[str, float, float]]) -> str:
    """
    Minimal, paper-safe, no hallucination: ask for plausible QC reasons and checks.
    """
    lines = []
    lines.append("You are assisting with motion time-series quality control.")
    lines.append("We detected an anomalous trial within a single task using a multivariate distance score.")
    lines.append("Do NOT invent dataset details. Only use what is provided.")
    lines.append("")
    lines.append(f"Task: {task}")
    lines.append(f"File: {meta.get('path')}")
    lines.append(f"Subject: {meta.get('subject_id')}")
    lines.append("")
    lines.append("Top deviating features (value, robust_z):")
    for fn, val, rz in diffs:
        lines.append(f"- {fn}: {val:.6g} (z={rz:.2f})")
    lines.append("")
    lines.append("Write:")
    lines.append("1) A one-sentence interpretation of what this suggests (generic, conservative).")
    lines.append("2) 3 likely causes (choose from: sensor artifact, segmentation/cropping, wrong label, subject-specific movement strategy, file corruption/format issue).")
    lines.append("3) 2 concrete checks to validate (e.g., plot signals, verify filename label, check missing values, check duration).")
    lines.append("Keep it concise, bullet points, no fluff.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g., stand, chair, wheelchair, sit, stand-frame")
    ap.add_argument("--outdir", default="outputs", help="where outputs/<task>_*.csv are")
    ap.add_argument("--model", default="phi3:mini", help="ollama model name")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="ollama base url")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--max-feats", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    outdir = args.outdir
    task = args.task

    topn_txt = os.path.join(outdir, f"{task}_anomaly_topN.txt")
    feat_csv = os.path.join(outdir, f"{task}_feature_matrix.csv")

    if not os.path.exists(topn_txt):
        print(f"ERROR: missing {topn_txt}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(feat_csv):
        print(f"ERROR: missing {feat_csv}", file=sys.stderr)
        sys.exit(2)

    paths = read_topN_paths(topn_txt, args.topn)
    if not paths:
        print("ERROR: could not parse any paths from topN file.", file=sys.stderr)
        sys.exit(2)

    feature_names, rows = load_feature_matrix_csv(feat_csv)
    med, scale = robust_baseline(feature_names, rows)

    out_csv = os.path.join(outdir, f"{task}_anomaly_explanations.csv")
    out_md = os.path.join(outdir, f"{task}_anomaly_explanations.md")

    results = []
    md_lines = [f"# Anomaly explanations — task: {task}", ""]

    for rank, p in enumerate(paths, start=1):
        r = find_row_by_path(rows, p)
        if r is None:
            txt = f"[rank {rank}] {p} — row not found in feature matrix"
            results.append({"rank": rank, "path": p, "subject_id": "", "label": task, "explanation": txt})
            md_lines.append(f"## Rank {rank}")
            md_lines.append(f"- **File:** {p}")
            md_lines.append(f"- **Note:** row not found in feature matrix")
            md_lines.append("")
            continue

        diffs = build_feature_diff_summary(feature_names, r, med, scale, max_feats=args.max_feats)
        prompt = make_prompt(task, r, diffs)

        try:
            explanation = ollama_generate(
                url=args.ollama_url,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                num_predict=260,
            )
        except Exception as e:
            explanation = f"OLLAMA_ERROR: {e}"

        results.append(
            {
                "rank": rank,
                "path": r.get("path", p),
                "subject_id": r.get("subject_id", ""),
                "label": r.get("label", task),
                "explanation": explanation,
            }
        )

        md_lines.append(f"## Rank {rank}")
        md_lines.append(f"- **File:** {r.get('path', p)}")
        md_lines.append(f"- **Subject:** {r.get('subject_id','')}")
        md_lines.append("")
        md_lines.append(explanation)
        md_lines.append("")

    # write CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "path", "subject_id", "label", "explanation"])
        w.writeheader()
        for row in results:
            w.writerow(row)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).rstrip() + "\n")

    print(f"✅ Wrote: {out_csv}")
    print(f"✅ Wrote: {out_md}")


if __name__ == "__main__":
    main()