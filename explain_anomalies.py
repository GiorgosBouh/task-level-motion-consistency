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

        rows: List[Dict[str, Any]] = []
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
    out: List[Tuple[str, float, float]] = []
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
    lines: List[str] = []
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
    lines.append(
        "2) 3 likely causes (choose from:'sensor artifact, segmentation/cropping, wrong label, "
        "subject-specific movement strategy, file corruption/format issue)."
    )
    lines.append(
        "3) 2 concrete checks to validate (e.g., plot signals, verify filename label, check missing values, check duration)."
    )
    lines.append("Keep it concise, bullet points, no fluff.")
    return "\n".join(lines)


# =========================
# NEW: Cross-task consistency audit (SML)
# =========================

LABELS = {"task_invariant", "task_specific", "inconsistent"}


def safe_json_load(s: str) -> Dict[str, Any]:
    """Extract first JSON object from a model response."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(s[start : end + 1])


def make_consistency_prompt(subject_id: str, task_profiles: Dict[str, List[Tuple[str, float]]], topk: int = 3) -> str:
    """
    task_profiles: {task: [(family, score), ...]} sorted desc by score.
    Returns a STRICT-JSON instruction prompt for consistency labeling.
    """
    lines: List[str] = []
    lines.append("You are a cross-task consistency auditor for motion analysis.")
    lines.append("Label the SUBJECT as ONE of: task_invariant, task_specific, inconsistent.")
    lines.append("Do NOT invent data. Use ONLY the provided task summaries.")
    lines.append("Return STRICT JSON only. No markdown. No extra text.")
    lines.append("")
    lines.append(f"SUBJECT_ID: {subject_id}")
    lines.append("TASK_SUMMARIES (top families per task):")
    for task, fams in task_profiles.items():
        top = ", ".join([f"{fam}:{score:.3f}" for fam, score in fams[:topk]])
        lines.append(f"- {task}: {top}")
    lines.append("")
    lines.append("JSON schema:")
    lines.append("{")
    lines.append('  "subject_id": "<id>",')
    lines.append('  "label": "task_invariant|task_specific|inconsistent",')
    lines.append('  "supporting_tasks": ["<task1>", "<task2>"],')
    lines.append('  "dominant_families": ["<family1>", "<family2>"],')
    lines.append('  "confidence": 0.0')
    lines.append("}")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- task_invariant: similar dominant families repeat across multiple tasks.")
    lines.append("- task_specific: anomalies concentrated in one task or a clear subset of tasks.")
    lines.append("- inconsistent: dominant families differ across tasks without a coherent subset pattern.")
    return "\n".join(lines)


def run_consistency_audit(
    profiles_csv: str,
    out_csv: str,
    model: str,
    ollama_url: str,
    topk: int = 3,
    temperature: float = 0.0,
) -> None:
    """
    profiles_csv long format: subject_id,task,family,score
    Writes out_csv with labels + evidence.

    This is the "task consistency editor" role:
    - uses ONLY aggregated profiles
    - does NOT modify anomaly scores
    - outputs structured labels for cross-task behavior
    """
    by_subj_task: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    with open(profiles_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        req = {"subject_id", "task", "family", "score"}
        if not req.issubset(set(r.fieldnames or [])):
            raise ValueError(f"profiles_csv must contain columns: {sorted(req)}. Got: {r.fieldnames}")

        for row in r:
            sid = str(row["subject_id"]).strip()
            task = str(row["task"]).strip()
            fam = str(row["family"]).strip()
            score = float(row["score"])
            by_subj_task.setdefault(sid, {}).setdefault(task, []).append((fam, score))

    rows_out: List[Dict[str, Any]] = []
    for sid, task_map in by_subj_task.items():
        task_profiles = {t: sorted(v, key=lambda x: x[1], reverse=True) for t, v in task_map.items()}
        prompt = make_consistency_prompt(sid, task_profiles, topk=topk)

        try:
            resp = ollama_generate(
                url=ollama_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                num_predict=220,
            )
            obj = safe_json_load(resp)
            label = str(obj.get("label", "")).strip()
            if label not in LABELS:
                raise ValueError(f"Invalid label: {label}")

            rows_out.append(
                {
                    "subject_id": sid,
                    "label": label,
                    "confidence": obj.get("confidence", ""),
                    "supporting_tasks": json.dumps(obj.get("supporting_tasks", []), ensure_ascii=False),
                    "dominant_families": json.dumps(obj.get("dominant_families", []), ensure_ascii=False),
                    "error": "",
                }
            )

        except Exception as e:
            rows_out.append(
                {
                    "subject_id": sid,
                    "label": "ERROR",
                    "confidence": "",
                    "supporting_tasks": "[]",
                    "dominant_families": "[]",
                    "error": str(e),
                }
            )

    fieldnames = ["subject_id", "label", "confidence", "supporting_tasks", "dominant_families", "error"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"✅ Wrote: {out_csv}")


# =========================
# Existing functionality wrapped as subcommand
# =========================

def run_explain(args: argparse.Namespace) -> None:
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

    results: List[Dict[str, Any]] = []
    md_lines: List[str] = [f"# Anomaly explanations — task: {task}", ""]

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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1) existing functionality
    p1 = sub.add_parser("explain", help="Explain top-N anomalies within a single task")
    p1.add_argument("--task", required=True, help="e.g., stand, chair, wheelchair, sit, stand-frame")
    p1.add_argument("--outdir", default="outputs", help="where outputs/<task>_*.csv are")
    p1.add_argument("--model", default="phi3:mini", help="ollama model name")
    p1.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="ollama base url")
    p1.add_argument("--topn", type=int, default=20)
    p1.add_argument("--max-feats", type=int, default=8)
    p1.add_argument("--temperature", type=float, default=0.2)

    # 2) new functionality: cross-task consistency audit
    p2 = sub.add_parser("audit", help="Audit cross-task consistency per subject (SML)")
    p2.add_argument("--profiles-csv", required=True, help="CSV: subject_id,task,family,score (long format)")
    p2.add_argument("--out-csv", default="outputs/consistency_labels.csv")
    p2.add_argument("--model", default="phi3:mini")
    p2.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p2.add_argument("--topk", type=int, default=3)
    p2.add_argument("--temperature", type=float, default=0.0)

    args = ap.parse_args()

    if args.cmd == "explain":
        run_explain(args)
    elif args.cmd == "audit":
        run_consistency_audit(
            profiles_csv=args.profiles_csv,
            out_csv=args.out_csv,
            model=args.model,
            ollama_url=args.ollama_url,
            topk=args.topk,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()