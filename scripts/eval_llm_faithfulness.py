#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

FAMILIES = ["speed","accel","jerk","smoothness"]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def support_label_from_evidence(pct: float, contrib: float) -> str:
    # Must match prompt rules
    if pct >= 0.90 or contrib >= 0.70:
        return "high"
    if (0.75 <= pct < 0.90) or (0.45 <= contrib < 0.70):
        return "medium"
    return "low"

def should_abstain(score01_pct: float, has_high_family: bool) -> bool:
    return (score01_pct < 0.90) and (not has_high_family)

def normalize_model_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Enforce expected structure; missing keys -> safe defaults
    path = str(obj.get("path",""))
    task = str(obj.get("task",""))
    sev = obj.get("severity", {}) or {}
    score01_pct = float(sev.get("score01_pct", np.nan))
    mahal2_pct  = float(sev.get("mahal2_pct", np.nan))
    abstain = bool(obj.get("abstain", False))

    fam_list = obj.get("supported_families", []) or []
    fam_map = {f: {"support":"low"} for f in FAMILIES}
    for it in fam_list:
        fam = str(it.get("family",""))
        if fam in fam_map:
            fam_map[fam] = {
                "support": str(it.get("support","low")).lower(),
                "pct": float((it.get("evidence") or {}).get("pct", np.nan)),
                "contribution": float((it.get("evidence") or {}).get("contribution", np.nan)),
            }
    return {
        "path": path, "task": task,
        "score01_pct": score01_pct, "mahal2_pct": mahal2_pct,
        "abstain": abstain,
        "fam_support": {f: fam_map[f]["support"] for f in FAMILIES},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-csv", required=True)
    ap.add_argument("--llm-jsonl", required=True, help="JSONL with STRICT outputs (one per trial).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summ = pd.read_csv(args.summaries_csv)
    summ = summ.dropna(subset=["path","task","score01_pct","mahal2_pct"]).copy()
    summ["path"] = summ["path"].astype(str)

    # Build gold labels per family per trial from evidence
    gold = summ[["path","task","score01_pct","mahal2_pct"]].copy()
    for f in FAMILIES:
        gold[f"{f}_gold"] = [
            support_label_from_evidence(pct=float(p), contrib=float(c))
            for p, c in zip(summ[f"{f}_pct"], summ[f"{f}_c"])
        ]
    gold["has_high_family"] = gold[[f"{f}_gold" for f in FAMILIES]].apply(lambda r: any(x=="high" for x in r), axis=1)
    gold["abstain_gold"] = [
        should_abstain(score01_pct=float(s), has_high_family=bool(h))
        for s, h in zip(gold["score01_pct"], gold["has_high_family"])
    ]

    # Load model outputs
    outs = load_jsonl(args.llm_jsonl)
    norm = [normalize_model_output(o) for o in outs]
    out_df = pd.DataFrame(norm)

    # Merge by path (unique id)
    m = gold.merge(out_df, on="path", how="left", suffixes=("","_model"))

    # Basic coverage
    m["has_output"] = m["task_model"].notna()
    coverage = float(m["has_output"].mean())

    # Score per family: treat "high" as positive class (objective)
    rows = []
    for f in FAMILIES:
        y_true = (m[f"{f}_gold"] == "high")
        y_pred = (m["fam_support"].apply(lambda d: (isinstance(d, dict) and d.get(f,"low")=="high")))
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        prec = tp / (tp + fp) if (tp+fp) else np.nan
        rec  = tp / (tp + fn) if (tp+fn) else np.nan
        rows.append({"family": f, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec})

    fam_metrics = pd.DataFrame(rows)

    # Abstain metrics
    abstain_pred = m["abstain"].fillna(False).astype(bool)
    abstain_true = m["abstain_gold"].astype(bool)
    abstain_acc = float((abstain_pred == abstain_true).mean())

    # Overclaim rate: any family predicted high when gold is not high
    def overclaim(row):
        if not isinstance(row.get("fam_support"), dict):
            return False
        for f in FAMILIES:
            if row["fam_support"].get(f,"low")=="high" and row[f"{f}_gold"]!="high":
                return True
        return False
    m["overclaim_any"] = m.apply(overclaim, axis=1)
    overclaim_rate = float(m["overclaim_any"].mean())

    # Save detailed report
    report_cols = ["path","task","score01_pct","mahal2_pct","abstain_gold","abstain","has_output","overclaim_any"]
    for f in FAMILIES:
        report_cols += [f"{f}_gold"]
    m_out = m[report_cols].copy()
    out_csv = os.path.join(args.out_dir, "layerA_faithfulness_report.csv")
    m_out.to_csv(out_csv, index=False)

    # Save metrics json
    metrics = {
        "n_trials": int(len(m)),
        "coverage": coverage,
        "abstain_accuracy": abstain_acc,
        "overclaim_rate_any": overclaim_rate,
        "family_metrics": fam_metrics.to_dict(orient="records"),
    }
    out_json = os.path.join(args.out_dir, "layerA_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # LaTeX table (family precision/recall)
    tex_path = os.path.join(args.out_dir, "table_layerA_family_metrics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Layer A faithfulness metrics for high-support family claims (objective, evidence-based).}\n")
        f.write("\\label{tab:layerA_family_metrics}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Family & TP & FP & Precision & Recall \\\\\n\\hline\n")
        for _, r in fam_metrics.iterrows():
            prec = "-" if pd.isna(r["precision"]) else f"{r['precision']:.3f}"
            rec  = "-" if pd.isna(r["recall"]) else f"{r['recall']:.3f}"
            f.write(f"{r['family']} & {int(r['tp'])} & {int(r['fp'])} & {prec} & {rec} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    print("✅ wrote:", out_csv)
    print("✅ wrote:", out_json)
    print("✅ wrote:", tex_path)
    print(f"coverage={coverage:.3f} | abstain_acc={abstain_acc:.3f} | overclaim_rate_any={overclaim_rate:.3f}")

if __name__ == "__main__":
    main()
