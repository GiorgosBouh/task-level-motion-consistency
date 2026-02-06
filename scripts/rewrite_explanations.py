import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

JSONL_FILES = [
    "outputs_unfiltered/llm_outputs_highconf.jsonl",
    "outputs_unfiltered/llm_outputs_highconf_remaining.jsonl",
]

TASKS = ["stand","chair","wheelchair","sit","stand-frame"]

def load_llm(jsonl_files):
    by_path = {}
    skipped = defaultdict(int)

    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    skipped["bad_json_parse"] += 1
                    continue
                if not isinstance(o, dict):
                    skipped["non_dict"] += 1
                    continue
                if o.get("error"):
                    skipped["error_rows"] += 1
                    continue
                p = o.get("path")
                if not p:
                    skipped["missing_path"] += 1
                    continue
                by_path[p] = o
    return by_path, dict(skipped)

def fam_pretty(f):
    # normalize some LLM variants -> canonical-ish display
    if not f: return "unknown"
    f = str(f).strip().lower()
    mapping = {
        "acceleration|deceleration": "accel/decel",
        "accel|decel": "accel/decel",
        "jerk (derivative of acceleration)": "jerk",
    }
    return mapping.get(f, f)

def support_phrase(s):
    s = str(s).lower().strip()
    if s == "high": return "strong"
    if s == "medium": return "moderate"
    return "weak"

def make_text(o):
    sev = o.get("severity", {})
    score_pct = sev.get("score01_pct", None)

    dom = o.get("dominant_family", {})
    dom_f = fam_pretty(dom.get("family"))
    dom_c = dom.get("contribution", None)
    dom_p = dom.get("pct", None)

    fams = o.get("supported_families", []) or []
    # sort by contribution desc if exists
    def key(sf):
        ev = sf.get("evidence", {}) if isinstance(sf, dict) else {}
        return float(ev.get("contribution", -1) or -1)
    fams_sorted = sorted([sf for sf in fams if isinstance(sf, dict)], key=key, reverse=True)

    # Build a non-templated but consistent sentence structure
    parts = []
    if score_pct is not None:
        parts.append(f"Anomaly percentile={score_pct:.3f}.")

    if dom_f and dom_f != "unknown":
        dom_bits = [f"dominant={dom_f}"]
        if dom_c is not None: dom_bits.append(f"mass={dom_c:.2f}")
        if dom_p is not None: dom_bits.append(f"pct={dom_p:.3f}")
        parts.append(" ".join(dom_bits) + ".")

    # add up to 3 supported families with evidence
    if fams_sorted:
        supp_bits = []
        for sf in fams_sorted[:3]:
            fam = fam_pretty(sf.get("family"))
            sup = support_phrase(sf.get("support"))
            ev  = sf.get("evidence", {}) if isinstance(sf.get("evidence"), dict) else {}
            pct = ev.get("pct", None)
            con = ev.get("contribution", None)
            b = f"{fam}:{sup}"
            if con is not None: b += f" (c={float(con):.2f}"
            else: b += " ("
            if pct is not None: b += f", p={float(pct):.3f})"
            else: b += ")"
            supp_bits.append(b)
        parts.append("Support → " + "; ".join(supp_bits) + ".")

    note = o.get("notes")
    if isinstance(note, str) and note.strip():
        # keep it short, but keep it (it is evidence-grounded)
        parts.append(note.strip())

    # final cleanup
    txt = " ".join(parts).strip()
    return txt

def main():
    by_path, skipped = load_llm(JSONL_FILES)
    print("LLM rows loaded:", len(by_path))
    print("skipped:", skipped)

    for task in TASKS:
        scores_p = Path(f"outputs/{task}_anomaly_scores.csv")
        expl_p   = Path(f"outputs/{task}_anomaly_explanations.csv")

        if not scores_p.exists():
            print("missing:", scores_p)
            continue

        scores = pd.read_csv(scores_p)
        if "path" not in scores.columns:
            print("no path col in:", scores_p)
            continue

        # build explanations for all paths that exist in LLM map
        out_rows = []
        for _, r in scores.iterrows():
            p = str(r["path"])
            o = by_path.get(p)
            if not o:
                continue
            out_rows.append({
                "rank": int(r.get("rank", 0)) if "rank" in scores.columns else None,
                "path": p,
                "subject_id": r.get("subject_id", ""),
                "label": r.get("label", ""),
                "explanation": make_text(o),
            })

        df = pd.DataFrame(out_rows)
        # keep only columns that viewer expects (rank optional)
        cols = ["rank","path","subject_id","label","explanation"]
        df = df[[c for c in cols if c in df.columns]]

        # backup once
        if expl_p.exists():
            bak = expl_p.with_suffix(".csv.bak2")
            if not bak.exists():
                expl_p.replace(bak)

        df.to_csv(expl_p, index=False)
        print(f"WROTE {expl_p} rows={len(df)}")

if __name__ == "__main__":
    main()
