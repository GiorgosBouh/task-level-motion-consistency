#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def ollama_generate(url: str, model: str, prompt: str, temperature: float = 0.0, num_predict: int = 220) -> str:
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

LABELS = {"task_invariant", "task_specific", "inconsistent"}

def make_prompt(subject_id: str, task_profiles: Dict[str, List[Tuple[str, float]]], topk: int = 3) -> str:
    """
    task_profiles: {task: [(family, score), ...]} sorted desc by score
    """
    lines = []
    lines.append("You are a cross-task consistency auditor for motion analysis.")
    lines.append("Your job: label a SUBJECT as one of: task_invariant, task_specific, inconsistent.")
    lines.append("You MUST NOT invent any information beyond the provided summaries.")
    lines.append("Return STRICT JSON only, no markdown, no extra text.")
    lines.append("")
    lines.append(f"SUBJECT_ID: {subject_id}")
    lines.append("TASK_SUMMARIES (top families per task):")
    for task, fams in task_profiles.items():
        top = ", ".join([f"{fam}:{score:.3f}" for fam, score in fams[:topk]])
        lines.append(f"- {task}: {top}")
    lines.append("")
    lines.append("JSON schema:")
    lines.append('{')
    lines.append('  "subject_id": "<id>",')
    lines.append('  "label": "task_invariant|task_specific|inconsistent",')
    lines.append('  "supporting_tasks": ["<task1>", "<task2>"],')
    lines.append('  "dominant_families": ["<family1>", "<family2>"],')
    lines.append('  "confidence": 0.0')
    lines.append('}')
    lines.append("")
    lines.append("Definitions:")
    lines.append("- task_invariant: similar dominant families repeat across multiple tasks.")
    lines.append("- task_specific: anomalies/families concentrated in one task or one clear subset of tasks.")
    lines.append("- inconsistent: dominant families differ across tasks without a coherent subset pattern.")
    return "\n".join(lines)

def safe_json_load(s: str) -> Dict[str, Any]:
    # tries to extract first JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(s[start:end+1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-csv", required=True, help="CSV: subject_id,task,family,score (long format)")
    ap.add_argument("--out-csv", default="consistency_labels.csv")
    ap.add_argument("--model", default="phi3:mini")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # read profiles
    by_subj_task = defaultdict(lambda: defaultdict(list))  # subj -> task -> list[(family, score)]
    with open(args.profiles_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row["subject_id"]).strip()
            task = str(row["task"]).strip()
            fam = str(row["family"]).strip()
            score = float(row["score"])
            by_subj_task[sid][task].append((fam, score))

    results = []
    for sid, task_map in by_subj_task.items():
        # sort families per task
        task_profiles = {}
        for task, fams in task_map.items():
            task_profiles[task] = sorted(fams, key=lambda x: x[1], reverse=True)

        prompt = make_prompt(sid, task_profiles, topk=args.topk)

        try:
            resp = ollama_generate(args.ollama_url, args.model, prompt, temperature=args.temperature, num_predict=220)
            obj = safe_json_load(resp)
            label = obj.get("label", "").strip()
            if label not in LABELS:
                raise ValueError(f"Invalid label: {label}")
            results.append({
                "subject_id": sid,
                "label": label,
                "confidence": obj.get("confidence", ""),
                "supporting_tasks": json.dumps(obj.get("supporting_tasks", []), ensure_ascii=False),
                "dominant_families": json.dumps(obj.get("dominant_families", []), ensure_ascii=False),
            })
        except Exception as e:
            results.append({
                "subject_id": sid,
                "label": "ERROR",
                "confidence": "",
                "supporting_tasks": "[]",
                "dominant_families": "[]",
                "error": str(e),
            })

    # write output
    fieldnames = ["subject_id", "label", "confidence", "supporting_tasks", "dominant_families", "error"]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            if "error" not in row:
                row["error"] = ""
            w.writerow(row)

    print(f"✅ Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()