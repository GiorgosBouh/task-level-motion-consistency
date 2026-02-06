#!/usr/bin/env python3
import argparse
import json
import subprocess
from typing import Any, Dict, Optional

def extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON from model output.
    - Try full parse.
    - Else slice first {...} or [...] region and try again.
    """
    t = (text or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try object slice
    i1 = t.find("{")
    i2 = t.rfind("}")
    if i1 != -1 and i2 != -1 and i2 > i1:
        try:
            return json.loads(t[i1:i2+1])
        except Exception:
            pass

    # Try array slice
    j1 = t.find("[")
    j2 = t.rfind("]")
    if j1 != -1 and j2 != -1 and j2 > j1:
        try:
            return json.loads(t[j1:j2+1])
        except Exception:
            pass

    return None

def ollama_run(model: str, prompt: str, timeout_s: int = 180) -> Dict[str, Any]:
    """
    IMPORTANT: `ollama run` expects the prompt as a CLI arg (or interactive).
    Passing prompt via stdin can hang. So we pass prompt as the last arg.
    """
    cmd = ["ollama", "run", model, "--format", "json", prompt]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"error": "ollama_timeout", "raw_err": f"timeout>{timeout_s}s"}
    except Exception as e:
        return {"error": "ollama_exec_failed", "raw_err": str(e)}

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()

    if p.returncode != 0:
        return {"error": "ollama_failed", "raw_err": (err or out)[:2000]}

    j = extract_json(out)
    if j is None:
        return {"error": "bad_json", "raw_err": out[:2000]}

    return j

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    n_in = 0
    ok = 0
    bad_json = 0
    fail = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            if args.limit and (ok + bad_json + fail) >= args.limit:
                break

            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            meta_path = obj.get("path", obj.get("evidence", {}).get("path", ""))
            meta_task = obj.get("task", obj.get("evidence", {}).get("task", ""))

            out = ollama_run(args.model, prompt, timeout_s=args.timeout)

            # attach identifiers for tracking
            if isinstance(out, dict):
                out.setdefault("path", meta_path)
                out.setdefault("task", meta_task)

            if isinstance(out, dict) and out.get("error") == "bad_json":
                bad_json += 1
            elif isinstance(out, dict) and "error" in out:
                fail += 1
            else:
                ok += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"IN: {n_in} | OK: {ok} | bad_json: {bad_json} | fail: {fail}")
    print(f"WROTE: {args.out_jsonl}")

if __name__ == "__main__":
    main()
