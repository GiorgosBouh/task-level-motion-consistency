1) Project structure (paths you must know)

Inputs
	•	Raw data directory (not included in Git, ignored via .gitignore):
	•	data/RawData/  (example)
	•	The scripts expect trial files in a consistent format (see your existing dataset export).

Outputs (paper artifacts are generated here)
	•	outputs/ (generated; not committed)
	•	Feature matrices per task:
	•	outputs/stand_feature_matrix.csv
	•	outputs/chair_feature_matrix.csv
	•	outputs/sit_feature_matrix.csv
	•	outputs/wheelchair_feature_matrix.csv
	•	outputs/stand-frame_feature_matrix.csv
	•	Anomaly scoring per task:
	•	outputs/stand_anomaly_scores.csv
	•	outputs/chair_anomaly_scores.csv
	•	outputs/sit_anomaly_scores.csv
	•	outputs/wheelchair_anomaly_scores.csv
	•	outputs/stand-frame_anomaly_scores.csv
	•	Top-N tables (used in paper tables/appendix):
	•	outputs/*_anomaly_topN.txt
	•	Cross-task family profiles (used for cosine similarity analysis):
	•	outputs/_profiles_main.csv (recommended canonical)
	•	outputs/cross_task_profiles.csv (older / may contain duplicates depending on past runs)
	•	Cross-task null tests:
	•	outputs/cosine_nulls_<taskA>_<taskB>_summary.csv
	•	outputs/cosine_nulls_<taskA>_<taskB>_same_subject.csv
	•	outputs/cosine_nulls_<taskA>_<taskB>_diff_subject.csv
	•	outputs/cosine_nulls_<taskA>_<taskB>_perm_null.csv

⸻

2) What each main script does

Root-level pipeline
	•	main.py
Orchestrates the end-to-end run (depending on your configuration). Use it if you want a single entry point.
	•	anomaly_task.py
Task-generic anomaly scoring pipeline (reads a feature matrix, fits Ledoit–Wolf covariance, computes Mahalanobis², ranks trials, writes *_anomaly_scores.csv and *_anomaly_topN.txt).
	•	anomaly_stand.py
Specialization for stand / stand-frame variants (if your dataset splits stand contexts).
	•	build_cross_task_profiles.py
Builds subject-level family-composition profiles per task by aggregating the top-K anomalies per subject and normalizing family contributions to sum to 1.
Output schema:
	•	subject_id, task, family, score

scripts/ utilities (canonical for reviewer “red flags”)
	•	scripts/extract_features.py
Generic feature extractor for non-Kinect formatted inputs (depends on your file structure). Includes Butterworth filtering option.
	•	scripts/extract_features_kinect.py
Kinect-style feature extractor:
	•	Computes a simple CoM-like proxy (mean of joints positions)
	•	Derivatives via np.gradient (vel/acc/jerk magnitudes)
	•	Optional Butterworth low-pass filtering
	•	Smoothness is a single monotonic proxy:
smoothness = 1 / (1 + RMS(jerk))
Outputs per-trial summary stats (mean/std/range for speed/accel/jerk + scalar smoothness).
	•	scripts/score_anomalies.py and scripts/score_anomalies_kinect.py
Standalone scoring scripts (Ledoit–Wolf + Mahalanobis²) if you want to run scoring outside anomaly_task.py.
	•	scripts/eval_cross_task_nulls.py
Computes:
	1.	same-subject cosine similarity across two tasks
	2.	different-subject cosine similarity (cross-paired)
	3.	a permutation null by shuffling subject-task pairing (or equivalent null construction)
Writes the 4 CSV outputs listed above + prints summary.
	•	explain_anomalies.py + scripts build_llm_prompts.py, run_ollama_jsonl.py, rewrite_explanations.py, make_llm_summaries.py, eval_llm_faithfulness.py
Optional LLM reporting layer:
	•	Builds prompts for high-confidence anomalies
	•	Runs an LLM locally (Ollama) and enforces JSON schema
	•	Evaluates coverage/faithfulness metrics

⸻

3) Canonical run order (paper reproduction)

The commands below assume:
	•	you are in /home/ilab/task-level-motion-consistency
	•	source .venv/bin/activate
	•	raw data exist under data/RawData/ (or your configured path)

Step 1 — Feature extraction (per task)

Choose the extractor that matches your data format.

A) Kinect-like skeleton trials
# Example: produce a unified features CSV (adjust input glob/path to your data)
python scripts/extract_features_kinect.py \
  --in-dir data/RawData \
  --out outputs/features.csv \
  --fs 30 \
  --filter none

  python scripts/extract_features_kinect.py \
  --in-dir data/RawData \
  --out outputs/features_filtered.csv \
  --fs 30 \
  --filter butter \
  --cutoff-hz 6.0

  Step 2 — Build per-task feature matrices

Depending on your setup, you may already have per-task matrices produced by the pipeline.
If not, you must split outputs/features.csv into task-specific matrices:
	•	outputs/stand_feature_matrix.csv
	•	outputs/chair_feature_matrix.csv
	•	outputs/sit_feature_matrix.csv
	•	outputs/wheelchair_feature_matrix.csv
	•	outputs/stand-frame_feature_matrix.csv

(If you want, we can add a tiny scripts/split_by_task.py helper; currently this split may be handled inside main.py or in your existing workflow.)

⸻

Step 3 — Score anomalies per task (Mahalanobis²)

Run for each task:
	•	outputs/<task>_anomaly_scores.csv
	•	outputs/<task>_anomaly_topN.txt
Step 4 — Build cross-task family profiles (Top-K aggregation)

This is the canonical step that feeds the cosine tables in the paper:
python build_cross_task_profiles.py \
  --topk 10 \
  --scores-glob "outputs/*_anomaly_scores.csv" \
  --out-csv outputs/_profiles_main.csv
  Sanity check:
  python - <<'PY'
import pandas as pd
df=pd.read_csv("outputs/_profiles_main.csv")
print("rows:", len(df))
print("tasks:", sorted(df["task"].unique()))
print("subjects:", df["subject_id"].nunique())
PY

Step 5 — Cross-task similarity + null tests (chair vs stand)

This directly supports:
	•	same-subject vs different-subject comparison
	•	permutation null

   python scripts/eval_cross_task_nulls.py \
  --in-csv outputs/_profiles_main.csv \
  --task-a chair \
  --task-b stand \
  --perm 5000 \
  --out-prefix outputs/cosine_nulls_chair_stand

 Outputs:
	•	outputs/cosine_nulls_chair_stand_summary.csv  (table-ready)
	•	outputs/cosine_nulls_chair_stand_same_subject.csv
	•	outputs/cosine_nulls_chair_stand_diff_subject.csv
	•	outputs/cosine_nulls_chair_stand_perm_null.csv

Repeat for other task pairs if needed (e.g., sit vs stand):
python scripts/eval_cross_task_nulls.py \
  --in-csv outputs/_profiles_main.csv \
  --task-a sit \
  --task-b stand \
  --perm 5000 \
  --out-prefix outputs/cosine_nulls_sit_stand

  Step 6 (Optional) — LLM reporting layer (coverage/abstention tables)

Only if you want to reproduce the JSON coverage + abstention tables:
	1.	create prompts:
   python scripts/build_llm_prompts.py --in outputs/stand_anomaly_scores.csv --out outputs/stand_prompts.jsonl
	2.	run a local LLM (example with Ollama JSONL runner):
   python scripts/run_ollama_jsonl.py --in outputs/stand_prompts.jsonl --out outputs/stand_llm.jsonl
   3.	rewrite/normalize outputs:
   python scripts/rewrite_explanations.py --in outputs/stand_llm.jsonl --out outputs/stand_explanations.csv
   4.	optional faithfulness checks:
   python scripts/eval_llm_faithfulness.py --in outputs/stand_explanations.csv

 4) Mapping outputs to paper tables
	•	Cross-task coverage & eligibility
Derived from outputs/_profiles_main.csv by counting tasks per subject:
	•	subjects with 1 task → insufficient_tasks
	•	subjects with ≥2 tasks → eligible
(Your paper reports N=30 profiled subjects, N=13 with exactly 2 tasks used for similarity.)
	•	Cosine similarity summary & per-subject
Derived from:
	•	outputs/_profiles_main.csv
	•	the cosine computations for eligible subjects
and/or the summary output:
	•	outputs/cosine_nulls_chair_stand_summary.csv (mean/median, etc.)
	•	outputs/cosine_nulls_chair_stand_same_subject.csv (per-subject cosines)
	•	LLM coverage / abstention / dominant-family agreement
Derived from the explanation CSVs produced in Step 6.

⸻

5) Notes on reviewer “red flags” (what is already implemented)

✅ Filtering option exists (Butterworth) in the Kinect extractor:
	•	--filter butter --cutoff-hz 6.0

✅ Smoothness is defined explicitly and consistent:
	•	smoothness = 1 / (1 + RMS(jerk))

✅ Mahalanobis decomposition implemented in scripts/score_anomalies.py:
	•	Uses the quadratic-form identity d² = sum_i (x−μ)_i * (P(x−μ))_i

✅ Null / permutation + same-subject vs different-subject:
	•	scripts/eval_cross_task_nulls.py

⸻

6) Common pitfalls / sanity checks

If you see duplicated header rows in CSV

If a CSV accidentally contains repeated headers, clean it before analysis.

If tasks look wrong in cross_task_profiles.csv

Always use the canonical:
	•	outputs/_profiles_main.csv produced by build_cross_task_profiles.py

If eval_cross_task_nulls.py errors with KeyError(task)

python - <<'PY'
import pandas as pd
df=pd.read_csv("outputs/_profiles_main.csv")
print(sorted(df["task"].unique()))
PY

7) Re-run everything from scratch (clean)

rm -rf outputs/*
mkdir -p outputs
# then run Steps 1→5 again



