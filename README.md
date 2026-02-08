Task-Level Motion Consistency
=============================

This repository contains the full analysis pipeline used in the paper:

"When to Explain, When to Abstain: A Conditional Framework for Biomechanical Anomaly Interpretation"

The code implements a task-level anomaly detection framework for human motion
data, followed by a conditional interpretation layer that explicitly abstains
from explanation when evidential support is insufficient.

The repository is designed for transparency and full reproducibility.


----------------------------------------------------------------------
Project structure (paths you must know)
----------------------------------------------------------------------

Inputs
------
Raw data directory (not included in Git; ignored via .gitignore):

- data/RawData/        (example path)

The scripts expect trial files in a consistent format (see dataset export).
Raw data are not distributed with this repository.


Outputs (paper artifacts are generated here)
--------------------------------------------
- outputs/             (generated; not committed)

Feature matrices per task:
- outputs/stand_feature_matrix.csv
- outputs/chair_feature_matrix.csv
- outputs/sit_feature_matrix.csv
- outputs/wheelchair_feature_matrix.csv
- outputs/stand-frame_feature_matrix.csv

Anomaly scoring per task:
- outputs/stand_anomaly_scores.csv
- outputs/chair_anomaly_scores.csv
- outputs/sit_anomaly_scores.csv
- outputs/wheelchair_anomaly_scores.csv
- outputs/stand-frame_anomaly_scores.csv

Top-N tables (used in paper tables / appendix):
- outputs/*_anomaly_topN.txt

Cross-task family profiles (used for cosine similarity analysis):
- outputs/profiles_main.csv          (canonical file used in the paper)
- outputs/cross_task_profiles.csv    (older file; may contain duplicates)

Cross-task null tests (example: chair vs stand):
When running with:
  --out-prefix outputs/cosine_nulls_chair_stand

The following files are produced:
- outputs/cosine_nulls_chair_stand_summary.csv
- outputs/cosine_nulls_chair_stand_same_subject.csv
- outputs/cosine_nulls_chair_stand_diff_subject.csv
- outputs/cosine_nulls_chair_stand_perm_null.csv


----------------------------------------------------------------------
What each main script does
----------------------------------------------------------------------

Root-level pipeline
-------------------
- main.py
  Orchestrates the end-to-end pipeline (depending on configuration).
  Can be used as a single entry point for full reproduction.

- anomaly_task.py
  Task-generic anomaly scoring pipeline.
  Reads a feature matrix, fits a Ledoit–Wolf covariance model,
  computes squared Mahalanobis distance, ranks trials,
  and writes:
    *_anomaly_scores.csv
    *_anomaly_topN.txt

- anomaly_stand.py
  Specialization for stand / stand-frame variants,
  if the dataset distinguishes these contexts.

- build_cross_task_profiles.py
  Builds subject-level family-composition profiles per task by:
  - selecting the top-K anomalous trials per subject,
  - aggregating feature-family contributions,
  - normalizing family scores to sum to 1.

  Output schema:
  subject_id, task, family, score


scripts/ (utilities)
--------------------
- scripts/extract_features_kinect.py
  Kinect-style feature extractor:
  - Computes a simple center-of-mass-like proxy (mean joint position)
  - Derivatives via numpy.gradient (speed / acceleration / jerk magnitudes)
  - Optional Butterworth low-pass filtering
  - Smoothness defined as:
        smoothness = 1 / (1 + RMS(jerk))

  Outputs per-trial summary statistics:
  - mean / std / range for speed, acceleration, jerk
  - scalar smoothness

- scripts/extract_features.py
  Generic feature extractor for non-Kinect formatted inputs.
  Includes optional filtering.

- scripts/score_anomalies_kinect.py
- scripts/score_anomalies.py
  Standalone anomaly scoring scripts (Ledoit–Wolf + Mahalanobis²),
  useful if scoring is run outside anomaly_task.py.

- scripts/eval_cross_task_nulls.py
  Computes:
    1. same-subject cosine similarity across two tasks
    2. different-subject cosine similarity
    3. permutation-based null distribution (task-label shuffle)

  Writes four CSV outputs and prints summary statistics.


Optional language-based reporting layer
---------------------------------------
Used only to reproduce the reporting / abstention analyses.

- scripts/build_llm_prompts.py
- scripts/run_ollama_jsonl.py
- scripts/rewrite_explanations.py
- scripts/make_llm_summaries.py
- scripts/eval_llm_faithfulness.py

This layer generates constrained natural-language summaries for
high-confidence anomalies and explicitly abstains otherwise.
It does not influence anomaly detection or ranking.


----------------------------------------------------------------------
Canonical run order (paper reproduction)
----------------------------------------------------------------------

Assumptions:
- You are in: /home/ilab/task-level-motion-consistency
- Virtual environment is active:
    source .venv/bin/activate
- Raw data exist under: data/RawData/


Step 1 — Feature extraction
---------------------------

A) Kinect-like skeleton trials (no filtering)

python scripts/extract_features_kinect.py \
  --in-dir data/RawData \
  --out outputs/features.csv \
  --fs 30 \
  --filter none


B) Filtering sensitivity analysis (Butterworth, 6 Hz)

python scripts/extract_features_kinect.py \
  --in-dir data/RawData \
  --out outputs/features_filtered.csv \
  --fs 30 \
  --filter butter \
  --cutoff-hz 6.0


Step 2 — Build per-task feature matrices
----------------------------------------
If not produced automatically by main.py,
split outputs/features.csv into:

- outputs/stand_feature_matrix.csv
- outputs/chair_feature_matrix.csv
- outputs/sit_feature_matrix.csv
- outputs/wheelchair_feature_matrix.csv
- outputs/stand-frame_feature_matrix.csv


Step 3 — Score anomalies per task
---------------------------------
Run anomaly scoring for each task to produce:

- outputs/*_anomaly_scores.csv
- outputs/*_anomaly_topN.txt


Step 4 — Build cross-task family profiles (Top-K aggregation)
-------------------------------------------------------------

python build_cross_task_profiles.py \
  --topk 10 \
  --scores-glob "outputs/*_anomaly_scores.csv" \
  --out-csv outputs/profiles_main.csv


Sanity check:
-------------
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/profiles_main.csv")
print("rows:", len(df))
print("tasks:", sorted(df["task"].unique()))
print("subjects:", df["subject_id"].nunique())
PY


Step 5 — Cross-task similarity and null tests (chair vs stand)
--------------------------------------------------------------

python scripts/eval_cross_task_nulls.py \
  --in-csv outputs/profiles_main.csv \
  --task-a chair \
  --task-b stand \
  --perm 5000 \
  --out-prefix outputs/cosine_nulls_chair_stand


Outputs:
- outputs/cosine_nulls_chair_stand_summary.csv
- outputs/cosine_nulls_chair_stand_same_subject.csv
- outputs/cosine_nulls_chair_stand_diff_subject.csv
- outputs/cosine_nulls_chair_stand_perm_null.csv


Step 6 (Optional) — Language-based reporting layer
--------------------------------------------------
Used only to reproduce coverage / abstention tables.

1. Build prompts:
   python scripts/build_llm_prompts.py \
     --in outputs/stand_anomaly_scores.csv \
     --out outputs/stand_prompts.jsonl

2. Run local LLM (example with Ollama):
   python scripts/run_ollama_jsonl.py \
     --in outputs/stand_prompts.jsonl \
     --out outputs/stand_llm.jsonl

3. Rewrite and normalize explanations:
   python scripts/rewrite_explanations.py \
     --in outputs/stand_llm.jsonl \
     --out outputs/stand_explanations.csv

4. Optional faithfulness checks:
   python scripts/eval_llm_faithfulness.py \
     --in outputs/stand_explanations.csv


----------------------------------------------------------------------
Mapping outputs to paper results
----------------------------------------------------------------------

- Task coverage and eligibility:
  Derived from outputs/profiles_main.csv
  Subjects with one task → insufficient_tasks
  Subjects with two tasks → eligible for cross-task analysis

- Cross-task cosine similarity:
  Derived from:
    outputs/profiles_main.csv
    outputs/cosine_nulls_*_same_subject.csv

- Permutation null statistics:
  Derived from:
    outputs/cosine_nulls_*_perm_null.csv

- Reporting coverage and abstention:
  Derived from the optional LLM outputs in Step 6


----------------------------------------------------------------------
Environment and requirements
----------------------------------------------------------------------

Python version:
- Python 3.10 or newer

Required packages:
- numpy >= 1.26
- pandas >= 2.2
- scipy >= 1.11
- scikit-learn >= 1.4
- tqdm >= 4.66
- python-dotenv >= 1.0

Optional (for plotting or reporting):
- matplotlib >= 3.8
- requests >= 2.31


Installation:
-------------
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


----------------------------------------------------------------------
Notes for reviewers
----------------------------------------------------------------------

- Raw data are excluded by design.
- All analysis steps are deterministic.
- Filtering, anomaly detection, and interpretation are explicitly separated.
- Abstention is treated as a valid scientific outcome.
- The repository corresponds exactly to the analyses reported in the paper.