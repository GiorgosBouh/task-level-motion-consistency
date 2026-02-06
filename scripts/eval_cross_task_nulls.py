#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="Long profiles: subject_id,task,family,score")
    ap.add_argument("--task-a", required=True, help="e.g., chair")
    ap.add_argument("--task-b", required=True, help="e.g., stand")
    ap.add_argument("--perm", type=int, default=2000, help="Permutation iterations")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", default="outputs/nulls")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # pivot -> rows: (subject_id, task), cols: family, values: score
    M = df.pivot_table(index=["subject_id","task"], columns="family", values="score", aggfunc="sum", fill_value=0.0)
    # L1 normalize per row (family composition)
    X = M.to_numpy(dtype=float)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    Xn = X / row_sum
    Mnorm = pd.DataFrame(Xn, index=M.index, columns=M.columns)

    # Extract task vectors
    A = Mnorm.xs(args.task_a, level="task", drop_level=False)
    B = Mnorm.xs(args.task_b, level="task", drop_level=False)

    # Subjects in both tasks
    sA = set(A.index.get_level_values("subject_id"))
    sB = set(B.index.get_level_values("subject_id"))
    common = sorted(list(sA.intersection(sB)))

    if len(common) < 2:
        raise SystemExit(f"Not enough common subjects between {args.task_a} and {args.task_b}: {len(common)}")

    # Build aligned matrices for common subjects
    VA = np.vstack([A.loc[(sid, args.task_a)].to_numpy() for sid in common])
    VB = np.vstack([B.loc[(sid, args.task_b)].to_numpy() for sid in common])

    # Same-subject cosine
    same = np.array([cosine(VA[i], VB[i]) for i in range(len(common))], dtype=float)

    # Different-subject cosines (all pairs i != j)
    diff = []
    for i in range(len(common)):
        for j in range(len(common)):
            if i == j:
                continue
            diff.append(cosine(VA[i], VB[j]))
    diff = np.array(diff, dtype=float)

    # Permutation null: shuffle subject mapping in VB
    rng = np.random.default_rng(args.seed)
    null_means = []
    for _ in range(args.perm):
        perm_idx = rng.permutation(len(common))
        vals = [cosine(VA[i], VB[perm_idx[i]]) for i in range(len(common))]
        null_means.append(np.nanmean(vals))
    null_means = np.array(null_means, dtype=float)

    # Observed mean
    obs_mean = float(np.nanmean(same))
    p_value = float((np.sum(null_means >= obs_mean) + 1) / (len(null_means) + 1))

    # Save outputs (for manuscript tables/figures)
    out_same = pd.DataFrame({"subject_id": common, "cosine_same_subject": same})
    out_same.to_csv(f"{args.out_prefix}_same_subject.csv", index=False)

    out_diff = pd.DataFrame({"cosine_diff_subject": diff})
    out_diff.to_csv(f"{args.out_prefix}_diff_subject.csv", index=False)

    out_null = pd.DataFrame({"null_mean_cosine": null_means})
    out_null.to_csv(f"{args.out_prefix}_perm_null.csv", index=False)

    summary = pd.DataFrame([{
        "task_a": args.task_a,
        "task_b": args.task_b,
        "n_common_subjects": len(common),
        "same_mean": np.nanmean(same),
        "same_median": np.nanmedian(same),
        "diff_mean": np.nanmean(diff),
        "diff_median": np.nanmedian(diff),
        "perm_iterations": args.perm,
        "perm_null_mean": np.mean(null_means),
        "perm_null_std": np.std(null_means),
        "observed_mean": obs_mean,
        "p_value_one_sided": p_value
    }])
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print("✅ Wrote:")
    print(f"  {args.out_prefix}_same_subject.csv")
    print(f"  {args.out_prefix}_diff_subject.csv")
    print(f"  {args.out_prefix}_perm_null.csv")
    print(f"  {args.out_prefix}_summary.csv")
    print(f"Observed mean cosine={obs_mean:.4f} vs null mean={np.mean(null_means):.4f} (p={p_value:.4g})")

if __name__ == "__main__":
    main()
