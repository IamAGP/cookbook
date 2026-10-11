"""Reproduce a k-sample baseline from its pass folders and ask what the pre-registered bar can detect.

Bar (pre-registered 2026-10-03): per-question mean of strict over k samples; after minus before,
paired over questions; a gain is claimed only if the 95% bootstrap interval's lower bound is
above 0 AND the point estimate is at least 0.05.

The noise question: if training changed nothing, how far apart would two independent k-sample
evaluations of the same model land? That is measured here two ways from the baseline itself:
(a) analytically, from each question's within-question variance p(1-p)/k with the unbiased
    estimate of p(1-p); (b) empirically, by splitting the k passes into halves.

    python bird_graph_rl/baseline_k_analysis.py <out.json> <pass dir> <pass dir> ...
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path


def load(d: str) -> dict[int, dict]:
    return {r["question_id"]: r for r in map(json.loads, (Path(d).expanduser() / "results.jsonl").open())}


def main() -> None:
    out_path, dirs = sys.argv[1], sys.argv[2:]
    passes = [load(d) for d in dirs]
    k = len(passes)
    qids = sorted(set.intersection(*(set(p) for p in passes)))
    n = len(qids)
    X = {q: [float(p[q]["correct"]) for p in passes] for q in qids}
    mean_q = {q: sum(v) / k for q, v in X.items()}
    point = sum(mean_q.values()) / n

    rng = random.Random(0)
    boots = sorted(sum(mean_q[rng.choice(qids)] for _ in range(n)) / n for _ in range(10_000))
    ci = (boots[249], boots[9749])

    by_diff: dict[str, list[float]] = {}
    for q in qids:
        by_diff.setdefault(passes[0][q].get("difficulty") or "?", []).append(mean_q[q])

    # (a) analytic: Var(mean of k Bernoulli) = p(1-p)/k; unbiased p(1-p) is s*(k-s)/(k*(k-1)).
    within = [sum(v) * (k - sum(v)) / (k * (k - 1)) for v in X.values()]
    se_one_eval = math.sqrt(sum(w / k for w in within)) / n            # sampling SE of one k-sample mean
    se_null_diff = math.sqrt(2) * se_one_eval                          # two independent k-sample evals, no change

    # (b) empirical: every split of the passes into two halves, difference of half means.
    half_diffs = []
    for a in itertools.combinations(range(k), k // 2):
        b = [i for i in range(k) if i not in a]
        if a[0] != 0:
            continue  # each unordered split once
        da = sum(sum(X[q][i] for i in a) / len(a) for q in qids) / n
        db = sum(sum(X[q][i] for i in b) / len(b) for q in qids) / n
        half_diffs.append(da - db)

    def se_for(kk: int) -> float:
        return math.sqrt(2) * math.sqrt(sum(w / kk for w in within)) / n

    report = {
        "passes": dirs, "k": k, "n_questions": n,
        "strict_per_pass": [round(sum(p[q]["correct"] for q in qids) / n, 4) for p in passes],
        "per_question_mean_strict": round(point, 4),
        "bootstrap_95_over_questions": [round(ci[0], 4), round(ci[1], 4)],
        "by_difficulty": {d: {"n": len(v), "mean": round(sum(v) / len(v), 4)} for d, v in sorted(by_diff.items())},
        "solved_in_s_of_k": dict(sorted(Counter(int(sum(v)) for v in X.values()).items())),
        "bar": {"min_point_gain": 0.05, "after_must_reach": round(point + 0.05, 4)},
        "noise_if_nothing_changed": {
            "se_of_one_k_sample_mean_sampling_only": round(se_one_eval, 4),
            "se_of_difference_between_two_independent_k_sample_evals": round(se_null_diff, 4),
            "two_se": round(2 * se_null_diff, 4),
            "empirical_half_split_differences_k_over_2": [round(d, 4) for d in half_diffs],
            "se_of_difference_by_k": {str(kk): round(se_for(kk), 4) for kk in (1, 2, 4, 8)},
        },
    }
    Path(out_path).expanduser().write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
