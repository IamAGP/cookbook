"""Independent recomputation of the run 2 result from the run folders. Reads results.jsonl only.

Three arms on the 186 benchmark questions, four samples each: base, run 1 (step 37), run 2
(step 37). Also the two generated evaluation sets at one sample, and a breakdown of where
each sample lands (strict / lenient only / wrong) to see why lenient moved.

    python bird_graph_rl/run2_check.py <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from milestone1_check import RUNS, boot, load, paired  # noqa: E402

ARMS = {
    "base": ["e3_qwen3_5_9b", "e5_base9b_s2", "e5_base9b_s3", "e5_base9b_s4"],
    "run1": [f"e7_186_s37_s{i}" for i in range(1, 5)],
    "run2": [f"e9_186_s37_s{i}" for i in range(1, 5)],
}


def category(r: dict) -> str:
    if r["correct"] == 1.0:
        return "strict"
    if r["lenient"] == 1.0:
        return "lenient_only"
    if r["no_query"] == 1.0:
        return "no_query"
    if r["stop_reason"] == "max_turns":
        return "wrong_turn_cap"
    return "wrong"


def main() -> None:
    P = {a: [load(n) for n in names] for a, names in ARMS.items()}
    q = sorted(set.intersection(*(set(p) for ps in P.values() for p in ps)))
    k = 4
    mean = {a: {key: {i: sum(float(p[i][key]) for p in ps) / k for i in q} for key in ("correct", "lenient")} for a, ps in P.items()}
    avg = lambda d: round(sum(d.values()) / len(q), 4)  # noqa: E731

    def diff(a: str, b: str, key: str) -> dict:
        d = [mean[b][key][i] - mean[a][key][i] for i in q]
        pt, lo, hi = boot(d)
        return {"diff": pt, "ci95": [lo, hi], "questions_up": sum(x > 0 for x in d), "questions_down": sum(x < 0 for x in d)}

    out: dict = {"benchmark_186": {
        "n": len(q), "k": k,
        "strict": {a: avg(mean[a]["correct"]) for a in P}, "lenient": {a: avg(mean[a]["lenient"]) for a in P},
        "lenient_minus_strict": {a: round(avg(mean[a]["lenient"]) - avg(mean[a]["correct"]), 4) for a in P},
        "strict_per_pass": {a: [round(sum(p[i]["correct"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "lenient_per_pass": {a: [round(sum(p[i]["lenient"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "run2_minus_base_strict": diff("base", "run2", "correct"),
        "run1_minus_base_strict": diff("base", "run1", "correct"),
        "run2_minus_run1_strict": diff("run1", "run2", "correct"),
        "run2_minus_run1_lenient": diff("run1", "run2", "lenient"),
        "run2_minus_base_lenient": diff("base", "run2", "lenient"),
        "harness_errors": {a: sum(1 for p in ps for r in p.values() if r["error"] or r["rescore_error"]) for a, ps in P.items()},
    }}
    b = out["benchmark_186"]
    pt, lo = b["run2_minus_base_strict"]["diff"], b["run2_minus_base_strict"]["ci95"][0]
    b["bar_met_run2"] = bool(pt >= 0.05 and lo > 0)

    # --- where each of the 744 samples per arm lands
    cats = {a: Counter(category(p[i]) for p in ps for i in q) for a, ps in P.items()}
    b["sample_categories_of_744"] = {a: dict(sorted(c.items())) for a, c in cats.items()}
    b["behaviour"] = {a: {
        "mean_turns": round(sum(p[i]["turns"] for p in ps for i in q) / (k * len(q)), 2),
        "mean_queries": round(sum(p[i]["n_queries"] for p in ps for i in q) / (k * len(q)), 2),
        "query_errors": sum(p[i]["n_query_errors"] for p in ps for i in q),
        "turn_cap_hits": sum(p[i]["stop_reason"] == "max_turns" for p in ps for i in q),
        "stop_reasons": dict(Counter(p[i]["stop_reason"] for p in ps for i in q)),
    } for a, ps in P.items()}

    # --- lenient, run 2 against run 1: which questions lost it, and what the lost samples became
    lost = sorted((i for i in q if mean["run2"]["lenient"][i] < mean["run1"]["lenient"][i]),
                  key=lambda i: mean["run2"]["lenient"][i] - mean["run1"]["lenient"][i])
    gained = [i for i in q if mean["run2"]["lenient"][i] > mean["run1"]["lenient"][i]]
    tiers = Counter(P["base"][0][i]["difficulty"] for i in q)
    b["lenient_run2_vs_run1"] = {
        "questions_lost": len(lost), "questions_gained": len(gained),
        "sum_lost": round(sum(mean["run2"]["lenient"][i] - mean["run1"]["lenient"][i] for i in lost), 2),
        "sum_gained": round(sum(mean["run2"]["lenient"][i] - mean["run1"]["lenient"][i] for i in gained), 2),
        "lost_by_tier": dict(Counter(P["base"][0][i]["difficulty"] for i in lost)), "tier_sizes": dict(tiers),
        "run2_categories_on_lost_questions": dict(Counter(category(p[i]) for p in P["run2"] for i in lost)),
        "run1_categories_on_lost_questions": dict(Counter(category(p[i]) for p in P["run1"] for i in lost)),
        "biggest_losses": [{"question_id": i, "tier": P["base"][0][i]["difficulty"],
                            "lenient": {a: mean[a]["lenient"][i] for a in P}, "strict": {a: mean[a]["correct"][i] for a in P},
                            "run2_stop": [p[i]["stop_reason"] for p in P["run2"]],
                            "run2_rows": [p[i]["n_model_rows"] for p in P["run2"]],
                            "run1_rows": [p[i]["n_model_rows"] for p in P["run1"]],
                            "ref_rows": P["base"][0][i]["n_reference_rows"]} for i in lost[:12]],
    }

    # --- generated sets, one sample each
    gen = {}
    for label, base, r1, r2, r2_30 in (("heldout_instance", "e6_hi_base", "e6_hi_s37", "e9_hi_s37", "e9_hi_s30"),
                                       ("heldout_structure", "e8_hs_base", "e8_hs_s37", "e9_hs_s37", "e9_hs_s30")):
        d = {n: load(n) for n in (base, r1, r2, r2_30)}
        ids = sorted(set.intersection(*(set(v) for v in d.values())))
        gen[label] = {"n": len(ids),
                      "strict": {n: round(sum(v[i]["correct"] for i in ids) / len(ids), 4) for n, v in d.items()},
                      "lenient": {n: round(sum(v[i]["lenient"] for i in ids) / len(ids), 4) for n, v in d.items()},
                      "run2_minus_run1_strict": paired(d[r1], d[r2], ids),
                      "run2_minus_base_strict": paired(d[base], d[r2], ids),
                      "harness_errors": {n: sum(1 for r in v.values() if r["error"] or r["rescore_error"]) for n, v in d.items()}}
    out["generated_sets"] = gen

    Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
