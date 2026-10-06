"""FW-R1 primary test, as pre-registered in JOURNAL.md (2026-10-03, 18:05).

Qwen 3.8 27B on the 186 benchmark questions, four samples per question, trained (step 37 of
``fw_rl_27b_run1b``) minus base, both sampled on Fireworks. Reads results.jsonl files only.

    python bird_graph_rl/fw_r1_check.py <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from milestone1_check import boot, load  # noqa: E402
from run2_check import category  # noqa: E402

ARMS = {
    "base": ["e2fw_full_qwen3_8_27b", "fw27b_base_s2", "fw27b_base_s3", "fw27b_base_s4"],
    "trained": [f"fw27b_r1b_s37_s{i}" for i in range(1, 5)],
}
BAR_POINT = 0.05
NINE_B_RUN1_GAIN = 0.0457  # ledger: Tinker run 1 minus base, strict, 9B; the prediction's threshold


def main() -> None:
    P = {a: [load(n) for n in names] for a, names in ARMS.items()}
    sizes = {a: [len(p) for p in ps] for a, ps in P.items()}
    q = sorted(set.intersection(*(set(p) for ps in P.values() for p in ps)))
    k = 4
    mean = {a: {key: {i: sum(float(p[i][key]) for p in ps) / k for i in q} for key in ("correct", "lenient")} for a, ps in P.items()}
    avg = lambda d: round(sum(d.values()) / len(q), 4)  # noqa: E731

    def diff(key: str) -> dict:
        d = [mean["trained"][key][i] - mean["base"][key][i] for i in q]
        pt, lo, hi = boot(d)
        return {"diff": pt, "ci95": [lo, hi], "questions_up": sum(x > 0 for x in d), "questions_down": sum(x < 0 for x in d)}

    strict, lenient = diff("correct"), diff("lenient")
    tiers: dict[str, list[int]] = {}
    for i in q:
        tiers.setdefault(P["base"][0][i]["difficulty"], []).append(i)
    solved = {i: int(sum(p[i]["correct"] for p in P["base"])) for i in q}
    groups = {"never_0_of_4": [i for i in q if solved[i] == 0], "mixed_1_to_3": [i for i in q if 0 < solved[i] < 4],
              "always_4_of_4": [i for i in q if solved[i] == 4]}
    out = {
        "n_questions": len(q), "k": k, "questions_per_pass": sizes,
        "strict": {a: avg(mean[a]["correct"]) for a in P}, "lenient": {a: avg(mean[a]["lenient"]) for a in P},
        "lenient_minus_strict": {a: round(avg(mean[a]["lenient"]) - avg(mean[a]["correct"]), 4) for a in P},
        "strict_per_pass": {a: [round(sum(p[i]["correct"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "lenient_per_pass": {a: [round(sum(p[i]["lenient"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "trained_minus_base_strict": strict, "trained_minus_base_lenient": lenient,
        "bar": {"needs_point_at_least": BAR_POINT, "needs_lower_bound_above": 0.0,
                "point_met": strict["diff"] >= BAR_POINT, "lower_bound_met": strict["ci95"][0] > 0,
                "bar_met": bool(strict["diff"] >= BAR_POINT and strict["ci95"][0] > 0)},
        "prediction_gain_smaller_than_9b_run1": {"threshold": NINE_B_RUN1_GAIN, "holds": bool(strict["diff"] < NINE_B_RUN1_GAIN)},
        "sample_categories_of_744": {a: dict(sorted(Counter(category(p[i]) for p in ps for i in q).items())) for a, ps in P.items()},
        "behaviour": {a: {"mean_turns": round(sum(p[i]["turns"] for p in ps for i in q) / (k * len(q)), 2),
                          "mean_queries": round(sum(p[i]["n_queries"] for p in ps for i in q) / (k * len(q)), 2),
                          "query_errors": sum(p[i]["n_query_errors"] for p in ps for i in q),
                          "turn_cap_hits": sum(p[i]["stop_reason"] == "max_turns" for p in ps for i in q)} for a, ps in P.items()},
        "harness_errors": {a: sum(1 for p in ps for r in p.values() if r["error"] or r["rescore_error"]) for a, ps in P.items()},
        "by_tier": {t: {"n": len(v), "base": round(sum(mean["base"]["correct"][i] for i in v) / len(v), 4),
                        "trained": round(sum(mean["trained"]["correct"][i] for i in v) / len(v), 4)} for t, v in sorted(tiers.items())},
        # Groups are defined by the base passes, so base is at its extreme by construction in the
        # first and last: movement there includes regression to the mean and is not an effect size.
        "by_base_group_descriptive_only": {g: {"n": len(v), "base": round(sum(mean["base"]["correct"][i] for i in v) / max(len(v), 1), 4),
                                               "trained": round(sum(mean["trained"]["correct"][i] for i in v) / max(len(v), 1), 4)} for g, v in groups.items()},
        "identical_final_query_base_pass1_vs_trained_pass1": sum(P["base"][0][i]["final_query"] == P["trained"][0][i]["final_query"] for i in q),
        "identical_final_query_base_pass1_vs_base_pass2": sum(P["base"][0][i]["final_query"] == P["base"][1][i]["final_query"] for i in q),
    }
    Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
