"""FW-E3, the serving-route control, as pre-registered in JOURNAL.md (2026-10-03, 23:10).

Three arms on the 186 benchmark questions, four samples each, Qwen 3.8 27B on Fireworks:
  base     the session's base-only route (no adapter);
  control  a never-trained rank-32 adapter, state loaded into a fresh session and snapshotted;
  trained  FW-R1 step 37, served exactly as the control is.

Fixed in advance: "the routes differ" if the 95% interval of control minus base on strict
excludes zero; the claim about training is trained minus control at the bar (point at least
+0.05, lower bound above zero). Reads results.jsonl files only.

    python bird_graph_rl/fw_e3_check.py <out.json>
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
    "control": [f"fw27b_untrained_s{i}" for i in range(1, 5)],
    "trained": [f"fw27b_r1b_s37_s{i}" for i in range(1, 5)],
}
BAR_POINT = 0.05


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

    route, training = diff("base", "control", "correct"), diff("control", "trained", "correct")
    out = {
        "n_questions": len(q), "k": k, "questions_per_pass": {a: [len(p) for p in ps] for a, ps in P.items()},
        "strict": {a: avg(mean[a]["correct"]) for a in P}, "lenient": {a: avg(mean[a]["lenient"]) for a in P},
        "lenient_minus_strict": {a: round(avg(mean[a]["lenient"]) - avg(mean[a]["correct"]), 4) for a in P},
        "strict_per_pass": {a: [round(sum(p[i]["correct"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "lenient_per_pass": {a: [round(sum(p[i]["lenient"] for i in q) / len(q), 4) for p in ps] for a, ps in P.items()},
        "route_control_minus_base_strict": route, "route_control_minus_base_lenient": diff("base", "control", "lenient"),
        "routes_differ": bool(route["ci95"][0] > 0 or route["ci95"][1] < 0),
        "training_trained_minus_control_strict": training, "training_trained_minus_control_lenient": diff("control", "trained", "lenient"),
        "bar_trained_minus_control": {"needs_point_at_least": BAR_POINT, "needs_lower_bound_above": 0.0,
                                      "point_met": training["diff"] >= BAR_POINT, "lower_bound_met": training["ci95"][0] > 0,
                                      "bar_met": bool(training["diff"] >= BAR_POINT and training["ci95"][0] > 0)},
        "for_reference_trained_minus_base_strict": diff("base", "trained", "correct"),
        "sample_categories_of_744": {a: dict(sorted(Counter(category(p[i]) for p in ps for i in q).items())) for a, ps in P.items()},
        "behaviour": {a: {"mean_turns": round(sum(p[i]["turns"] for p in ps for i in q) / (k * len(q)), 2),
                          "query_errors": sum(p[i]["n_query_errors"] for p in ps for i in q),
                          "turn_cap_hits": sum(p[i]["stop_reason"] == "max_turns" for p in ps for i in q)} for a, ps in P.items()},
        "harness_errors": {a: sum(1 for p in ps for r in p.values() if r["error"] or r["rescore_error"]) for a, ps in P.items()},
    }
    Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
