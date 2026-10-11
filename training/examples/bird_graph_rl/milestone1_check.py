"""Independent recomputation of milestone 1 from the run folders. Reads results.jsonl files only.

    python bird_graph_rl/milestone1_check.py <out.json>
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path("~/bird_rl_runs").expanduser()
BASE_186 = ["e3_qwen3_5_9b", "e5_base9b_s2", "e5_base9b_s3", "e5_base9b_s4"]
S37_186 = ["e7_186_s37_s1", "e7_186_s37_s2", "e7_186_s37_s3", "e7_186_s37_s4"]


def load(name: str) -> dict:
    return {r["question_id"]: r for r in map(json.loads, (RUNS / name / "results.jsonl").open())}


def boot(diffs: list[float], seed: int = 0, n: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return [round(sum(diffs) / len(diffs), 4), round(bs[int(0.025 * n) - 1], 4), round(bs[int(0.975 * n) - 1], 4)]


def paired(a: dict, b: dict, ids: list, key: str = "correct") -> dict:
    d = [float(b[i][key]) - float(a[i][key]) for i in ids]
    pt, lo, hi = boot(d)
    return {"n": len(ids), "before": round(sum(float(a[i][key]) for i in ids) / len(ids), 4),
            "after": round(sum(float(b[i][key]) for i in ids) / len(ids), 4), "diff": pt, "ci95": [lo, hi],
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d)}


def main() -> None:
    out: dict = {}

    # --- selection set: held-out instances, one sample each
    hi = {n: load(f"e6_hi_{n}") for n in ("base", "s10", "s20", "s30", "s37")}
    ids = sorted(set.intersection(*(set(v) for v in hi.values())))
    out["heldout_instance"] = {
        "n": len(ids), "strict": {n: round(sum(v[i]["correct"] for i in ids) / len(ids), 4) for n, v in hi.items()},
        "s37_minus_base": paired(hi["base"], hi["s37"], ids),
        "harness_errors": {n: sum(1 for r in v.values() if r["error"] or r["rescore_error"]) for n, v in hi.items()},
        "turns": {n: round(sum(v[i]["turns"] for i in ids) / len(ids), 2) for n, v in hi.items()},
        "turn_cap_hits": {n: sum(v[i]["stop_reason"] == "max_turns" for i in ids) for n, v in hi.items()},
        "query_errors": {n: sum(v[i]["n_query_errors"] for i in ids) for n, v in hi.items()},
    }

    # --- the 186, four samples each, baseline vs step 37
    B, S = [load(n) for n in BASE_186], [load(n) for n in S37_186]
    q = sorted(set.intersection(*(set(p) for p in B + S)))
    def mean_q(passes: list[dict], key: str) -> dict:
        return {i: sum(float(p[i][key]) for p in passes) / len(passes) for i in q}
    b_s, s_s, b_l, s_l = mean_q(B, "correct"), mean_q(S, "correct"), mean_q(B, "lenient"), mean_q(S, "lenient")
    diff = [s_s[i] - b_s[i] for i in q]
    pt, lo, hi_ = boot(diff)
    tier = defaultdict(list)
    for i in q:
        tier[B[0][i]["difficulty"]].append(i)
    solved = {i: int(sum(p[i]["correct"] for p in B)) for i in q}
    groups = {"never_0_of_4": [i for i in q if solved[i] == 0], "mixed_1_to_3": [i for i in q if 0 < solved[i] < 4],
              "always_4_of_4": [i for i in q if solved[i] == 4]}
    reference = 0.608
    out["benchmark_186"] = {
        "n": len(q), "k": 4,
        "strict_per_pass": {"base": [round(sum(p[i]["correct"] for i in q) / len(q), 4) for p in B],
                            "s37": [round(sum(p[i]["correct"] for i in q) / len(q), 4) for p in S]},
        "per_question_mean_strict": {"base": round(sum(b_s.values()) / len(q), 4), "s37": round(sum(s_s.values()) / len(q), 4)},
        "paired_diff": pt, "ci95": [lo, hi_],
        "bar": {"needs_point_at_least": 0.05, "needs_lower_bound_above": 0.0,
                "point_met": pt >= 0.05, "lower_bound_met": lo > 0, "bar_met": pt >= 0.05 and lo > 0,
                "short_by": round(0.05 - pt, 4)},
        "fraction_of_gap_to_reference_closed": round(pt / (reference - sum(b_s.values()) / len(q)), 4),
        "lenient": {"base": round(sum(b_l.values()) / len(q), 4), "s37": round(sum(s_l.values()) / len(q), 4)},
        "by_tier": {t: {"n": len(v), "base": round(sum(b_s[i] for i in v) / len(v), 4), "s37": round(sum(s_s[i] for i in v) / len(v), 4)} for t, v in sorted(tier.items())},
        "by_baseline_group": {g: {"n": len(v), "base": round(sum(b_s[i] for i in v) / len(v), 4), "s37": round(sum(s_s[i] for i in v) / len(v), 4),
                                  "contribution_to_mean": round(sum(s_s[i] - b_s[i] for i in v) / len(q), 4)} for g, v in groups.items()},
        "s37_passes_pairwise_identical_final_query": [sum(S[a][i]["final_query"] == S[b][i]["final_query"] for i in q) for a in range(4) for b in range(a + 1, 4)],
        "harness_errors_s37": sum(1 for p in S for r in p.values() if r["error"] or r["rescore_error"]),
        "turns": {"base": round(sum(p[i]["turns"] for p in B for i in q) / (4 * len(q)), 2), "s37": round(sum(p[i]["turns"] for p in S for i in q) / (4 * len(q)), 2)},
        "turn_cap_hits": {"base": sum(p[i]["stop_reason"] == "max_turns" for p in B for i in q), "s37": sum(p[i]["stop_reason"] == "max_turns" for p in S for i in q)},
    }

    # --- unseen structures, one sample each, by hop and novelty; seen-structure side from held-out instances
    hb, hs = load("e8_hs_base"), load("e8_hs_s37")
    ids = sorted(set(hb) & set(hs))
    structures = {}
    for line in (RUNS / "datagen_v3" / "structures.snapshot.jsonl").open():
        s = json.loads(line); structures[s["structure_id"]] = s
    # results.jsonl does not carry the structure id; the question file maps instance id to it.
    sid = {}
    for line in (RUNS / "datagen_v3" / "all_questions.jsonl").open():
        d = json.loads(line); sid[d["instance_id"]] = d["structure_id"]
    novelty = {i: structures[sid[i]]["novelty"] for i in ids}
    hop = lambda r: r["difficulty"]
    res = {"overall": paired(hb, hs, ids)}
    for nv in ("novel_combination", "novel_component"):
        res[nv] = paired(hb, hs, [i for i in ids if novelty[i] == nv])
    res["by_hop"] = {h: paired(hb, hs, [i for i in ids if hop(hb[i]) == h]) for h in sorted({hop(hb[i]) for i in ids})}
    two = [i for i in ids if hop(hb[i]) == "hops2"]
    res["two_hop_unseen_structures"] = paired(hb, hs, two)
    res["two_hop_by_novelty"] = {nv: paired(hb, hs, [i for i in two if novelty[i] == nv]) for nv in ("novel_combination", "novel_component")}
    seen_ids = sorted(set(hi["base"]) & set(hi["s37"]))
    res["two_hop_seen_structures"] = paired(hi["base"], hi["s37"], [i for i in seen_ids if hop(hi["base"][i]) == "hops2"])
    res["harness_errors"] = [sum(1 for r in d.values() if r["error"] or r["rescore_error"]) for d in (hb, hs)]
    out["heldout_structure"] = res

    Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
