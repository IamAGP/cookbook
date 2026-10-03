"""Join the generator's structures to its questions and set them beside the human query shapes.

A check, not a target, except for the depth rule the spec itself declares as one (amendment C2).
Reads the regenerated files and BIRD's human-shape report; writes one JSON.

    python bird_graph_rl/datagen_join.py <datagen dir> <human shapes json> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AGG = {"count distinct": "count_distinct"}
ORDER = {"order-by": "order_by", "top-k": "top_k", "argmax": "argmax_argmin", "argmin": "argmax_argmin", "n-th ranked": "nth_ranked"}


def share(counter: Counter, n: int) -> dict[str, list]:
    return {str(k): [c, round(c / n, 4)] for k, c in counter.most_common()}


def main() -> None:
    d, human_path, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser(), Path(sys.argv[3]).expanduser()
    S = {}
    for line in (d / "structures.snapshot.jsonl").open():
        s = json.loads(line)
        s["sig"] = json.loads(s["signature"]) if isinstance(s["signature"], str) else s["signature"]
        S[s["structure_id"]] = s
    Q = [json.loads(line) for line in (d / "all_questions.jsonl").open()]
    human = json.loads(human_path.read_text())
    report: dict = {"n_structures": len(S), "n_questions": len(Q),
                    "structures_by_split": dict(Counter(s["split"] for s in S.values())),
                    "structures_by_novelty": dict(Counter(str(s["novelty"]) for s in S.values())),
                    "questions_missing_structure": sum(1 for q in Q if q["structure_id"] not in S),
                    "question_split_disagrees_with_structure_split": 0, "splits": {}}
    by_split = defaultdict(list)
    for q in Q:
        by_split[q["split"]].append(q)
        st = S.get(q["structure_id"])
        if st and (st["split"] == "heldout_structure") != (q["split"] == "heldout_structure"):
            report["question_split_disagrees_with_structure_split"] += 1
    for split, qs in sorted(by_split.items()):
        n = len(qs)
        sig = [S[q["structure_id"]]["sig"] for q in qs]
        hops = Counter(int(q["hops"]) for q in qs)
        per_struct = Counter(q["structure_id"] for q in qs)
        hinted = sum(1 for q in qs if (q.get("hint") or "").strip())
        report["splits"][split] = {
            "questions": n, "structures": len(per_struct),
            "questions_per_structure": {"min": min(per_struct.values()), "max": max(per_struct.values())},
            "hops": share(hops, n),
            "share_at_most_1_hop": round(sum(c for h, c in hops.items() if h <= 1) / n, 4),
            "share_3_to_4_hops": round(sum(c for h, c in hops.items() if h >= 3) / n, 4),
            "aggregation": share(Counter(AGG.get(x["aggregation"][0], x["aggregation"][0]) for x in sig), n),
            "grouping_share": round(sum(1 for x in sig if x["grouping"] != "none") / n, 4),
            "ordering": share(Counter(ORDER.get(x["ordering"], x["ordering"]) for x in sig), n),
            "no_extras_share": round(sum(1 for x in sig if not x["extras"]) / n, 4),
            "extras": share(Counter(e for x in sig for e in x["extras"]), n),
            "anchor_kind": share(Counter(str(S[q["structure_id"]]["anchor_kind"]) for q in qs), n),
            "hint_share": round(hinted / n, 4),
            "novelty": dict(Counter(str(S[q["structure_id"]]["novelty"]) for q in qs)),
        }
    def h(key: str) -> dict[str, float]:
        return {str(x["value"]): x["share"] for x in human[key]}
    report["human_reference"] = {
        "source": human["source"], "scope_note": human["scope_note"],
        "hops": h("hops"), "aggregation": h("aggregation"), "ordering": h("ordering"),
        "grouping_share": {str(x["value"]): x["share"] for x in human["grouping"]}.get("True"),
        "no_extras_share": human["extras_none"]["share"],
    }
    # Which novel_component structures are novel only in something the scorer cannot see?
    unseen = Counter()
    order_only = 0
    for s in S.values():
        if s["novelty"] == "novel_component":
            comps = s.get("unseen_components") or []
            for c in comps:
                unseen[json.dumps(c)[:80]] += 1
            if comps and all("order" in json.dumps(c).lower() for c in comps):
                order_only += 1
    report["novel_component_unseen_components"] = dict(unseen.most_common(40))
    report["novel_component_structures_novel_only_in_ordering"] = order_only
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
