"""Coarse shape of the generator's Cypher, derived from the query text alone.

An independent re-derivation of a generated training set's mix: the generator reports its mix
from its own shape tags; this reads the queries. Text only, no database access.

Regular expressions over Cypher are approximate. Each label is defined below so a reader can
see what was counted; disagreements with the generator's tags are listed, not hidden.

    python bird_graph_rl/cypher_shapes.py <questions.jsonl> <instances.jsonl> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
REL = re.compile(r"-\[[^\]]*\]-")
AGG = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.I)
COUNT_DISTINCT = re.compile(r"\bcount\s*\(\s*DISTINCT\b", re.I)


def split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def shape_of(cypher: str) -> dict:
    q = STRINGS.sub("''", cypher)
    hops = len(REL.findall(q))
    aggs = {m.lower() for m in AGG.findall(q)}
    if COUNT_DISTINCT.search(q):
        aggs.add("count_distinct")
    final = re.split(r"\bRETURN\b", q, flags=re.I)[-1]
    body = re.split(r"\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b", final, flags=re.I)[0]
    distinct_projection = bool(re.match(r"\s*DISTINCT\b", body, re.I))
    items = split_top_level(re.sub(r"^\s*DISTINCT\b", "", body, flags=re.I))
    has_order = re.search(r"\bORDER\s+BY\b", q, re.I) is not None
    limit = re.search(r"\bLIMIT\s+(\$?\w+)", q, re.I)
    skip = re.search(r"\bSKIP\b", q, re.I) is not None
    if skip and has_order:
        ordering = "nth_ranked"
    elif has_order and limit:
        ordering = "argmax_argmin" if limit.group(1) == "1" else "top_k"
    elif has_order:
        ordering = "order_by"
    elif limit:
        ordering = "limit_only"
    else:
        ordering = "none"
    kinds = aggs - {"count_distinct"} if "count_distinct" in aggs and not re.search(r"\bcount\s*\((?!\s*DISTINCT)", q, re.I) else aggs
    measures = {a for a in kinds if a in ("sum", "avg", "min", "max")}
    how_many = bool(kinds & {"count", "count_distinct"})
    if not kinds:
        family = "lookup"
    elif measures and how_many:
        family = "more_than_one_aggregate"
    elif len(measures) > 1:
        family = "more_than_one_aggregate"
    elif measures:
        family = "sum_avg_min_max"
    else:
        family = "how_many"
    extras = set()
    if distinct_projection:
        extras.add("distinct_projection")
    if re.search(r"/", q) and kinds:
        extras.add("ratio")
    if re.search(r"\b(?:count|sum|avg)\s*\(\s*CASE\b", q, re.I) or re.search(r"\bCASE\b.*\bEND\b", q, re.I | re.S):
        extras.add("conditional_aggregate")
    if re.search(r"\bNOT\b|<>", q, re.I):
        extras.add("negation")
    if re.search(r"\bEXISTS\s*\{", q, re.I):
        extras.add("existence")
    # HAVING in Cypher: an aggregate in a WITH, then a WHERE on its alias.
    if re.search(r"\bWITH\b[^;]*\b(?:count|sum|avg|min|max)\s*\([^;]*\bWHERE\b", q, re.I | re.S):
        extras.add("having")
    return {"hops": hops, "family": family, "aggregations": sorted(aggs), "ordering": ordering,
            "extras": sorted(extras), "return_items": len(items)}


def main() -> None:
    questions = [json.loads(line) for line in Path(sys.argv[1]).expanduser().open()]
    instances = {}
    for line in Path(sys.argv[2]).expanduser().open():
        d = json.loads(line); instances[d["instance_id"]] = d
    n = len(questions)
    shapes, hop_mismatch, width_mismatch, tag_family = [], [], [], Counter()
    for t in questions:
        inst = instances[t["instance_id"]]
        s = shape_of(inst["cypher"])
        s["instance_id"] = t["instance_id"]
        shapes.append(s)
        if s["hops"] != int(t["hops"]):
            hop_mismatch.append(t["instance_id"])
        width = len(t["rows"][0]) if t["rows"] else None
        if width is not None and width != s["return_items"]:
            width_mismatch.append(t["instance_id"])
        tag_family[str(inst.get("shape"))[:60]] += 1

    def share(c: Counter) -> dict:
        return {str(k): [v, round(v / n, 4)] for k, v in c.most_common()}

    hops = Counter(s["hops"] for s in shapes)
    report = {
        "n": n,
        "family_from_query_text": share(Counter(s["family"] for s in shapes)),
        "hops_from_query_text": share(hops),
        "share_at_most_1_hop": round(sum(v for k, v in hops.items() if k <= 1) / n, 4),
        "share_2_hops": round(hops[2] / n, 4),
        "share_3_to_4_hops": round(sum(v for k, v in hops.items() if k >= 3) / n, 4),
        "ordering_from_query_text": share(Counter(s["ordering"] for s in shapes)),
        "no_extras": [sum(1 for s in shapes if not s["extras"]), round(sum(1 for s in shapes if not s["extras"]) / n, 4)],
        "extras_from_query_text": share(Counter(e for s in shapes for e in s["extras"])),
        "returned_columns_from_rows": share(Counter(min(len(t["rows"][0]), 3) if t["rows"] else 0 for t in questions)),
        "hops_disagree_with_file": len(hop_mismatch),
        "return_items_disagree_with_row_width": len(width_mismatch),
        "hint_share": round(sum(1 for t in questions if (t.get("hint") or "").strip()) / n, 4),
        "questions_per_structure_max": max(Counter(t["structure_id"] for t in questions).values()),
        "generator_shape_tags_top": dict(tag_family.most_common(12)),
        "hop_mismatch_ids": hop_mismatch[:10], "width_mismatch_ids": width_mismatch[:10],
    }
    Path(sys.argv[3]).expanduser().write_text(json.dumps({"report": report, "shapes": shapes}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
