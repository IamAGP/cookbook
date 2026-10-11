"""Coarse shape of a SQL query: which kinds of question do people actually ask?

Purpose: an evidence-based check on the data generator's naturalness filter (SPEC amendment B3).
It reports the distribution of coarse query shapes in BIRD's human-written training questions.
It is a CHECK, not a target: the generated distribution is not to be tuned to match it.

Source: Hugging Face ``birdsql/bird23-train-filtered`` (CC BY-SA 4.0), the subset BIRD itself
filtered for quality: 6,601 of the 9,428 training questions. Shapes are therefore measured on
that filtered subset, not on the whole split.

A coarse shape is ``(hops, aggregation, grouping, ordering, extras)``:
- hops: table references in the whole statement minus one, bucketed 0..4+. This is the
  relational count. A graph has no junction tables, so the same question usually needs fewer
  hops as a graph query; treat it as an upper bound on graph hops.
- aggregation: none | count | count_distinct | sum | avg | min | max | multiple
- grouping: whether any GROUP BY is present
- ordering: none | order_by | top_k | argmax_argmin | nth_ranked | limit_only (outermost SELECT)
- extras: sorted subset of the tags in EXTRA_TAGS

    python bird_graph_rl/sql_shapes.py <train.jsonl> <out.json>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

EXTRA_TAGS = (
    "subquery", "existence", "negation", "ratio", "difference_of_aggregates", "having",
    "conditional_aggregate", "comparison_two_entities", "set_operation", "distinct",
    "argmax_via_subquery", "window", "cte",
)
AGG_TYPES: dict[type, str] = {exp.Count: "count", exp.Sum: "sum", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}
EVAL_DB = "codebase_community"
EXPECTED_ROWS = 6601
_DATE_FN = re.compile(r"\b(strftime|julianday|date|datetime)\s*\(", re.I)


def _agg_name(node: exp.Expression) -> str:
    name = AGG_TYPES[type(node)]
    if name == "count" and (isinstance(node.this, exp.Distinct) or node.args.get("distinct")):
        return "count_distinct"
    return name


def _has_agg(node: exp.Expression) -> bool:
    return any(node.find(t) is not None for t in AGG_TYPES)


def _outer_select(tree: exp.Expression) -> exp.Select | None:
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def _ordering(select: exp.Select | None) -> str:
    if select is None:
        return "none"
    order, limit = select.args.get("order"), select.args.get("limit")
    if limit is None:
        return "order_by" if order is not None else "none"
    if order is None:
        return "limit_only"
    if select.args.get("offset") is not None:
        return "nth_ranked"  # ORDER BY ... LIMIT n OFFSET k, or SQLite's LIMIT k, n: "the k-th ranked"
    value = limit.expression
    try:
        return "argmax_argmin" if int(value.name) == 1 else "top_k"
    except (ValueError, AttributeError):
        return "top_k"


def shape_of(sql: str) -> dict[str, Any]:
    """Coarse shape of one SQL string. Raises ``sqlglot.errors.SqlglotError`` if unparseable."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    selects = list(tree.find_all(exp.Select))
    if not selects:
        raise sqlglot.errors.ParseError("no SELECT statement")
    outer = _outer_select(tree)

    table_refs = len(list(tree.find_all(exp.Table)))
    hops = max(0, table_refs - 1)

    aggs = {_agg_name(n) for t in AGG_TYPES for n in tree.find_all(t)}
    aggregation = "none" if not aggs else next(iter(aggs)) if len(aggs) == 1 else "multiple"

    extras: set[str] = set()
    if len(selects) > 1 and not isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        extras.add("subquery")
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)) or tree.find(exp.Union, exp.Intersect, exp.Except):
        extras.add("set_operation")
    if tree.find(exp.Exists) or any(n.args.get("query") is not None for n in tree.find_all(exp.In)):
        extras.add("existence")
    if tree.find(exp.Not, exp.NEQ, exp.Except):
        extras.add("negation")
    if tree.find(exp.Div):
        extras.add("ratio")
    if any(_has_agg(n.left) and _has_agg(n.right) for n in tree.find_all(exp.Sub)):
        extras.add("difference_of_aggregates")
    if tree.find(exp.Having):
        extras.add("having")
    conditional_literals: set[str] = set()
    for t in AGG_TYPES:
        for agg in tree.find_all(t):
            if agg.find(exp.Case, exp.If):
                extras.add("conditional_aggregate")
                conditional_literals |= {lit.name for lit in agg.find_all(exp.Literal) if lit.is_string}
    if len(conditional_literals) >= 2:
        extras.add("comparison_two_entities")  # approximate: two conditional aggregates on different literals
    if any(s.args.get("distinct") is not None for s in selects):
        extras.add("distinct")
    for sub in tree.find_all(exp.Subquery):
        inner = sub.find(exp.Select)
        if inner is not None and isinstance(sub.parent, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)) and (
                inner.find(exp.Max, exp.Min) or _ordering(inner) == "argmax_argmin"):
            extras.add("argmax_via_subquery")
    if tree.find(exp.Window):
        extras.add("window")
    if tree.find(exp.With):
        extras.add("cte")

    filters: set[str] = set()
    for where in tree.find_all(exp.Where, exp.Having):
        if where.find(exp.EQ):
            filters.add("equality")
        if where.find(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between):
            filters.add("range")
        if where.find(exp.Like):
            filters.add("like")
        if where.find(exp.Is):
            filters.add("null_test")
        if any(n.args.get("query") is None for n in where.find_all(exp.In)):
            filters.add("in_list")
    if _DATE_FN.search(sql):
        filters.add("date_function")

    return {
        "hops": hops, "hops_bucket": str(hops) if hops < 4 else "4+",
        "aggregation": aggregation, "aggregations": sorted(aggs),
        "grouping": tree.find(exp.Group) is not None,
        "ordering": _ordering(outer),
        "extras": sorted(extras), "filters": sorted(filters),
        "return_columns": len(outer.expressions) if outer is not None else 0,
    }


def coarse_key(shape: dict[str, Any]) -> str:
    extras = "+".join(shape["extras"]) or "-"
    return f"hops={shape['hops_bucket']}|agg={shape['aggregation']}|group={int(shape['grouping'])}|order={shape['ordering']}|extras={extras}"


def analyse(path: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    db_ids = Counter(r["db_id"] for r in rows)
    # Independence from the evaluation set and identity of the subset are asserted, not assumed.
    if EVAL_DB in db_ids:
        raise SystemExit(f"STOP: {EVAL_DB} appears in this file ({db_ids[EVAL_DB]} rows); it is not independent of the evaluation set")
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"STOP: expected the {EXPECTED_ROWS}-row filtered subset, found {len(rows)} rows")

    shapes, failures = [], []
    for i, r in enumerate(rows):
        try:
            shapes.append(shape_of(r["SQL"]))
        except Exception as e:  # noqa: BLE001 - every unparsed query is counted and listed, never dropped silently
            failures.append({"index": i, "db_id": r["db_id"], "error": f"{type(e).__name__}: {e}"[:160], "sql": r["SQL"][:300]})

    n = len(shapes)

    def dist(values: list[Any]) -> list[dict[str, Any]]:
        return [{"value": v, "count": c, "share": round(c / n, 4)} for v, c in Counter(values).most_common()]

    keys = Counter(coarse_key(s) for s in shapes)
    return {
        "source": "birdsql/bird23-train-filtered (CC BY-SA 4.0)",
        "scope_note": f"measured on the {EXPECTED_ROWS} questions BIRD filtered for quality, not the full 9,428-question train split",
        "n_rows": len(rows), "n_databases": len(db_ids), "eval_db_present": False,
        "n_classified": n, "n_unclassified": len(failures),
        "unclassified_share": round(len(failures) / len(rows), 4),
        "hops": dist([s["hops_bucket"] for s in shapes]),
        "aggregation": dist([s["aggregation"] for s in shapes]),
        "grouping": dist([s["grouping"] for s in shapes]),
        "ordering": dist([s["ordering"] for s in shapes]),
        "extras_each": [{"value": t, "count": c, "share": round(c / n, 4)}
                        for t, c in Counter(t for s in shapes for t in s["extras"]).most_common()],
        "extras_none": {"count": sum(1 for s in shapes if not s["extras"]), "share": round(sum(1 for s in shapes if not s["extras"]) / n, 4)},
        "filters_each": [{"value": t, "count": c, "share": round(c / n, 4)}
                         for t, c in Counter(t for s in shapes for t in s["filters"]).most_common()],
        "return_columns": dist([min(s["return_columns"], 4) for s in shapes]),
        "distinct_coarse_shapes": len(keys),
        "coarse_shapes_seen_once": sum(1 for c in keys.values() if c == 1),
        "coarse_shapes": [{"shape": k, "count": c, "share": round(c / n, 4)} for k, c in keys.most_common()],
        "failures": failures,
    }


def main() -> None:
    report = analyse(sys.argv[1])
    Path(sys.argv[2]).expanduser().write_text(json.dumps(report, indent=1))
    brief = {k: report[k] for k in ("source", "scope_note", "n_rows", "n_databases", "eval_db_present",
                                    "n_classified", "n_unclassified", "unclassified_share", "hops", "aggregation",
                                    "grouping", "ordering", "extras_each", "extras_none", "filters_each",
                                    "return_columns", "distinct_coarse_shapes", "coarse_shapes_seen_once")}
    brief["top_coarse_shapes"] = report["coarse_shapes"][:15]
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
