"""Independent cross-check of the shape mapper, plus anchor types in equality filters.

The mapper reads a parsed syntax tree. This script recomputes five of its labels from the raw
SQL text with regular expressions, a different method with different failure modes, and reports
where the two disagree. Zero parse failures does not show the labels are right; agreement between
two independent methods is evidence that they are.

    python bird_graph_rl/sql_shapes_crosscheck.py <train.jsonl> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sql_shapes import shape_of  # noqa: E402

STRINGS = re.compile(r"'(?:[^']|'')*'")


def by_regex(sql: str) -> dict[str, object]:
    s = STRINGS.sub("''", sql)  # literals must not be mistaken for keywords
    limit = re.search(r"\bLIMIT\s+(\d+)", s, re.I)
    order = re.search(r"\bORDER\s+BY\b", s, re.I) is not None
    return {
        "grouping": re.search(r"\bGROUP\s+BY\b", s, re.I) is not None,
        "having": re.search(r"\bHAVING\b", s, re.I) is not None,
        "any_limit_1_with_order": bool(order and limit and any(int(m) == 1 for m in re.findall(r"\bLIMIT\s+(\d+)", s, re.I))),
        "n_selects": len(re.findall(r"\bSELECT\b", s, re.I)),
        "n_join_keywords": len(re.findall(r"\bJOIN\b", s, re.I)),
    }


def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).open() if line.strip()]
    disagree: dict[str, list[int]] = {k: [] for k in ("grouping", "having", "argmax_anywhere", "n_selects", "joins_vs_hops")}
    anchors: Counter[str] = Counter()
    for i, r in enumerate(rows):
        sql = r["SQL"]
        shape, rx = shape_of(sql), by_regex(sql)
        tree = sqlglot.parse_one(sql, read="sqlite")
        if shape["grouping"] != rx["grouping"]:
            disagree["grouping"].append(i)
        if ("having" in shape["extras"]) != rx["having"]:
            disagree["having"].append(i)
        tree_argmax = any(s.args.get("order") is not None and s.args.get("limit") is not None
                          and s.args["limit"].expression.name == "1" for s in tree.find_all(exp.Select))
        if tree_argmax != rx["any_limit_1_with_order"]:
            disagree["argmax_anywhere"].append(i)
        if len(list(tree.find_all(exp.Select))) != rx["n_selects"]:
            disagree["n_selects"].append(i)
        # Explicit JOIN keywords are a lower bound on hops (comma joins and subquery tables add more).
        if rx["n_join_keywords"] > shape["hops"]:
            disagree["joins_vs_hops"].append(i)
        # What do equality filters anchor on? Literal type and whether the column looks like an id.
        for where in tree.find_all(exp.Where):
            for eq in where.find_all(exp.EQ):
                col, lit = (eq.left, eq.right) if isinstance(eq.right, exp.Literal) else (eq.right, eq.left)
                if not isinstance(lit, exp.Literal) or not isinstance(col, exp.Column):
                    continue
                looks_id = re.search(r"(^|_|\s)id$|id$", col.name, re.I) is not None
                anchors[("string" if lit.is_string else "number") + ("_on_id_column" if looks_id else "_on_other_column")] += 1
    n, total = len(rows), sum(anchors.values())
    report = {
        "n_rows": n,
        "disagreements": {k: {"count": len(v), "share": round(len(v) / n, 4), "first_indices": v[:5]} for k, v in disagree.items()},
        "equality_anchor_literals": {"total": total, **{k: {"count": c, "share": round(c / total, 4)} for k, c in anchors.most_common()}},
    }
    Path(sys.argv[2]).expanduser().write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
