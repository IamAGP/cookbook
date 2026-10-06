"""Do hint terms come from the question? Generated hints against human-written ones.

For every hint with clauses of the form "X refers to Y", ask whether some X shares no word at
all with the question. A crude test (stop list, plural stripping) for a large difference.

    python bird_graph_rl/hint_overlap.py <out.json> name=path[:question_field:hint_field[:split]] ...
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

STOP = {"the", "a", "an", "of", "each", "is", "are", "to", "in", "for", "when", "present", "otherwise"}


def words(s: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower())} - STOP


def measure(pairs: list[tuple[str, str | None]]) -> dict:
    hinted = with_clause = orphan_hints = clauses = orphan_clauses = 0
    terms: Counter[str] = Counter()
    for question, hint in pairs:
        hint = (hint or "").strip()
        if not hint:
            continue
        hinted += 1
        qw, bad, had = words(question), 0, 0
        for clause in hint.split(";"):
            m = re.match(r"\s*(.+?) refers to (.+)", clause.strip(), re.I | re.S)
            if not m:
                continue
            had += 1
            clauses += 1
            tw = words(m.group(1))
            if tw and not (tw & qw):
                bad += 1
                orphan_clauses += 1
                terms[m.group(1).strip().lower()] += 1
        with_clause += had > 0
        orphan_hints += bad > 0
    return {"questions": len(pairs), "hinted": hinted, "hint_share": round(hinted / len(pairs), 4),
            "hints_with_refers_to": with_clause, "hints_with_a_term_absent_from_question": orphan_hints,
            "share_of_hints": round(orphan_hints / max(with_clause, 1), 4),
            "clauses": clauses, "clauses_absent_from_question": orphan_clauses,
            "share_of_clauses": round(orphan_clauses / max(clauses, 1), 4),
            "most_common_absent_terms": terms.most_common(10)}


def main() -> None:
    out = {}
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path, qf, hf = parts[0], (parts[1] if len(parts) > 1 else "question"), (parts[2] if len(parts) > 2 else "hint")
        split = parts[3] if len(parts) > 3 else None
        rows = [json.loads(line) for line in Path(path).expanduser().open()]
        if split:
            rows = [r for r in rows if r.get("split") == split]
        out[name] = measure([(r[qf], r.get(hf)) for r in rows])
    Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "most_common_absent_terms"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
