"""Paired, per-question comparison of two E2 baseline runs (Tinker vs Fireworks).

Both runs are single samples at temperature 1.0 over the same 186 questions, so the right
test is paired: McNemar's exact test on the questions where the two runs disagree.
Aggregates alone hide whether the same questions pass.

Usage:
    python bird_graph_rl/compare_runs.py <tinker results.jsonl> <fireworks results.jsonl> [out.json]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load(path: str) -> dict[int, dict[str, Any]]:
    return {r["question_id"]: r for r in map(json.loads, Path(path).open())}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on discordant pairs b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    mean = lambda k: round(sum(float(r.get(k, 0)) for r in rows) / n, 3) if n else 0.0  # noqa: E731
    return {
        "n": n, "strict": mean("correct"), "lenient": mean("lenient"), "no_query": mean("no_query"),
        "turns": mean("turns"),
        "turn_cap_hits": sum(1 for r in rows if r.get("stop_reason") == "max_turns"),
        "query_errors": sum(int(r.get("n_query_errors", 0)) for r in rows),
        "harness_errors": sum(1 for r in rows if r.get("error") or r.get("rescore_error")),
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in rows),
        "sampled_tokens": sum(int(r.get("sampled_tokens", 0)) for r in rows),
        "est_usd_upper": round(sum(float(r.get("est_usd_upper", 0)) for r in rows), 2),
    }


def compare(a_path: str, b_path: str) -> dict[str, Any]:
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    out: dict[str, Any] = {
        "a": a_path, "b": b_path, "n_common": len(common),
        "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)),
        "overall": {"a": summarize([a[q] for q in common]), "b": summarize([b[q] for q in common])},
        "by_difficulty": {},
    }
    by_diff: dict[str, list[int]] = defaultdict(list)
    for q in common:
        by_diff[a[q].get("difficulty") or "?"].append(q)
    for d, qs in sorted(by_diff.items()):
        out["by_difficulty"][d] = {"a": summarize([a[q] for q in qs]), "b": summarize([b[q] for q in qs])}

    for metric in ("correct", "lenient"):
        cells = Counter((int(a[q][metric]), int(b[q][metric])) for q in common)
        only_a, only_b = cells[(1, 0)], cells[(0, 1)]
        out[f"paired_{metric}"] = {
            "both_right": cells[(1, 1)], "both_wrong": cells[(0, 0)],
            "a_only": only_a, "b_only": only_b,
            "agreement": round((cells[(1, 1)] + cells[(0, 0)]) / len(common), 3),
            "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
            "a_only_ids": [q for q in common if a[q][metric] == 1 and b[q][metric] == 0],
            "b_only_ids": [q for q in common if a[q][metric] == 0 and b[q][metric] == 1],
        }
    return out


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    result = compare(a_path, b_path)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
    ov = result["overall"]
    print(f"common questions: {result['n_common']}")
    print(f"{'':14s}{'A (tinker)':>12s}{'B (fireworks)':>15s}")
    for k in ("strict", "lenient", "no_query", "turns", "turn_cap_hits", "query_errors",
              "harness_errors", "prompt_tokens", "sampled_tokens", "est_usd_upper"):
        print(f"{k:14s}{ov['a'][k]!s:>12s}{ov['b'][k]!s:>15s}")
    for d, v in result["by_difficulty"].items():
        print(f"{d:14s} n={v['a']['n']:<4d} strict {v['a']['strict']:.3f} vs {v['b']['strict']:.3f}"
              f" | lenient {v['a']['lenient']:.3f} vs {v['b']['lenient']:.3f}")
    for metric in ("correct", "lenient"):
        p = result[f"paired_{metric}"]
        print(f"paired {metric}: both right {p['both_right']}, both wrong {p['both_wrong']}, "
              f"A only {p['a_only']}, B only {p['b_only']}, agreement {p['agreement']}, "
              f"McNemar exact p={p['mcnemar_exact_p']}")


if __name__ == "__main__":
    main()
