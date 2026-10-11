"""Independent check of the Tinker side's partial-credit measurement.

Re-executes every stored ``final_query`` from both E2 runs against the live graph and scores
row-level F1 against the reference rows, capped so that F1 == 1.0 only where her strict scorer
also says correct. No sampling, no API spend: database reads only.

Her reported numbers (2026-10-03): mean reward 0.639 Tinker / 0.642 Fireworks; 52 never-solved
at strict; 13 of those with non-zero partial credit in at least one run; 39 still zero;
spread on 42 of 186 binary and 47 of 186 with partial credit.

Usage:
    PYTHONPATH=<tinker-cookbook> python bird_graph_rl/verify_partial_credit.py \\
        <tinker results.jsonl> <fireworks results.jsonl> <out.json>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file, reference_uri, tinker_cookbook_dir  # noqa: E402
HER_REPO = tinker_cookbook_dir()
sys.path.insert(0, str(HER_REPO))

from tinker_cookbook.recipes.bird_graph_rl.baseline_eval import (  # noqa: E402
    SCORE_ROW_CAP, load_references, row_key, score_rows,
)

REFERENCE_URI = reference_uri()
GOLD_SET = "corrected_20251106"
CONCURRENCY = 8


def f1(model_rows: list[list[Any]], ref: dict[str, Any]) -> float:
    """Row-level F1 on normalized row keys, as a multiset. 1.0 only when strict agrees."""
    strict = score_rows(model_rows, ref)["correct"] == 1.0
    if strict:
        return 1.0
    ref_rows = ref.get("rows") or []
    if not model_rows or not ref_rows:
        return 0.0
    got, want = Counter(map(row_key, model_rows)), Counter(map(row_key, ref_rows))
    overlap = sum((got & want).values())
    if not overlap:
        return 0.0
    # For truncated references only the first 200 rows are stored; recall uses the true count.
    n_ref = int(ref.get("n_rows", len(ref_rows)))
    precision = overlap / sum(got.values())
    recall = overlap / n_ref
    value = 2 * precision * recall / (precision + recall)
    return min(value, 0.999)  # never award 1.0 where strict said wrong


async def rescore(results_path: str, refs: dict[int, dict[str, Any]], driver: Any) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in Path(results_path).open()]
    sem = asyncio.Semaphore(CONCURRENCY)
    out: dict[int, dict[str, Any]] = {}

    async def one(rec: dict[str, Any]) -> None:
        qid = rec["question_id"]
        ref = refs[qid]
        query = rec.get("final_query") or ""
        model_rows: list[list[Any]] = []
        error = ""
        if query:
            async with sem:
                try:
                    async def work(tx: Any) -> list[list[Any]]:
                        res = await tx.run(query)
                        acc: list[list[Any]] = []
                        async for record in res:
                            acc.append(list(record.values()))
                            if len(acc) >= SCORE_ROW_CAP:
                                break
                        return acc

                    async with driver.session(database="neo4j") as session:
                        model_rows = await session.execute_read(work)
                except Exception as e:  # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"[:200]
        out[qid] = {"reward": f1(model_rows, ref) if query and not error else 0.0,
                    "strict_stored": float(rec.get("correct", 0.0)),
                    "strict_now": score_rows(model_rows, ref)["correct"] if query and not error else 0.0,
                    "n_rows": len(model_rows), "had_query": bool(query), "error": error}

    await asyncio.gather(*(one(r) for r in rows))
    return out


async def main() -> None:
    a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    load_dotenv(HER_REPO / ".env", override=True)
    refs = {r["question_id"]: r for r in load_references(REFERENCE_URI, GOLD_SET)}
    driver = AsyncGraphDatabase.driver(os.environ["BIRD_NEO4J_URI"],
                                       auth=(os.environ["BIRD_NEO4J_USER"], os.environ["BIRD_NEO4J_PASSWORD"]))
    await driver.verify_connectivity()
    t0 = time.time()
    a = await rescore(a_path, refs, driver)
    print(f"[{time.strftime('%H:%M:%S')}] tinker rescored: {len(a)} ({time.time()-t0:.0f}s)", flush=True)
    b = await rescore(b_path, refs, driver)
    print(f"[{time.strftime('%H:%M:%S')}] fireworks rescored: {len(b)} ({time.time()-t0:.0f}s)", flush=True)
    await driver.close()

    common = sorted(set(a) & set(b))
    never = [q for q in common if a[q]["strict_stored"] == 0 and b[q]["strict_stored"] == 0]
    partial_rescued = [q for q in never if max(a[q]["reward"], b[q]["reward"]) > 0]
    spread_binary = [q for q in common if a[q]["strict_stored"] != b[q]["strict_stored"]]
    spread_partial = [q for q in common if abs(a[q]["reward"] - b[q]["reward"]) > 1e-9]
    drift = [q for q in common for side in (a, b) if side[q]["had_query"]
             and side[q]["strict_now"] != side[q]["strict_stored"]]
    report = {
        "n_common": len(common),
        "mean_reward": {"tinker": round(sum(a[q]["reward"] for q in common) / len(common), 3),
                        "fireworks": round(sum(b[q]["reward"] for q in common) / len(common), 3)},
        "mean_strict_stored": {"tinker": round(sum(a[q]["strict_stored"] for q in common) / len(common), 3),
                               "fireworks": round(sum(b[q]["strict_stored"] for q in common) / len(common), 3)},
        "never_solved": len(never),
        "never_solved_with_partial_credit": len(partial_rescued),
        "never_solved_still_zero": len(never) - len(partial_rescued),
        "spread_binary": len(spread_binary), "spread_partial": len(spread_partial),
        "rescore_errors": sum(1 for q in common for side in (a, b) if side[q]["error"]),
        "strict_drift_vs_stored": len(set(drift)),
        "partial_rescued_ids": partial_rescued,
    }
    Path(out_path).write_text(json.dumps({"report": report, "per_question": {str(q): {"tinker": a[q], "fireworks": b[q]} for q in common}}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
