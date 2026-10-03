"""How much of a rollout is the local database? Replay a run's own queries and time them.

Replays every tool query recorded in a baseline run (display cap, as the model saw it) and the
final query's uncapped re-execution used for scoring, one at a time, against the live graph.
Reads only. Serial on purpose: it measures database seconds per query without contention, the
floor that concurrency can only add to. Between rollouts it checks for other transactions so a
contaminated measurement is visible rather than silent.

    PYTHONPATH=<tinker-cookbook> python bird_graph_rl/db_time_measure.py <results.jsonl> <out.json>
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file, reference_uri, tinker_cookbook_dir  # noqa: E402
HER_REPO = tinker_cookbook_dir()
sys.path.insert(0, str(HER_REPO))
from tinker_cookbook.recipes.bird_graph_rl.baseline_eval import (  # noqa: E402
    SCORE_ROW_CAP, STRING_LITERALS, TOOL_ROW_CAP, WRITE_KEYWORDS, CypherTool,
)


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


async def main() -> None:
    rows = [json.loads(line) for line in Path(sys.argv[1]).expanduser().open()]
    load_dotenv(HER_REPO / ".env", override=True)
    driver = AsyncGraphDatabase.driver(os.environ["BIRD_NEO4J_URI"],
                                       auth=(os.environ["BIRD_NEO4J_USER"], os.environ["BIRD_NEO4J_PASSWORD"]))
    tool = CypherTool(driver, "neo4j")

    async def timed(query: str, cap: int) -> tuple[float, bool]:
        t0 = time.perf_counter()
        try:
            await tool.execute(query, cap)
            return time.perf_counter() - t0, True
        except Exception:  # noqa: BLE001 - a query that errored in the run errors here too; its time still counts
            return time.perf_counter() - t0, False

    async def others() -> int:
        async with driver.session(database="neo4j") as s:
            res = await s.run("SHOW TRANSACTIONS YIELD currentQuery RETURN currentQuery")
            return sum(1 for r in [x async for x in res] if not r["currentQuery"].startswith("SHOW TRANSACTIONS"))

    per_rollout, per_query, rescore, interference = [], [], [], 0
    replay_mismatch = 0
    t_start = time.time()
    for i, r in enumerate(rows):
        tool_s = 0.0
        for entry in r.get("queries", []):
            q = entry["query"]
            if WRITE_KEYWORDS.search(STRING_LITERALS.sub("''", q)):
                continue  # blocked client-side in the run, never reached the database
            dt, ok = await timed(q, TOOL_ROW_CAP + 1)
            replay_mismatch += int(ok != bool(entry["ok"]))
            per_query.append(dt)
            tool_s += dt
        rs = 0.0
        if r.get("final_query"):
            rs, _ = await timed(r["final_query"], SCORE_ROW_CAP)
            rescore.append(rs)
        per_rollout.append({"question_id": r["question_id"], "tool_db_s": round(tool_s, 4), "rescore_db_s": round(rs, 4),
                            "n_queries": len(r.get("queries", [])), "rollout_wall_s_in_run": r.get("seconds")})
        if i % 10 == 0:
            interference += int(await others() > 0)
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(rows)} rollouts replayed, {len(per_query)} queries, "
                  f"{time.time() - t_start:.0f}s elapsed", flush=True)
    await driver.close()

    tool_tot = [p["tool_db_s"] for p in per_rollout]
    both = [p["tool_db_s"] + p["rescore_db_s"] for p in per_rollout]
    wall = [p["rollout_wall_s_in_run"] for p in per_rollout if p["rollout_wall_s_in_run"]]
    report = {
        "source_run": sys.argv[1], "n_rollouts": len(rows), "n_tool_queries": len(per_query),
        "serial": True, "interference_checks_with_other_transactions": interference,
        "replay_ok_mismatches_vs_run": replay_mismatch,
        "per_query_s": {"mean": round(st.mean(per_query), 4), "median": round(st.median(per_query), 4),
                        "p90": round(pct(per_query, 0.9), 4), "p99": round(pct(per_query, 0.99), 4), "max": round(max(per_query), 3)},
        "rescore_query_s": {"mean": round(st.mean(rescore), 4), "median": round(st.median(rescore), 4),
                            "p90": round(pct(rescore, 0.9), 4), "max": round(max(rescore), 3)},
        "db_s_per_rollout_tool_only": {"mean": round(st.mean(tool_tot), 3), "median": round(st.median(tool_tot), 3),
                                       "p90": round(pct(tool_tot, 0.9), 3), "max": round(max(tool_tot), 2)},
        "db_s_per_rollout_tool_plus_rescore": {"mean": round(st.mean(both), 3), "median": round(st.median(both), 3),
                                               "p90": round(pct(both, 0.9), 3), "max": round(max(both), 2)},
        "total_db_s": round(sum(both), 1),
        "rollout_wall_s_in_source_run": {"mean": round(st.mean(wall), 2), "median": round(st.median(wall), 2)},
        "db_share_of_rollout_wall_time_in_source_run": round(sum(both) / sum(wall), 4),
        "per_rollout": per_rollout,
    }
    Path(sys.argv[2]).expanduser().write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_rollout"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
