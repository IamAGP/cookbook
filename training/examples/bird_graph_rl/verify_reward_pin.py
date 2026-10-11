"""Re-score both E2 runs with the peer's pinned reward (import, not reimplementation).

Gates on the git blob of her reward.py, re-executes every stored final_query (reads only),
and reports mean reward and the structural counts so they can be diffed against her numbers.
"""
from __future__ import annotations

import asyncio, json, os, subprocess, sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file, reference_uri, tinker_cookbook_dir  # noqa: E402
HER_REPO = tinker_cookbook_dir()
REWARD_FILE = "tinker_cookbook/recipes/bird_graph_rl/reward.py"
PINNED_REWARD_BLOB = "e326e77114d181f12f7fd95db6278b3c4cdb1cc2"
REFERENCE_URI = reference_uri()
sys.path.insert(0, str(HER_REPO))

disk = subprocess.run(["git", "-C", str(HER_REPO), "hash-object", REWARD_FILE],
                      check=True, capture_output=True, text=True).stdout.strip()
if disk != PINNED_REWARD_BLOB:
    raise SystemExit(f"reward drift: disk {disk} != pinned {PINNED_REWARD_BLOB}")

from tinker_cookbook.recipes.bird_graph_rl.baseline_eval import SCORE_ROW_CAP  # noqa: E402
from tinker_cookbook.recipes.bird_graph_rl.reward import load_references_exact, partial_credit  # noqa: E402


async def rescore(path: str, refs: dict[int, dict[str, Any]], driver: Any) -> dict[int, dict[str, float]]:
    sem, out = asyncio.Semaphore(8), {}

    async def one(rec: dict[str, Any]) -> None:
        rows: list[list[Any]] = []
        if rec.get("final_query"):
            async with sem:
                async def work(tx: Any) -> list[list[Any]]:
                    acc = []
                    async for r in await tx.run(rec["final_query"]):
                        acc.append(list(r.values()))
                        if len(acc) >= SCORE_ROW_CAP:
                            break
                    return acc
                async with driver.session(database="neo4j") as s:
                    rows = await s.execute_read(work)
            out[rec["question_id"]] = partial_credit(rows, refs[rec["question_id"]])
        else:
            out[rec["question_id"]] = {"reward": 0.0, "strict": 0.0, "approximate": 0.0}

    await asyncio.gather(*(one(json.loads(l)) for l in Path(path).open()))
    return out


async def main() -> None:
    load_dotenv(HER_REPO / ".env", override=True)
    refs = {r["question_id"]: r for r in load_references_exact(REFERENCE_URI)}
    driver = AsyncGraphDatabase.driver(os.environ["BIRD_NEO4J_URI"],
                                       auth=(os.environ["BIRD_NEO4J_USER"], os.environ["BIRD_NEO4J_PASSWORD"]))
    a = await rescore(sys.argv[1], refs, driver)
    if len(sys.argv) > 3 and sys.argv[3] == "--single":
        rs = {json.loads(l)["question_id"]: json.loads(l) for l in Path(sys.argv[1]).open()}
        fails = [q for q in a if a[q]["strict"] == 0]
        shape = [q for q in fails if rs[q]["lenient"] == 1]
        print(json.dumps({"n": len(a), "mean_reward": round(sum(v["reward"] for v in a.values()) / len(a), 3), "mean_strict": round(sum(v["strict"] for v in a.values()) / len(a), 3), "failures": len(fails), "failures_with_partial_credit": sum(1 for q in fails if a[q]["reward"] > 0), "lenient_only_failures": len(shape), "lenient_only_with_partial_credit": sum(1 for q in shape if a[q]["reward"] > 0)}, indent=1))
        await driver.close()
        return
    b = await rescore(sys.argv[2], refs, driver)
    await driver.close()
    qs = sorted(set(a) & set(b))
    never = [q for q in qs if a[q]["strict"] == 0 and b[q]["strict"] == 0]
    rescued = [q for q in never if max(a[q]["reward"], b[q]["reward"]) > 0]
    print(json.dumps({
        "reward_blob": disk, "n": len(qs),
        "truncated_refs_remaining": sum(1 for r in refs.values() if r.get("truncated")),
        "approximate_scores": sum(int(x[q]["approximate"]) for x in (a, b) for q in qs),
        "mean_reward": {"tinker": round(sum(a[q]["reward"] for q in qs) / len(qs), 3),
                        "fireworks": round(sum(b[q]["reward"] for q in qs) / len(qs), 3)},
        "mean_strict": {"tinker": round(sum(a[q]["strict"] for q in qs) / len(qs), 3),
                        "fireworks": round(sum(b[q]["strict"] for q in qs) / len(qs), 3)},
        "never_solved": len(never), "rescued": len(rescued), "still_zero": len(never) - len(rescued),
        "spread_binary": sum(1 for q in qs if a[q]["strict"] != b[q]["strict"]),
        "spread_partial": sum(1 for q in qs if abs(a[q]["reward"] - b[q]["reward"]) > 1e-9),
        "rescued_ids": rescued}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
