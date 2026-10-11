"""Verbatim transcripts of chosen questions on Fireworks, for the write-up's trajectory viewer.

Runs the peer's ``eval_transcripts.py`` (tinker-cookbook, commit 79c3cd7) unchanged, with
``tinker.ServiceClient`` replaced so that every model it asks for is served the way this side's
evaluation served it: each model path is a saved training state, loaded into its own fresh
serverless session and sampled as a snapshot (``baseline_eval_fireworks.FireworksServiceShim``).

    PYTHONPATH=<tinker-cookbook> python bird_graph_rl/record_transcripts.py \\
        base_model=Qwen/Qwen3.8-27B models=untrained=<state>,trained=<state> human_ids=551,613,531 \\
        reference_uri=... env_file=<tinker-cookbook>/.env out_path=~/bird_rl_runs/blog_transcripts_fireworks.json
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_eval_fireworks as bef  # noqa: E402  (loads the Fireworks key and pins the harness)

import chz  # noqa: E402
import tinker  # noqa: E402
from tinker_cookbook.recipes.bird_graph_rl import eval_transcripts as et  # noqa: E402


class PerModelService:
    """One fresh Fireworks session per requested model, closed at exit."""

    shims: list[bef.FireworksServiceShim] = []

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def create_sampling_client_async(self, model_path: str | None = None, base_model: str | None = None) -> Any:
        shim = bef.FireworksServiceShim()
        type(self).shims.append(shim)
        return await shim.create_sampling_client_async(base_model=base_model, model_path=model_path)


def main() -> None:
    bef.load_dotenv(bef.MY_ENV)  # FIREWORKS_API_KEY; the peer's script loads her .env for the graph
    bef.verify_harness_pinned()
    cfg = chz.entrypoint(et.Config)
    bef.FireworksServiceShim.base_model = cfg.base_model
    try:
        with mock.patch.object(tinker, "ServiceClient", PerModelService), mock.patch.object(et.tinker, "ServiceClient", PerModelService):
            asyncio.run(et.main(cfg))
    finally:
        for shim in PerModelService.shims:
            print("session", shim.service.training_session_id, "snapshot", shim.sampler_path, flush=True)
            shim.close()


if __name__ == "__main__":
    main()
