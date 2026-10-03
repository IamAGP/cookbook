"""E2 zero-shot baseline on Fireworks: the Tinker experiment with only the sampler swapped.

The peer's harness (tinker-cookbook @ ed9e7f7,
``tinker_cookbook/recipes/bird_graph_rl/baseline_eval.py``) is imported and run UNCHANGED:
system prompt, renderer (``qwen3_8_xhigh_reasoning``), run_cypher tool, rollout loop and
limits, cost meter, scorer (``normalize_value`` / ``row_key`` / ``score_rows``), results.jsonl
and summary.json all come from her module, not from a copy.

The one substitution: inside ``baseline_eval.main()`` the name ``tinker.ServiceClient`` is
patched to :class:`FireworksServiceShim`, whose ``create_sampling_client_async(base_model=...)``
returns a Fireworks serverless *base-model* sampler. That sampler implements the same
``tinker.SamplingClient.sample_async(prompt, num_samples, sampling_params)`` contract her
``TinkerTokenCompleter`` calls, and is token-in / token-out, so the prompt tokens are the ones
her renderer builds.

Before running anything the script refuses to start unless her harness file is byte-identical
to the pinned commit, and it records provenance in ``<log_path>/fireworks_meta.json``.

Usage (same config keys as her harness; env_file points at HER .env for BIRD_NEO4J_*):
    PYTHONPATH=<tinker-cookbook> .venv/bin/python bird_graph_rl/baseline_eval_fireworks.py \\
        reference_uri=s3://<bucket>/codebase_community/gold/reference_answers.json \\
        env_file=<tinker-cookbook>/.env log_path=~/bird_rl_runs/e2_fireworks_smoke \\
        limit=10 budget_usd=1.5
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file, reference_uri, tinker_cookbook_dir  # noqa: E402
HER_REPO = tinker_cookbook_dir()
HER_FILE = "tinker_cookbook/recipes/bird_graph_rl/baseline_eval.py"
PINNED_COMMIT = "d600861"
MY_ENV = fireworks_env_file()

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
# Same Hugging Face checkpoint on both platforms (firectl model get: Hugging Face Url
# https://huggingface.co/Qwen/Qwen3.8-27B). Only the serving name differs.
FIREWORKS_MODEL = {"Qwen/Qwen3.8-27B": "accounts/fireworks/models/qwen3p8-27b"}

sys.path.insert(0, str(HER_REPO))

import chz  # noqa: E402
from fireworks.training.sdk import FiretitanServiceClient  # noqa: E402

from tinker_cookbook import tokenizer_utils  # noqa: E402
from tinker_cookbook.recipes.bird_graph_rl import baseline_eval as be  # noqa: E402

logger = logging.getLogger("bird_fireworks")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(HER_REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_harness_pinned() -> dict[str, str]:
    """Refuse to run unless her harness on disk is exactly the pinned commit's blob."""
    pinned_blob = _git("rev-parse", f"{PINNED_COMMIT}:{HER_FILE}")
    disk_blob = _git("hash-object", HER_FILE)
    if pinned_blob != disk_blob:
        raise SystemExit(f"harness drift: {HER_FILE} on disk ({disk_blob}) != {PINNED_COMMIT} ({pinned_blob})")
    sha256 = hashlib.sha256((HER_REPO / HER_FILE).read_bytes()).hexdigest()
    return {"harness_commit": PINNED_COMMIT, "harness_blob": pinned_blob, "harness_sha256": sha256,
            "her_repo_head": _git("rev-parse", "--short", "HEAD")}


class FireworksServiceShim:
    """Stands in for ``tinker.ServiceClient`` inside her ``main()``.

    Only ``create_sampling_client_async(base_model=...)`` is used by her harness.
    """

    live: list["FireworksServiceShim"] = []

    def __init__(self, user_metadata: dict[str, str] | None = None, **_: Any) -> None:
        self.user_metadata = user_metadata or {}
        self.service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                              base_url=SERVERLESS_URL)
        self.fireworks_model = ""
        FireworksServiceShim.live.append(self)

    async def create_sampling_client_async(self, base_model: str | None = None, model_path: str | None = None, **_: Any) -> Any:
        if model_path is not None or base_model is None:
            # Her harness can now sample a trained checkpoint. On Fireworks a checkpoint is bound to
            # the session or deployment that produced it, so this needs its own path. Not built.
            raise NotImplementedError("sampling a trained checkpoint through this wrapper is not built yet")
        self.fireworks_model = FIREWORKS_MODEL[base_model]
        # The same call her main() makes for the renderer; the sampler needs it to turn the
        # renderer's integer stop token ids into the string stops the completions API accepts.
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        # Binds the serverless session to the base model. The adapter is never trained and
        # never sampled: sampling below uses the SDK's base-only route (no checkpoint segment).
        await asyncio.to_thread(self.service.create_lora_training_client,
                                base_model=self.fireworks_model, rank=8)
        sampler = await asyncio.to_thread(self.service.create_sampling_client,
                                          base_model=self.fireworks_model, tokenizer=tokenizer)
        logger.info("fireworks base-only sampler: model=%s session=%s",
                    getattr(sampler.deployment_sampler, "model", "?"), self.service.training_session_id)
        return sampler

    def close(self) -> None:
        try:
            self.service.close()
        except Exception as e:  # noqa: BLE001 - teardown must not mask the run's result
            logger.warning("service close failed: %s", e)


def write_fireworks_meta(cfg: be.Config, provenance: dict[str, str], extra: dict[str, Any]) -> None:
    log_dir = Path(os.path.expanduser(cfg.log_path))
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "framework": "fireworks-serverless-training",
        "sampling_base_url": SERVERLESS_URL,
        "fireworks_model": FIREWORKS_MODEL.get(cfg.base_model),
        "versions": {p: md.version(p) for p in ("fireworks-ai", "tinker", "transformers")},
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **provenance, **extra,
    }
    path = log_dir / "fireworks_meta.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **meta}, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(MY_ENV)  # FIREWORKS_API_KEY; her main() then loads cfg.env_file for BIRD_NEO4J_*
    provenance = verify_harness_pinned()
    cfg = chz.entrypoint(be.Config)
    if cfg.base_model not in FIREWORKS_MODEL:
        raise SystemExit(f"no Fireworks mapping for {cfg.base_model}")
    write_fireworks_meta(cfg, provenance, {"status": "started"})
    try:
        with mock.patch.object(be.tinker, "ServiceClient", FireworksServiceShim):
            asyncio.run(be.main(cfg))
        status = "finished"
    except BaseException as e:
        status = f"aborted: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        sessions = [s.service.training_session_id for s in FireworksServiceShim.live]
        for s in FireworksServiceShim.live:
            s.close()
        write_fireworks_meta(cfg, provenance, {"status": status, "training_sessions": sessions})


if __name__ == "__main__":
    main()
