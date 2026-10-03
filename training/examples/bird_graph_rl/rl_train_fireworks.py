"""RL on Fireworks: the Tinker-side training entry point with only the service client swapped.

Runs ``tinker_cookbook.recipes.bird_graph_rl.train.cli_main`` from the peer's repo UNCHANGED.
Environment, reward, dataset builder, rollout loop, loss and optimiser calls are her code,
imported and gated on git blob hashes. The single substitution: ``tinker.ServiceClient`` is
replaced by :class:`FireworksServiceShim`.

What the shim has to bridge (read from ``tinker_cookbook/rl/train.py`` and the Fireworks SDK):
- ``training_client.save_weights_and_get_sampling_client_async()`` is unsupported on Fireworks
  by design, and ``training_client.create_sampling_client(path)`` needs a managed deployment
  that serverless lacks. Both become ``save_weights_for_sampler`` followed by
  ``service.create_sampling_client(model_path=..., tokenizer=...)``.
- Serverless checkpoint names must be 17 characters or fewer; the cookbook's longer names are
  shortened deterministically.
- Model names: Tinker's ``Qwen/...`` ids map to ``accounts/fireworks/models/...``.

It also adds what a billed run needs and her loop does not have: a token meter that converts
to dollars at list prices and stops the run once ``FW_BUDGET_USD`` is crossed, writing one
timestamped line per training call to ``<log_path>/fw_cost_meter.jsonl``.

Dedicated (hourly) training is built but gated: it refuses to start unless
``FW_ALLOW_DEDICATED=1`` and an explicit deadline, inactivity timeout and deployment id are
set, and it replaces the token meter with a wall-clock meter (dollars per hour since the
resources were requested). It has never been run against real hardware. Resuming a run from a
saved state is still refused.

    FW_BUDGET_USD=3 PYTHONPATH=<tinker-cookbook> .venv/bin/python \\
        bird_graph_rl/rl_train_fireworks.py model_name=Qwen/Qwen3.8-27B \\
        instances_path=... log_path=... max_steps=2 batch_size=4 group_size=4 \\
        env_file=<tinker-cookbook>/.env behavior_if_log_dir_exists=delete
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fireworks_env_file, reference_uri, tinker_cookbook_dir  # noqa: E402
HER_REPO = tinker_cookbook_dir()
RECIPE = "tinker_cookbook/recipes/bird_graph_rl"
PINNED_COMMIT = "d0ab818"
PINNED_BLOBS = {
    "rl_env.py": "44f4db865e7cfc6cb61c3e731018f45d68cdd706",
    "train.py": "e6d5b6048a4f69dd87b0f4151783af2e4b3b1275",
    "reward.py": "e326e77114d181f12f7fd95db6278b3c4cdb1cc2",
    "baseline_eval.py": "d17b8ffba28e57feb7396704159ffe60ee0cd1a3",
}
MY_ENV = fireworks_env_file()
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
MAX_CHECKPOINT_NAME = 17  # Fireworks docs, serverless "Save and resume training checkpoints"

# Tinker model id -> (Fireworks model id, training surface). Surfaces from the Fireworks
# registry and cost catalog, read 2026-10-03.
FIREWORKS_MODEL: dict[str, tuple[str, str]] = {
    "Qwen/Qwen3.8-27B": ("accounts/fireworks/models/qwen3p8-27b", "serverless"),
    "Qwen/Qwen3.5-9B": ("accounts/fireworks/models/qwen3p5-9b", "dedicated"),
}
# Dedicated shapes and what they bill. GPU counts from `firectl training-shape list` and the
# linked RL sampler deployment shape; $13.00 per B200-hour from the pricing page (2026-10-03).
DEDICATED: dict[str, dict[str, Any]] = {
    "accounts/fireworks/models/qwen3p5-9b": {
        "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p5-9b-65k-lora",
        "trainer_gpus": 2, "sampler_gpus": 1, "usd_per_gpu_hour": 13.0,
    },
}
# USD per million tokens, Fireworks docs cost catalog generated 2026-09-30. Prompt tokens are
# priced uncached, so the meter is an upper bound.
PRICES_PER_M: dict[str, dict[str, float]] = {
    "accounts/fireworks/models/qwen3p8-27b": {"prefill": 1.86, "sample": 5.595, "train": 4.103},
}

logger = logging.getLogger("bird_fireworks_rl")


class BudgetExceeded(RuntimeError):
    """Raised inside the training loop once the metered estimate crosses the budget."""


def short_name(name: str) -> str:
    """Deterministically fit a checkpoint name into the serverless limit."""
    if len(name) <= MAX_CHECKPOINT_NAME:
        return name
    words = [w for w in re.split(r"[_\-]+", name) if w]
    digits = "".join(w for w in words if w.isdigit())
    initials = "".join(w[0] for w in words if not w.isdigit())
    candidate = f"{initials}{digits}"
    if candidate and len(candidate) <= MAX_CHECKPOINT_NAME:
        return candidate
    return "c" + hashlib.sha1(name.encode()).hexdigest()[: MAX_CHECKPOINT_NAME - 1]


class CostMeter:
    """Running upper-bound dollar estimate for one run, with a hard stop."""

    def __init__(self, prices: dict[str, float], budget_usd: float | None, log_path: Path | None) -> None:
        self.prices, self.budget_usd, self.log_path = prices, budget_usd, log_path
        self.prompt_tokens = self.sampled_tokens = self.train_tokens = 0
        self.sample_calls = self.train_calls = self.datums = 0

    @property
    def usd(self) -> float:
        return (self.prompt_tokens * self.prices["prefill"] + self.sampled_tokens * self.prices["sample"]
                + self.train_tokens * self.prices["train"]) / 1e6

    def add_sample(self, prompt_tokens: int, sampled_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.sampled_tokens += sampled_tokens
        self.sample_calls += 1

    def add_train(self, n_datums: int, tokens: int) -> None:
        self.train_tokens += tokens
        self.train_calls += 1
        self.datums += n_datums
        self._write({"event": "forward_backward", "datums": n_datums, "tokens": tokens})

    def check(self) -> None:
        if self.budget_usd is not None and self.usd >= self.budget_usd:
            self._write({"event": "budget_stop"})
            raise BudgetExceeded(f"metered estimate ${self.usd:.2f} reached budget ${self.budget_usd:.2f}")

    def snapshot(self) -> dict[str, Any]:
        return {"usd_upper": round(self.usd, 4), "prompt_tokens": self.prompt_tokens,
                "sampled_tokens": self.sampled_tokens, "train_tokens": self.train_tokens,
                "sample_calls": self.sample_calls, "train_calls": self.train_calls, "datums": self.datums}

    def _write(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event, **self.snapshot()}
        with self.log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")


class WallClockMeter(CostMeter):
    """Hourly billing: dollars accrue with time since the resources were requested, whether or
    not anything is running. Tokens and datums are still counted, as measurements."""

    def __init__(self, usd_per_hour: float, budget_usd: float | None, deadline_s: float,
                 log_path: Path | None, clock: Callable[[], float] = time.time) -> None:
        super().__init__({"prefill": 0.0, "sample": 0.0, "train": 0.0}, budget_usd, log_path)
        self.usd_per_hour, self.deadline_s, self._clock = usd_per_hour, deadline_s, clock
        self.t0 = clock()

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self.t0

    @property
    def usd(self) -> float:
        return self.elapsed_s / 3600.0 * self.usd_per_hour

    def check(self) -> None:
        if self.elapsed_s >= self.deadline_s:
            self._write({"event": "deadline_stop"})
            raise BudgetExceeded(f"wall-clock deadline of {self.deadline_s:.0f}s reached after {self.elapsed_s:.0f}s")
        super().check()

    def snapshot(self) -> dict[str, Any]:
        return {**super().snapshot(), "elapsed_s": round(self.elapsed_s, 1), "usd_per_hour": self.usd_per_hour}


class MeteredSampler:
    """Wraps a Fireworks sampling client; counts tokens and refuses new calls past the budget."""

    def __init__(self, inner: Any, meter: CostMeter) -> None:
        self._inner, self._meter = inner, meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any, *args: Any, **kwargs: Any) -> Any:
        self._meter.check()
        response = await self._inner.sample_async(prompt, num_samples, sampling_params, *args, **kwargs)
        sampled = sum(len(seq.tokens) for seq in response.sequences)
        self._meter.add_sample(prompt.length * num_samples, sampled)
        return response


class TrainingClientProxy:
    """A Fireworks training client that answers the two calls the cookbook loop makes which
    Fireworks does not support, shortens checkpoint names, and meters training tokens."""

    def __init__(self, inner: Any, service: Any, tokenizer: Any, meter: CostMeter, keep_samplers: int | None = 2) -> None:
        self._inner, self._service, self._tokenizer, self._meter = inner, service, tokenizer, meter
        self._keep = keep_samplers
        self._samplers: list[Any] = []
        self._snapshots = 0
        self.name_map: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _short(self, name: str) -> str:
        short = short_name(name)
        if short != name:
            self.name_map[name] = short
            logger.info("checkpoint name %r shortened to %r", name, short)
        return short

    def _open(self, model_path: str) -> MeteredSampler:
        sampler = self._service.create_sampling_client(model_path=model_path, tokenizer=self._tokenizer)
        if self._keep is None:  # dedicated: one shared deployment sampler, closed by the service
            return MeteredSampler(sampler, self._meter)
        self._samplers.append(sampler)
        while len(self._samplers) > self._keep:  # older snapshots are no longer sampled from
            old = self._samplers.pop(0)
            try:
                old.close()
            except Exception as e:  # noqa: BLE001 - closing a stale sampler must not stop training
                logger.warning("closing stale sampler failed: %s", e)
        return MeteredSampler(sampler, self._meter)

    def create_sampling_client(self, model_path: str, retry_config: Any = None) -> MeteredSampler:
        return self._open(model_path)

    async def save_weights_and_get_sampling_client_async(self, name: str | None = None, retry_config: Any = None) -> MeteredSampler:
        self._snapshots += 1
        future = await self._inner.save_weights_for_sampler_async(self._short(name or f"s{self._snapshots:06d}"))
        return self._open((await future.result_async()).path)

    async def save_weights_for_sampler_async(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.save_weights_for_sampler_async(self._short(name), *args, **kwargs)

    async def save_state_async(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.save_state_async(self._short(name), *args, **kwargs)

    async def forward_backward_async(self, data: list[Any], *args: Any, **kwargs: Any) -> Any:
        self._meter.check()
        self._meter.add_train(len(data), sum(d.model_input.length for d in data))
        return await self._inner.forward_backward_async(data, *args, **kwargs)

    def close_samplers(self) -> None:
        for s in self._samplers:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("closing sampler failed: %s", e)
        self._samplers.clear()


def dedicated_settings(env: dict[str, str]) -> dict[str, Any]:
    """Read the dedicated-run guards. Every one is mandatory: nothing hourly starts on defaults."""
    if env.get("FW_ALLOW_DEDICATED") != "1":
        raise NotImplementedError(
            "this model trains only on Fireworks dedicated (hourly) GPUs. Refusing to provision: "
            "set FW_ALLOW_DEDICATED=1 only with the owner's approval and a written preflight.")
    missing = [k for k in ("FW_DEPLOYMENT_ID", "FW_DEADLINE_MIN", "FW_INACTIVITY_MIN") if not env.get(k)]
    if missing:
        raise SystemExit(f"dedicated run needs {', '.join(missing)}")
    return {"deployment_id": env["FW_DEPLOYMENT_ID"], "deadline_s": float(env["FW_DEADLINE_MIN"]) * 60,
            "inactivity_min": float(env["FW_INACTIVITY_MIN"]),
            "pending_timeout_s": float(env.get("FW_PENDING_MIN", "20")) * 60,
            "ready_timeout_s": float(env.get("FW_READY_MIN", "30")) * 60}


def _default_service_factory(spec: dict[str, Any]) -> Any:
    from fireworks.training.sdk import FiretitanServiceClient

    if spec["surface"] == "serverless":
        return FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=SERVERLESS_URL)
    d, shape = spec["dedicated"], DEDICATED[spec["fireworks_model"]]
    return FiretitanServiceClient.from_firetitan_config(
        api_key=os.environ["FIREWORKS_API_KEY"], user_metadata=spec.get("user_metadata"),
        base_model=spec["fireworks_model"], tokenizer_model=spec["hf_model"],
        lora_rank=spec["rank"], seed=spec["seed"], train_mlp=spec["train_mlp"],
        train_attn=spec["train_attn"], train_unembed=spec["train_unembed"],
        training_shape_id=shape["training_shape_id"], deployment_id=d["deployment_id"],
        inactivity_timeout=dt.timedelta(minutes=d["inactivity_min"]),
        trainer_pending_timeout_s=d["pending_timeout_s"], trainer_timeout_s=d["ready_timeout_s"],
        deployment_timeout_s=d["ready_timeout_s"],
        cleanup_trainer_on_close=True, cleanup_deployment_on_close="delete",
        display_name=d["deployment_id"])


SCALE_TO_ZERO_WINDOW_S = 300  # the API's minimum; its default is one hour and only applies at min replicas 0


def _default_deployment_hardener(deployment_id: str) -> dict[str, Any]:
    """Let an abandoned sampler deployment stop billing by itself.

    The SDK creates the RL sampler deployment with min replicas = max replicas = 1, so it never
    scales to zero, and deployments have no server-side expiry (``expireTime`` is deprecated).
    If this machine died, the deployment would bill until someone deleted it. Setting min
    replicas to 0 with the shortest scale-to-zero window bounds that to a few idle minutes.
    """
    from fireworks.training.sdk import DeploymentManager

    mgr = DeploymentManager(api_key=os.environ["FIREWORKS_API_KEY"])
    body = {"minReplicaCount": 0, "autoscalingPolicy": {"scaleToZeroWindow": f"{SCALE_TO_ZERO_WINDOW_S}s"}}
    mgr.update(deployment_id, body, ["min_replica_count", "autoscaling_policy.scale_to_zero_window"])
    return body


def _default_tokenizer_loader(model_name: str) -> Any:
    from tinker_cookbook import tokenizer_utils

    return tokenizer_utils.get_tokenizer(model_name)


class FireworksServiceShim:
    """Stands in for ``tinker.ServiceClient`` inside ``tinker_cookbook.rl.train.main``."""

    service_factory: Callable[[dict[str, Any]], Any] = staticmethod(_default_service_factory)
    resources_log: Path | None = None
    environ: dict[str, str] = os.environ  # type: ignore[assignment]
    tokenizer_loader: Callable[[str], Any] = staticmethod(_default_tokenizer_loader)
    deployment_hardener: Callable[[str], dict[str, Any]] = staticmethod(_default_deployment_hardener)
    budget_usd: float | None = None
    meter_log: Path | None = None
    live: list["FireworksServiceShim"] = []

    def __init__(self, base_url: str | None = None, user_metadata: dict[str, str] | None = None, **_: Any) -> None:
        self.user_metadata = user_metadata or {}
        self.service: Any = None  # created once the model, and so the billing surface, is known
        self.surface = ""
        self.training: TrainingClientProxy | None = None
        self.meter: CostMeter | None = None
        self.lora_request: dict[str, Any] = {}
        self.hardening: dict[str, Any] = {}
        self.fireworks_model = ""
        FireworksServiceShim.live.append(self)

    async def create_lora_training_client_async(
        self, base_model: str, rank: int = 32, seed: int | None = None, train_mlp: bool = True,
        train_attn: bool = True, train_unembed: bool = True, user_metadata: dict[str, str] | None = None,
    ) -> TrainingClientProxy:
        if base_model not in FIREWORKS_MODEL:
            raise KeyError(f"no Fireworks mapping for {base_model}")
        self.fireworks_model, self.surface = FIREWORKS_MODEL[base_model]
        dedicated = dedicated_settings(dict(type(self).environ)) if self.surface == "dedicated" else None
        self.lora_request = {"base_model": self.fireworks_model, "rank": rank, "seed": seed,
                             "train_mlp": train_mlp, "train_attn": train_attn, "train_unembed": train_unembed}
        logger.info("fireworks lora request (%s): %s", self.surface, self.lora_request)
        if dedicated is not None:
            shape = DEDICATED[self.fireworks_model]
            rate = (shape["trainer_gpus"] + shape["sampler_gpus"]) * shape["usd_per_gpu_hour"]
            # The clock starts before anything is requested: provisioning is budgeted as billed.
            self.meter = WallClockMeter(rate, type(self).budget_usd, dedicated["deadline_s"], type(self).meter_log)
        else:
            self.meter = CostMeter(PRICES_PER_M[self.fireworks_model], type(self).budget_usd, type(self).meter_log)
        self.service = type(self).service_factory({
            "surface": self.surface, "fireworks_model": self.fireworks_model, "hf_model": base_model,
            "dedicated": dedicated, "user_metadata": user_metadata, **{k: v for k, v in self.lora_request.items() if k != "base_model"}})
        tokenizer = type(self).tokenizer_loader(base_model)
        inner = await asyncio.to_thread(
            self.service.create_lora_training_client, base_model=self.fireworks_model, rank=rank, seed=seed,
            train_mlp=train_mlp, train_attn=train_attn, train_unembed=train_unembed, user_metadata=user_metadata)
        self._record_resources(dedicated)
        if dedicated is not None:
            deployment_id = getattr(self.service, "deployment_id", None) or dedicated["deployment_id"]
            try:
                self.hardening = type(self).deployment_hardener(deployment_id)
                logger.info("deployment %s set to scale to zero when idle: %s", deployment_id, self.hardening)
            except Exception as e:
                # An hourly resource that cannot stop itself is not worth the risk: tear down now.
                self.close()
                raise SystemExit(f"could not bound the deployment's idle billing ({type(e).__name__}: {e}); "
                                 "resources were deleted and the run was not started") from e
        self.training = TrainingClientProxy(inner, self.service, tokenizer, self.meter,
                                            keep_samplers=None if dedicated is not None else 2)
        return self.training

    def _record_resources(self, dedicated: dict[str, Any] | None) -> None:
        """Write what is now billing, for the independent watchdog and for the post-run check."""
        path = type(self).resources_log
        if path is None:
            return
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "surface": self.surface,
                  "fireworks_model": self.fireworks_model,
                  "training_session_id": getattr(self.service, "training_session_id", None)}
        if dedicated is not None:
            record.update({"trainer_job_id": getattr(self.service, "trainer_job_id", None),
                           "deployment_id": getattr(self.service, "deployment_id", None) or dedicated["deployment_id"],
                           "requested_epoch": self.meter.t0, "deadline_epoch": self.meter.t0 + dedicated["deadline_s"],
                           "usd_per_hour": self.meter.usd_per_hour})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))

    def create_sampling_client(self, base_model: str | None = None, model_path: str | None = None, **_: Any) -> Any:
        """KL-reference path: base-only sampling on the session (validated in the E2 baseline)."""
        if self.service is None:
            raise RuntimeError("no Fireworks service yet: the training client must be created first")
        fw_model = FIREWORKS_MODEL[base_model][0] if base_model else None
        tokenizer = type(self).tokenizer_loader(base_model) if base_model else None
        return self.service.create_sampling_client(model_path=model_path, base_model=fw_model, tokenizer=tokenizer)

    def create_rest_client(self) -> Any:
        return self.service.create_rest_client()

    async def create_training_client_from_state_async(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError("starting from a saved state on Fireworks is not built or tested yet")

    async def create_training_client_from_state_with_optimizer_async(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError("resuming a run on Fireworks is not built or tested yet")

    def close(self) -> None:
        if self.training is not None:
            self.training.close_samplers()
        if self.service is None:
            return
        try:
            self.service.close()
        except Exception as e:  # noqa: BLE001 - teardown must not mask the run's outcome
            logger.warning("service close failed: %s", e)


def verify_pins() -> dict[str, str]:
    """Refuse to run unless every shared file on disk is the pinned blob."""
    found = {}
    for name, blob in PINNED_BLOBS.items():
        disk = subprocess.run(["git", "-C", str(HER_REPO), "hash-object", f"{RECIPE}/{name}"],
                              check=True, capture_output=True, text=True).stdout.strip()
        if disk != blob:
            raise SystemExit(f"pin drift: {name} on disk {disk[:12]} != pinned {blob[:12]} ({PINNED_COMMIT})")
        found[name] = disk
    return found


REQUIRED_DB_VARS = ("BIRD_NEO4J_URI", "BIRD_NEO4J_USER", "BIRD_NEO4J_PASSWORD")


def local_preflight(cfg: Any, env_values: dict[str, str | None]) -> list[str]:
    """Everything that can fail on this machine, checked before any Fireworks resource exists.

    Her entry point loads ``cfg.env_file`` with override and reads the database settings from
    it only when the first environment is built, which is after the training client has been
    created. On hourly hardware that order would bill for a typo. Returns a list of problems;
    empty means go.
    """
    problems = []
    missing = [k for k in REQUIRED_DB_VARS if not env_values.get(k)]
    if missing:
        problems.append(f"{cfg.env_file} lacks {', '.join(missing)}")
    path = Path(os.path.expanduser(cfg.instances_path)) if cfg.instances_path else None
    if path is None or not path.exists():
        problems.append(f"instances_path does not exist: {cfg.instances_path!r}")
    else:
        n = 0
        with path.open() as f:
            for line in f:
                if line.strip() and json.loads(line).get("split") == cfg.train_split:
                    n += 1
        if n < cfg.batch_size:
            problems.append(f"{n} instances in split {cfg.train_split!r}, fewer than batch_size {cfg.batch_size}")
    return problems


def main() -> None:
    from dotenv import dotenv_values, load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(MY_ENV)
    pins = verify_pins()
    sys.path.insert(0, str(HER_REPO))

    import chz
    import tinker
    from tinker_cookbook.recipes.bird_graph_rl import train as her_train

    cfg = chz.entrypoint(her_train.CLIConfig)
    if not cfg.log_path:
        raise SystemExit("log_path is required: the cost meter and run metadata are written there")
    if "FW_BUDGET_USD" not in os.environ:
        raise SystemExit("FW_BUDGET_USD is required: no billed run without a cost stop")
    log_dir = Path(os.path.expanduser(cfg.log_path))
    FireworksServiceShim.budget_usd = float(os.environ["FW_BUDGET_USD"])
    FireworksServiceShim.meter_log = log_dir / "fw_cost_meter.jsonl"
    FireworksServiceShim.resources_log = log_dir / "fw_resources.json"
    problems = local_preflight(cfg, dict(dotenv_values(cfg.env_file)))
    if problems:
        raise SystemExit("local preflight failed, nothing was requested from Fireworks: " + "; ".join(problems))

    status = "started"
    try:
        with mock.patch.object(tinker, "ServiceClient", FireworksServiceShim):
            asyncio.run(her_train.cli_main(cfg))
        status = "finished"
    except BudgetExceeded as e:
        status = f"stopped: {e}"
        logger.error(status)
    except BaseException as e:
        status = f"aborted: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        runs = []
        for shim in FireworksServiceShim.live:
            runs.append({"session": getattr(shim.service, "training_session_id", None), "surface": shim.surface,
                         "trainer_job_id": getattr(shim.service, "trainer_job_id", None) if shim.surface == "dedicated" else None,
                         "deployment_id": getattr(shim.service, "deployment_id", None) if shim.surface == "dedicated" else None,
                         "lora_request": shim.lora_request,
                         "meter": shim.meter.snapshot() if shim.meter else None,
                         "checkpoint_name_map": shim.training.name_map if shim.training else {}})
            shim.close()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "fireworks_meta.json").write_text(json.dumps({
            "framework": "fireworks-training-api", "status": status,
            "pinned_commit": PINNED_COMMIT, "pinned_blobs": pins,
            "budget_usd": FireworksServiceShim.budget_usd, "runs": runs,
            "versions": {p: md.version(p) for p in ("fireworks-ai", "tinker", "transformers")},
            "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=1))


if __name__ == "__main__":
    main()
