"""Offline tests for the Fireworks RL shim. No network, no database, no spend.

    PYTHONPATH=<tinker-cookbook> .venv/bin/python -m unittest bird_graph_rl/rl_train_fireworks_test.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_train_fireworks as fw  # noqa: E402

sys.path.insert(0, str(fw.HER_REPO))


class FakeFuture:
    def __init__(self, path: str) -> None:
        self.path = path

    async def result_async(self) -> SimpleNamespace:
        return SimpleNamespace(path=self.path)


class FakeTraining:
    def __init__(self) -> None:
        self.saved_sampler: list[str] = []
        self.saved_state: list[str] = []
        self.fb_calls = 0

    async def save_weights_for_sampler_async(self, name: str, *a: object, **k: object) -> FakeFuture:
        self.saved_sampler.append(name)
        return FakeFuture(f"acct/run-1/{name}")

    async def save_state_async(self, name: str, *a: object, **k: object) -> FakeFuture:
        self.saved_state.append(name)
        return FakeFuture(f"acct/run-1/{name}")

    async def forward_backward_async(self, data: list, *a: object, **k: object) -> str:
        self.fb_calls += 1
        return "fb"

    async def optim_step_async(self, *a: object, **k: object) -> str:
        return "optim"  # reached through __getattr__ delegation


class FakeSampler:
    def __init__(self, model_path: str | None) -> None:
        self.model_path, self.closed = model_path, False

    async def sample_async(self, prompt: object, num_samples: int, sampling_params: object) -> SimpleNamespace:
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2, 3]) for _ in range(num_samples)])

    def close(self) -> None:
        self.closed = True


class FakeService:
    def __init__(self) -> None:
        self.lora_kwargs: dict = {}
        self.training = FakeTraining()
        self.samplers: list[FakeSampler] = []
        self.sampler_kwargs: list[dict] = []
        self.closed = False
        self.training_session_id = "ts-fake"
        self.trainer_job_id = "job-fake"
        self.deployment_id = "dep-fake"

    def create_lora_training_client(self, **kwargs: object) -> FakeTraining:
        self.lora_kwargs = dict(kwargs)
        return self.training

    def create_sampling_client(self, model_path: str | None = None, base_model: str | None = None, tokenizer: object = None) -> FakeSampler:
        self.sampler_kwargs.append({"model_path": model_path, "base_model": base_model, "tokenizer": tokenizer})
        self.samplers.append(FakeSampler(model_path))
        return self.samplers[-1]

    def close(self) -> None:
        self.closed = True


def datum(n: int) -> SimpleNamespace:
    return SimpleNamespace(model_input=SimpleNamespace(length=n))


class ShimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeService()
        self.specs: list[dict] = []
        self.hardened: list[str] = []
        self.harden_fails = False

        def factory(spec: dict) -> FakeService:
            self.specs.append(spec)
            return self.service

        self._orig_create = fw.FireworksServiceShim.create_lora_training_client_async
        self.patches = [
            mock.patch.object(fw.FireworksServiceShim, "service_factory", staticmethod(factory)),
            mock.patch.object(fw.FireworksServiceShim, "environ", {}),
            mock.patch.object(fw.FireworksServiceShim, "deployment_hardener", staticmethod(self._harden)),
            mock.patch.object(fw.FireworksServiceShim, "resources_log", None),
            mock.patch.object(fw.FireworksServiceShim, "tokenizer_loader", staticmethod(lambda name: f"tok:{name}")),
            mock.patch.object(fw.FireworksServiceShim, "budget_usd", None),
            mock.patch.object(fw.FireworksServiceShim, "meter_log", None),
        ]
        for p in self.patches:
            p.start()

    def _harden(self, deployment_id: str) -> dict:
        if self.harden_fails:
            raise RuntimeError("PATCH rejected")
        self.hardened.append(deployment_id)
        return {"minReplicaCount": 0}

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()
        fw.FireworksServiceShim.create_lora_training_client_async = self._orig_create
        fw.FireworksServiceShim.live.clear()

    def run_async(self, coro: object) -> object:
        return asyncio.run(coro)  # type: ignore[arg-type]

    def test_short_name_fits_limit_and_stays_distinct(self) -> None:
        self.assertEqual(fw.short_name("000010"), "000010")
        names = {fw.short_name(f"rolling_checkpoint_{i:06d}") for i in range(50)}
        names |= {fw.short_name(f"periodic_checkpoint_{i:06d}") for i in range(50)}
        self.assertEqual(len(names), 100)
        self.assertTrue(all(len(n) <= fw.MAX_CHECKPOINT_NAME for n in names))
        self.assertEqual(fw.short_name("rolling_checkpoint_000010"), fw.short_name("rolling_checkpoint_000010"))
        self.assertLessEqual(len(fw.short_name("x" * 60)), fw.MAX_CHECKPOINT_NAME)

    def test_peer_pin_of_train_unembed_reaches_fireworks(self) -> None:
        import tinker
        from tinker_cookbook.recipes.bird_graph_rl import train as her_train

        with mock.patch.object(tinker, "ServiceClient", fw.FireworksServiceShim):
            her_train._pin_lora_layers(False)  # her code, patching whatever tinker.ServiceClient is
            shim = tinker.ServiceClient(base_url=None, user_metadata={"recipe": "x"})
            self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B", rank=32, user_metadata={}))
        self.assertIs(self.service.lora_kwargs["train_unembed"], False)
        self.assertEqual(self.service.lora_kwargs["base_model"], "accounts/fireworks/models/qwen3p8-27b")
        self.assertEqual(self.service.lora_kwargs["rank"], 32)
        self.assertTrue(self.service.lora_kwargs["train_mlp"] and self.service.lora_kwargs["train_attn"])
        self.assertIs(shim.lora_request["train_unembed"], False)

    def test_default_without_pin_is_sdk_default(self) -> None:
        shim = fw.FireworksServiceShim()
        self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))
        self.assertIs(self.service.lora_kwargs["train_unembed"], True)

    def test_dedicated_model_is_refused_without_explicit_approval_flag(self) -> None:
        shim = fw.FireworksServiceShim()
        with self.assertRaises(NotImplementedError):
            self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.5-9B"))
        self.assertEqual(self.specs, [])  # nothing was requested from Fireworks
        self.assertEqual(self.service.lora_kwargs, {})

    def test_dedicated_refuses_to_start_on_defaults(self) -> None:
        with mock.patch.object(fw.FireworksServiceShim, "environ", {"FW_ALLOW_DEDICATED": "1", "FW_DEPLOYMENT_ID": "bird-probe"}):
            shim = fw.FireworksServiceShim()
            with self.assertRaises(SystemExit) as ctx:
                self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.5-9B"))
        self.assertIn("FW_DEADLINE_MIN", str(ctx.exception))
        self.assertIn("FW_INACTIVITY_MIN", str(ctx.exception))
        self.assertEqual(self.specs, [])

    def test_dedicated_with_all_guards_requests_the_pinned_shape_and_records_resources(self) -> None:
        import tinker
        from tinker_cookbook.recipes.bird_graph_rl import train as her_train

        env = {"FW_ALLOW_DEDICATED": "1", "FW_DEPLOYMENT_ID": "bird-probe", "FW_DEADLINE_MIN": "20", "FW_INACTIVITY_MIN": "10"}
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "fw_resources.json"
            with mock.patch.object(fw.FireworksServiceShim, "environ", env), \
                 mock.patch.object(fw.FireworksServiceShim, "resources_log", res), \
                 mock.patch.object(fw.FireworksServiceShim, "budget_usd", 15.0), \
                 mock.patch.object(tinker, "ServiceClient", fw.FireworksServiceShim):
                her_train._pin_lora_layers(False)
                shim = tinker.ServiceClient(user_metadata={})
                tc = self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.5-9B", rank=32))
                self.run_async(tc.save_weights_and_get_sampling_client_async())
                self.run_async(tc.save_weights_and_get_sampling_client_async())
                self.run_async(tc.save_weights_and_get_sampling_client_async())
            record = json.loads(res.read_text())
        spec = self.specs[0]
        self.assertEqual((spec["surface"], spec["fireworks_model"], spec["hf_model"]),
                         ("dedicated", "accounts/fireworks/models/qwen3p5-9b", "Qwen/Qwen3.5-9B"))
        self.assertIs(spec["train_unembed"], False)
        self.assertEqual(spec["rank"], 32)
        self.assertEqual(spec["dedicated"], {"deployment_id": "bird-probe", "deadline_s": 1200.0, "inactivity_min": 10.0,
                                             "pending_timeout_s": 1200.0, "ready_timeout_s": 1800.0})
        self.assertIsInstance(shim.meter, fw.WallClockMeter)
        shape = fw.DEDICATED["accounts/fireworks/models/qwen3p5-9b"]
        self.assertEqual(shim.meter.usd_per_hour, (shape["trainer_gpus"] + shape["sampler_gpus"]) * shape["usd_per_gpu_hour"])
        self.assertEqual((record["trainer_job_id"], record["deployment_id"], record["surface"]), ("job-fake", "dep-fake", "dedicated"))
        self.assertAlmostEqual(record["deadline_epoch"] - record["requested_epoch"], 1200.0, places=3)
        # The deployment's sampler is shared on dedicated, so the proxy must never close one.
        self.assertEqual([s.closed for s in self.service.samplers], [False, False, False])

    def test_dedicated_deployment_is_set_to_scale_to_zero_or_the_run_is_torn_down(self) -> None:
        env = {"FW_ALLOW_DEDICATED": "1", "FW_DEPLOYMENT_ID": "bird-probe", "FW_DEADLINE_MIN": "20", "FW_INACTIVITY_MIN": "10"}
        with mock.patch.object(fw.FireworksServiceShim, "environ", env):
            shim = fw.FireworksServiceShim()
            self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.5-9B"))
            self.assertEqual(self.hardened, ["dep-fake"])
            self.assertEqual(shim.hardening, {"minReplicaCount": 0})
            self.harden_fails = True
            self.service.closed = False
            failing = fw.FireworksServiceShim()
            with self.assertRaises(SystemExit) as ctx:
                self.run_async(failing.create_lora_training_client_async("Qwen/Qwen3.5-9B"))
        self.assertIn("idle billing", str(ctx.exception))
        self.assertTrue(self.service.closed)       # resources released rather than left unprotected
        self.assertIsNone(failing.training)

    def test_serverless_run_does_not_touch_any_deployment(self) -> None:
        shim = fw.FireworksServiceShim()
        self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))
        self.assertEqual(self.hardened, [])

    def test_wall_clock_meter_bills_time_and_stops_at_deadline_or_budget(self) -> None:
        now = [1000.0]
        meter = fw.WallClockMeter(39.0, budget_usd=None, deadline_s=900.0, log_path=None, clock=lambda: now[0])
        now[0] += 600
        self.assertAlmostEqual(meter.usd, 600 / 3600 * 39.0, places=9)
        meter.check()  # inside the deadline, no budget
        now[0] += 300
        with self.assertRaises(fw.BudgetExceeded) as ctx:
            meter.check()
        self.assertIn("deadline", str(ctx.exception))
        capped = fw.WallClockMeter(39.0, budget_usd=5.0, deadline_s=3600.0, log_path=None, clock=lambda: now[0])
        capped.check()
        now[0] += 5.0 / 39.0 * 3600 + 1
        with self.assertRaises(fw.BudgetExceeded) as ctx:
            capped.check()
        self.assertIn("budget", str(ctx.exception))

    def test_unsupported_calls_are_bridged(self) -> None:
        shim = fw.FireworksServiceShim()
        tc = self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))

        async def go() -> None:
            a = await tc.save_weights_and_get_sampling_client_async()
            b = await tc.save_weights_and_get_sampling_client_async()
            c = tc.create_sampling_client("acct/run-1/000010")
            self.assertIsInstance(a, fw.MeteredSampler)
            self.assertNotEqual(a.model_path, b.model_path)
            self.assertEqual(c.model_path, "acct/run-1/000010")
            self.assertEqual(await tc.optim_step_async(), "optim")

        self.run_async(go())
        self.assertEqual(self.service.training.saved_sampler, ["s000001", "s000002"])
        self.assertEqual([k["tokenizer"] for k in self.service.sampler_kwargs], ["tok:Qwen/Qwen3.8-27B"] * 3)
        self.assertEqual([s.closed for s in self.service.samplers], [True, False, False])  # keeps the last 2

    def test_checkpoint_names_are_shortened_on_save(self) -> None:
        shim = fw.FireworksServiceShim()
        tc = self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))

        async def go() -> None:
            await tc.save_state_async("periodic_checkpoint_000020")
            await tc.save_weights_for_sampler_async("000020")

        self.run_async(go())
        self.assertEqual(self.service.training.saved_state, ["pc000020"])
        self.assertEqual(self.service.training.saved_sampler, ["000020"])
        self.assertEqual(tc.name_map, {"periodic_checkpoint_000020": "pc000020"})

    def test_meter_counts_tokens_and_stops_at_budget(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "fw_cost_meter.jsonl"
            prices = fw.PRICES_PER_M["accounts/fireworks/models/qwen3p8-27b"]
            expected = (2000 * prices["prefill"] + 6 * prices["sample"] + 1200 * prices["train"]) / 1e6
            # Budget sits just under the cost of one sample call plus one training call, so the
            # first training call is admitted and everything after it is refused.
            with mock.patch.object(fw.FireworksServiceShim, "budget_usd", expected * 0.999), \
                 mock.patch.object(fw.FireworksServiceShim, "meter_log", log):
                shim = fw.FireworksServiceShim()
                tc = self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))

                async def go() -> None:
                    sampler = await tc.save_weights_and_get_sampling_client_async()
                    await sampler.sample_async(SimpleNamespace(length=1000), 2, None)
                    await tc.forward_backward_async([datum(500), datum(700)])
                    with self.assertRaises(fw.BudgetExceeded):
                        await tc.forward_backward_async([datum(10)])
                    with self.assertRaises(fw.BudgetExceeded):
                        await sampler.sample_async(SimpleNamespace(length=10), 1, None)

                self.run_async(go())
            m = shim.meter.snapshot()
            self.assertEqual((m["prompt_tokens"], m["sampled_tokens"], m["train_tokens"], m["datums"]), (2000, 6, 1200, 2))
            self.assertAlmostEqual(shim.meter.usd, expected, places=9)
            self.assertEqual(self.service.training.fb_calls, 1)  # the second call never reached Fireworks
            events = [json.loads(line)["event"] for line in log.open()]
            self.assertEqual(events, ["forward_backward", "budget_stop", "budget_stop"])

    def test_local_preflight_catches_what_would_fail_after_hardware_is_billing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inst = Path(d) / "instances.jsonl"
            inst.write_text("".join(json.dumps({"split": s}) + "\n" for s in ["train"] * 3 + ["heldout_instance"] * 5))
            good_env = {"BIRD_NEO4J_URI": "bolt://x", "BIRD_NEO4J_USER": "u", "BIRD_NEO4J_PASSWORD": "p"}
            cfg = SimpleNamespace(env_file=".env", instances_path=str(inst), train_split="train", batch_size=2)
            self.assertEqual(fw.local_preflight(cfg, good_env), [])
            self.assertIn("BIRD_NEO4J_PASSWORD", fw.local_preflight(cfg, {**good_env, "BIRD_NEO4J_PASSWORD": ""})[0])
            few = SimpleNamespace(env_file=".env", instances_path=str(inst), train_split="train", batch_size=4)
            self.assertIn("fewer than batch_size 4", fw.local_preflight(few, good_env)[0])
            gone = SimpleNamespace(env_file=".env", instances_path=str(Path(d) / "nope.jsonl"), train_split="train", batch_size=2)
            self.assertIn("does not exist", fw.local_preflight(gone, good_env)[0])
            self.assertEqual(len(fw.local_preflight(gone, {})), 2)

    def test_resume_is_refused_not_guessed(self) -> None:
        shim = fw.FireworksServiceShim()
        with self.assertRaises(NotImplementedError):
            self.run_async(shim.create_training_client_from_state_with_optimizer_async("x"))

    def test_close_closes_samplers_and_service(self) -> None:
        shim = fw.FireworksServiceShim()
        tc = self.run_async(shim.create_lora_training_client_async("Qwen/Qwen3.8-27B"))
        self.run_async(tc.save_weights_and_get_sampling_client_async())
        shim.close()
        self.assertTrue(self.service.closed and all(s.closed for s in self.service.samplers))


if __name__ == "__main__":
    unittest.main()
