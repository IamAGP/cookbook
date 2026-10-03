"""Fireworks serverless-training candidates for a second student: prices and a baseline-pass estimate.

Inputs
- PRICES: USD per 1M tokens, Fireworks pricing page "Serverless Training" section, fetched 2026-10-03.
- PARAMS, MOE: Fireworks registry (`firectl model get … -o json`), 2026-10-03.
- RENDERER: result of tinker_cookbook.model_info.get_recommended_renderer_name(<HF id>) at the
  peer's commit 939a1bf, i.e. whether her harness can render the model as it stands.
- Token counts: the Qwen3.5-9B baseline (`~/bird_rl_runs/e3_qwen3_5_9b/results.jsonl`) used as a
  STAND-IN. A different model will take a different number of turns and tokens, so every dollar
  figure below is an estimate, not a measurement.
- CACHED_FRACTION: measured on the Fireworks Qwen3.8-27B baseline (743,000 / 1,577,817).

    python bird_graph_rl/student_options.py [out.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PRICES = {  # prefill, cached prefill, sample, train
    "deepseek-v4-flash-0731": (1.74, 0.35, 4.33, 5.20),
    "glm-5p3-flash": (2.96, 0.593, 7.41, 8.89),
    "glm-5p3": (4.86, 0.972, 12.15, 14.58),
    "kimi-k3": (10.87, 2.17, 27.11, 32.55),
    "muse-glimmer-30b": (1.96, 0.39, 4.88, 5.86),
    "qwen3p8-27b": (1.86, 0.372, 5.595, 4.103),
}
PARAMS = {"deepseek-v4-flash-0731": 304_000_000_000, "glm-5p3-flash": 320_000_000_000, "glm-5p3": 743_377_019_904,
          "kimi-k3": 2_780_913_302_112, "muse-glimmer-30b": 29_776_626_688, "qwen3p8-27b": 27_356_728_560}
MOE = {"deepseek-v4-flash-0731": True, "glm-5p3-flash": True, "glm-5p3": True, "kimi-k3": True,
       "muse-glimmer-30b": False, "qwen3p8-27b": False}
RENDERER = {"deepseek-v4-flash-0731": None, "glm-5p3-flash": None, "glm-5p3": "glm5_3_max_reasoning",
            "kimi-k3": None, "muse-glimmer-30b": None, "qwen3p8-27b": "qwen3_8_xhigh_reasoning"}
CACHED_FRACTION = 743_000 / 1_577_817
FW_BALANCE, B200 = 147.88, 13.0


def main() -> None:
    rows = [json.loads(l) for l in (Path("~/bird_rl_runs/e3_qwen3_5_9b/results.jsonl").expanduser()).open()]
    prompt, sampled = sum(r["prompt_tokens"] for r in rows), sum(r["sampled_tokens"] for r in rows)
    out = {"stand_in_tokens": {"questions": len(rows), "prompt": prompt, "sampled": sampled}, "models": {}}
    for m, (pre, cached, sample, train) in PRICES.items():
        upper = (prompt * pre + sampled * sample) / 1e6
        with_cache = (prompt * (1 - CACHED_FRACTION) * pre + prompt * CACHED_FRACTION * cached + sampled * sample) / 1e6
        out["models"][m] = {
            "params_billion": round(PARAMS[m] / 1e9, 1), "moe": MOE[m], "harness_renderer": RENDERER[m],
            "baseline_186_usd_upper_estimate": round(upper, 2),
            "baseline_186_usd_at_measured_cache_rate_estimate": round(with_cache, 2),
            "train_usd_per_rollout_upper_estimate": round((prompt + sampled) / len(rows) * train / 1e6, 4),
            "price_vs_qwen27b_sample": round(sample / PRICES["qwen3p8-27b"][2], 2),
        }
    out["gemma4_dedicated"] = {
        "trainer_gpus_B200": 4, "trainer_usd_per_hour": 4 * B200,
        "hours_in_balance_trainer_only": round(FW_BALANCE / (4 * B200), 2),
        "hours_in_balance_with_one_gpu_sampler": round(FW_BALANCE / (5 * B200), 2),
        "sampler_gpu_count": "UNKNOWN (no deployment shape checked for Gemma 4); one GPU is an assumption",
    }
    if len(sys.argv) > 1:
        Path(sys.argv[1]).expanduser().write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
