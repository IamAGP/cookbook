"""Every cost figure this side states, computed from listed inputs. No mental arithmetic.

MEASURED inputs are read from files or were read from a billing API (source named).
PRICES are published list prices (source named). ASSUMED inputs are choices, not facts.
Anything that cannot be computed from these is printed as UNKNOWN rather than guessed.

    python bird_graph_rl/cost_estimates.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path("~/bird_rl_runs").expanduser()

# --- MEASURED -------------------------------------------------------------------------------
# firectl billing get-usage --start-time 2026-09-21 --end-time 2026-09-23 --group-by model_name
FW_BILL_USD = 3.12
FW_BILLED = {"prompt": 1_577_817, "cached_prompt": 743_000, "uncached_prompt": 834_817,
             "completion": 231_187}
FW_ROLLOUTS_BILLED = {"probe": 1, "smoke": 10, "full": 186}
FW_BALANCE_USD = 147.88          # owner's console, relayed 2026-10-03
TINKER_BALANCE_USD = 129.0       # peer's statement 2026-10-03 ("about $129")

# --- PRICES, USD per million tokens ---------------------------------------------------------
# Fireworks docs /fine-tuning/cost-estimator catalog, generated 2026-09-30.
FW_27B = {"prefill": 1.86, "cached": 0.372, "sample": 5.595, "train": 4.103}
# Same catalog's "tinker" entry for Qwen/Qwen3.5-9B; prefill/cached/sample also match the peer's
# harness price table (read from tinker-docs 2026-10-03). "train" comes from the catalog only.
TINKER_9B = {"prefill": 0.66, "cached": 0.132, "sample": 1.995, "train": 1.463}
FW_GPU_HOUR = {"B200": 13.0}     # same catalog, gpuRates, effective 2026-09-01
FW_9B_GPUS = {"trainer_B200": 2, "sampler_B200": 1}  # training-shape + rft deployment shape

# --- ASSUMED (choices, not facts) -----------------------------------------------------------
GROUPS_PER_STEP, ROLLOUTS_PER_GROUP = 16, 8          # peer's proposed run-1 config
TRAIN_INSTANCES = 1500                               # plan T1 target, not a measured count
REFERENCE_RUN_ROLLOUTS = 3200                        # 8 x 8 x 50, the size of the earlier runs
HELDOUT_STRUCTURE_INSTANCES = 360                    # 20% of 150 structures x 12; not measured
K = 4


def tokens(run: str) -> dict[str, float]:
    rows = [json.loads(line) for line in (RUNS / run / "results.jsonl").open()]
    n = len(rows)
    prompt = sum(r["prompt_tokens"] for r in rows)
    sampled = sum(r["sampled_tokens"] for r in rows)
    return {"n": n, "prompt": prompt, "sampled": sampled,
            "prompt_per_rollout": prompt / n, "sampled_per_rollout": sampled / n}


def main() -> None:
    fw27, tk9 = tokens("e2fw_full_qwen3_8_27b"), tokens("e3_qwen3_5_9b")
    n_billed = sum(FW_ROLLOUTS_BILLED.values())
    out: dict[str, object] = {}

    # Fireworks 27B, sampling: measured bill / measured rollouts.
    fw_sample_per_rollout = FW_BILL_USD / n_billed
    out["fw27_sampling_usd_per_rollout_measured"] = round(fw_sample_per_rollout, 5)
    recomputed = (FW_BILLED["uncached_prompt"] * FW_27B["prefill"] + FW_BILLED["cached_prompt"] * FW_27B["cached"]
                  + FW_BILLED["completion"] * FW_27B["sample"]) / 1e6
    out["fw_bill_recomputed_from_list_prices_usd"] = round(recomputed, 4)
    share = fw27["prompt"] / FW_BILLED["prompt"]
    one_pass = (FW_BILLED["uncached_prompt"] * share * FW_27B["prefill"]
                + FW_BILLED["cached_prompt"] * share * FW_27B["cached"]
                + fw27["sampled"] * FW_27B["sample"]) / 1e6
    out["fw27_one_186_pass_usd_prorated_estimate"] = round(one_pass, 2)
    out["fw27_cached_prompt_fraction_measured"] = round(FW_BILLED["cached_prompt"] / FW_BILLED["prompt"], 3)

    # Training tokens per rollout: upper bound is one datum per turn, i.e. every prompt and
    # sampled token passes through forward_backward once. The lower bound (turns merged into
    # one datum) needs the final sequence length, which results.jsonl does not store.
    fw_train_upper = (fw27["prompt_per_rollout"] + fw27["sampled_per_rollout"]) * FW_27B["train"] / 1e6
    out["fw27_train_tokens_per_rollout_upper"] = round(fw27["prompt_per_rollout"] + fw27["sampled_per_rollout"])
    out["fw27_train_usd_per_rollout_upper_estimate"] = round(fw_train_upper, 5)
    out["fw27_train_usd_per_rollout_lower"] = "UNKNOWN until a smoke run reports datums per trajectory"
    out["fw27_rl_run_usd_upper_estimate"] = {
        "rollouts": REFERENCE_RUN_ROLLOUTS,
        "usd": round(REFERENCE_RUN_ROLLOUTS * (fw_sample_per_rollout + fw_train_upper), 2)}
    out["fw27_rl_run_usd_sampling_only_floor"] = round(REFERENCE_RUN_ROLLOUTS * fw_sample_per_rollout, 2)

    # Evaluation on Fireworks 27B.
    out["fw27_eval_186_at_k_usd_estimate"] = {"k": K, "usd": round(K * one_pass, 2)}
    out["fw27_eval_heldout_structure_at_k_usd_estimate"] = {
        "instances_assumed": HELDOUT_STRUCTURE_INSTANCES, "k": K,
        "usd": round(HELDOUT_STRUCTURE_INSTANCES * K * fw_sample_per_rollout, 2)}

    # Fireworks 9B dedicated.
    usd_per_hour = sum(FW_9B_GPUS.values()) * FW_GPU_HOUR["B200"]
    out["fw9_dedicated_usd_per_hour"] = usd_per_hour
    out["fw9_dedicated_hours_in_balance"] = round(FW_BALANCE_USD / usd_per_hour, 2)
    out["fw9_rl_run_wall_time"] = "UNKNOWN (cannot be measured without provisioning)"

    # Tinker 9B per rollout and per step, from the peer's E3 token counts.
    tk_sample_upper = (tk9["prompt_per_rollout"] * TINKER_9B["prefill"]
                       + tk9["sampled_per_rollout"] * TINKER_9B["sample"]) / 1e6
    tk_train_upper = (tk9["prompt_per_rollout"] + tk9["sampled_per_rollout"]) * TINKER_9B["train"] / 1e6
    per_step = GROUPS_PER_STEP * ROLLOUTS_PER_GROUP
    steps = TRAIN_INSTANCES // GROUPS_PER_STEP
    out["tinker9_sampling_usd_per_rollout_upper_estimate"] = round(tk_sample_upper, 5)
    out["tinker9_train_usd_per_rollout_upper_estimate"] = round(tk_train_upper, 5)
    out["tinker9_usd_per_step"] = {
        "rollouts_per_step": per_step,
        "upper_estimate": round(per_step * (tk_sample_upper + tk_train_upper), 3),
        "sampling_only_floor": round(per_step * tk_sample_upper, 3),
        "lower": "UNKNOWN until the smoke run reports datums per trajectory"}
    out["tinker9_one_epoch"] = {
        "train_instances_assumed": TRAIN_INSTANCES, "steps": steps,
        "usd_upper_estimate": round(steps * per_step * (tk_sample_upper + tk_train_upper), 2),
        "usd_sampling_only_floor": round(steps * per_step * tk_sample_upper, 2),
        "tinker_balance_stated": TINKER_BALANCE_USD}
    out["tinker9_steps_affordable_at_upper"] = int(TINKER_BALANCE_USD // (per_step * (tk_sample_upper + tk_train_upper)))

    out["inputs"] = {"fw27_tokens": fw27, "tinker9_tokens": tk9, "FW_BILL_USD": FW_BILL_USD,
                     "FW_BILLED": FW_BILLED, "FW_ROLLOUTS_BILLED": FW_ROLLOUTS_BILLED,
                     "FW_27B": FW_27B, "TINKER_9B": TINKER_9B, "FW_GPU_HOUR": FW_GPU_HOUR,
                     "FW_9B_GPUS": FW_9B_GPUS, "FW_BALANCE_USD": FW_BALANCE_USD}
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).expanduser().write_text(text)
    print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
