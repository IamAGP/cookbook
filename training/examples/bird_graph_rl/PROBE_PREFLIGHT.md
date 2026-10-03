# Preflight — Fireworks dedicated timing probe (Qwen3.5-9B)

**Status: DRAFT, not approved. No hardware has been requested. Written 2026-10-03, before any
machine exists, as the owner's rented-GPU rule requires.**

Every figure here is either measured (source named) or computed from stated inputs and marked
as an estimate. Unknowns are listed as unknown.

## 1. What the probe is for

The owner wants the same Qwen3.5-9B experiment on Fireworks that ran on Tinker. On Fireworks
the 9B trains only on dedicated GPUs, billed by the hour. Whether a run fits $147.88 depends on
one number nobody has measured: how long a training step takes on that hardware. The probe
measures it. It does not train a useful model and cannot show a gain.

**Decision it feeds.** A 37-step run of 8 × 8 fits if a step takes about as long as on Tinker
(median 64.5 s there; at Tinker's observed wall times the run would cost $41.13 at $39 per
hour). The probe says whether Fireworks is in that range.

## 2. What will run

| Phase | Resources up | Rate | What is measured | Planned time |
|---|---|---|---|---|
| A | sampler deployment, 1 × B200 | $13.00/h | rollouts per minute and tokens per second at concurrency 16, 64, 128, base 9B, generated pilot questions | 10 min |
| B | trainer 2 × B200 + sampler 1 × B200 | $39.00/h | seconds for forward_backward, optim_step, saving sampler weights, weight sync into the deployment; 2 updates of 4 × 8 | 10 min |

Rates: Fireworks pricing page, fetched 2026-10-03 ($13.00 per B200-hour). GPU counts: training
shape `qwen3p5-9b-65k-lora` (2 × B200) and sampler deployment shape `rft-qwen3p5-9b-v2`
(1 × B200), from the registry.

Questions come from the generator's pilot/training files. **Never the 186.**

## 3. Cost (estimates)

- Planned: 10 min at $13 + 10 min at $39 = **$8.67**.
- If 10 minutes of provisioning are also billed at the full rate: **$15.17**. Whether trainer
  initialization is billed is not settled by the docs (cost catalog: "Queue wait is not
  billed", and its floor excludes initialization; pricing page: "no extra charges for start-up
  times"). Budget on the higher figure.
- Hard cap requested from the owner: **$20**, enforced as a wall-clock deadline.
- Opening balance to record immediately before launch: console credits (last known $147.88)
  and `firectl billing get-usage` for the month.

## 4. The four questions, answered from reading the code that will run

1. **Timestamped progress line per unit of work?** Yes. Phase A: the harness prints one line
   per finished question with running cost. Phase B: the loop writes `metrics.jsonl` per step
   and the shim writes one line to `fw_cost_meter.jsonl` per training call, each with a
   timestamp, elapsed seconds and dollars so far.
2. **Results written incrementally?** Yes. `results.jsonl` per question; `metrics.jsonl` per
   step; `fw_cost_meter.jsonl` per training call; `fw_resources.json` the moment resource ids
   exist. Nothing is held until the end.
3. **Does working memory grow, and what is the peak against the card?** The GPUs are
   Fireworks' validated shapes for this model; nothing of ours is loaded onto them beyond a
   rank-32 adapter. On the laptop, memory is bounded by concurrency: at most 128 conversations
   of at most 56K tokens, tool output capped at 50 rows / 6,000 characters. No growth across
   steps.
4. **If killed at 80%, what survives?** Every timing line already written, which is the
   probe's entire output. No checkpoint is needed. The resources do **not** survive: see §5.

## 5. What stops the meter

Three layers, because each fails differently.

1. **In the run.** Wall-clock meter from the moment resources are requested; a hard deadline
   (`FW_DEADLINE_MIN`) and a dollar cap (`FW_BUDGET_USD`); on either, the service is closed with
   `cleanup_trainer_on_close=True` and `cleanup_deployment_on_close="delete"`. The run refuses
   to start without `FW_ALLOW_DEDICATED=1`, a deployment id, a deadline and an inactivity
   timeout.
2. **Independent process** (`fw_watchdog.py`). Deletes both resources through the SDK and
   verifies, on: the deadline; no progress line for 5 minutes; or the training process gone
   with resources still up.
3. **Server side**, for when this laptop itself dies. Trainer inactivity timeout set
   explicitly to 10 minutes (the docs say the default is 10, the CLI says 60; neither is relied
   on). The sampler deployment is patched right after creation to minimum replicas 0 with a
   300 s scale-to-zero window; if the patch fails the run deletes everything and stops.

**Why layer 3 matters.** The SDK creates the sampler deployment with minimum replicas equal to
maximum, so by default it never scales to zero, and deployments have no expiry. Abandoned, it
bills $13 per hour, $312 per day, indefinitely. With the patch, a dead laptop costs an estimated
$5.42.

**Queue wait.** The SDK's default wait for a pending trainer is 172,800 s (48 hours). This run
sets 20 minutes.

## 6. Known gaps — the reasons this run must be attended

- **The deployment patch has never run against the real API.** If it silently does nothing, the
  $312-per-day exposure is back.
- **There is a window during provisioning before the patch applies.** Only something not on
  this laptop can cover it. Not built; it would need the owner's Fireworks key off this machine.
- **`firectl` refuses mutating commands inside an agent session**, so the manual fallback is the
  owner's, in his own terminal:

  ```
  firectl rlor-trainer-job list
  firectl deployment list
  firectl rlor-trainer-job delete <job id>
  firectl deployment delete <deployment id>
  ```

- **Phase A's sampler-only path is not built yet.** The baseline wrapper currently samples a
  base model only through the serverless route. A deployment-only path for the 9B has to be
  written and tested offline first.
- **Not exercised until the probe:** the SDK's managed provisioning, weight sync into the
  deployment, and the loop running against a dedicated trainer.

**Condition for the first run: the owner is at the keyboard, with the four commands above
ready, until both resources are confirmed deleted.**

## 7. Confirmation of zero resources at the end

1. `firectl rlor-trainer-job list` and `firectl deployment list` both return no rows (they do
   today: checked 2026-10-03).
2. `fw_watchdog.jsonl` shows a teardown record with `clean: true`.
3. Next day: `firectl billing get-usage` for the probe's date, compared with the wall-clock
   timestamps of create, ready and delete per resource. This also answers whether
   initialization is billed.

## 8. What the probe cannot tell us

Whether RL produces a gain; steady-state speed after warm-up; how long GPUs take to become
available on the day; the bill, until the billing API catches up.

## 9. What is needed from the owner

- Approval, a dollar cap, and a time when he can attend.
- A decision on an off-laptop scheduled deletion (covers the provisioning window; puts his key
  elsewhere), or attended-only runs.
