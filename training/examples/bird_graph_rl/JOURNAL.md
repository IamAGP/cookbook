# bird_graph_rl on Fireworks — experiment journal

> Published copy. One private individual's name is replaced by their role; storage bucket,
> account identifiers and local paths were moved to environment variables before publication.
> Entries are otherwise as written, including the ones that record my own mistakes.

Append-only. Each entry: Observed / Interpretation / Decision. Sourced facts are marked
with where they were checked; anything not checked with a tool is labelled as inference.

Companion log on the Tinker side: `tinker-cookbook/tinker_cookbook/recipes/bird_graph_rl/JOURNAL.md`
(peer session "atleast_retain this"). Same data, same task, same harness.

---

## 2026-09-22 — E2-FW: zero-shot baseline replication, preflight

**Request (peer, on the user's instruction):** replicate the Tinker E2 zero-shot baseline on
Fireworks. Science fixed (data, prompt, tool, sampling, scoring, metrics); framework free.
User decision 2026-09-21: base model Qwen3.8-27B on both sides.

### Design: run her harness unchanged, swap only the sampler

`baseline_eval_fireworks.py` imports `tinker_cookbook.recipes.bird_graph_rl.baseline_eval`
from her repo and calls its `main()` with one patch: `tinker.ServiceClient` →
`FireworksServiceShim`, which returns a Fireworks serverless base-model sampler. Prompt,
renderer, tool, rollout loop, limits, cost meter, scorer and output files are her code.
The wrapper refuses to start unless her file's git blob equals commit `ed9e7f7`.

### Parity checks done (2026-09-22, all by tool)

| Item | Tinker (hers) | Fireworks (mine) | Source |
|---|---|---|---|
| Weights | `Qwen/Qwen3.8-27B` | `accounts/fireworks/models/qwen3p8-27b`, HF URL `Qwen/Qwen3.8-27B` | `firectl model get` |
| Renderer / reasoning | `qwen3_8_xhigh_reasoning` | same renderer, same function call | her `model_info`, imported |
| Prompt tokens | client-side render | token-in completions (`prompt.to_ints()`) | SDK `FiretitanSamplingClient` source |
| Sampling | T=1.0, top_k=-1, top_p=1 | T=1.0, SDK forces top_k=0 (off), top_p=1.0 | tinker `SamplingParams` defaults; SDK `sampling.py` |
| Stop | int token ids | same ids, decoded to strings by the SDK | SDK `_normalize_stop` |
| Prices / M tokens | prefill 1.86, cached 0.372, sample 5.595 | identical | her `PRICES_PER_M`; Fireworks docs cost catalog |
| Python deps | tinker 0.30.0 | tinker 0.23.0 (pinned by fireworks-ai 1.2.11) | `importlib.metadata` |

**Open until probed:** (a) the SDK's base-only serverless sampling route — the maintained
cookbook never uses it (its unit test asserts `base_sampling_calls == []`); (b) whether the
server returns the stop token in `completion_token_ids` — her parser needs it. The Fireworks
cookbook's own renderer rollout hands raw completion tokens to `parse_response`, which is
indirect evidence it does.

**Known framework differences that are not controllable:** serving precision and kernels
(Fireworks reports `PRECISION_UNSPECIFIED`; Tinker's is not published), and the tinker
client version. These are part of "only the framework changes".

### Rule-7 preflight (read before any spend)

1. **Timestamped progress per unit of work?** Yes. Her `run_one` prints one line per finished
   question with running accuracy, running $ estimate and elapsed seconds.
2. **Results written incrementally?** Yes. One JSON line appended to `results.jsonl` per
   question under a lock; a rerun with the same `log_path` skips finished ids.
3. **Memory growth / peak?** Bounded per question. At most `concurrency=16` conversations
   live; tool output is capped at 50 rows / 6,000 chars before entering context; rescoring
   fetches up to 25,000 rows then drops them. No GPU on our side (serverless).
4. **If killed at 80%, what survives?** Every finished question in `results.jsonl`. Lost:
   up to 16 in-flight questions (their tokens are billed but unrecorded). `summary.json` is
   only written at the end, and is recomputed from `results.jsonl` on the resumed run.

**Watchdog:** serverless has no idle GPU charge. Cost guard = her `budget_usd` (no new
question starts once the upper-bound estimate crosses it) plus her per-question 16,384
sampled-token cap. Sessions are closed in a `finally` block.

**Opening balance:** `firectl billing get-usage` 2026-09-01..09-23 = **$0.00**. Console
credits were $151.00 on 2026-09-14 (screenshot); no usage since, so opening ≈ $151.00.

**Budget:** her full run was 1.53M prompt + 0.21M sampled tokens → upper bound
1.53×1.86 + 0.21×5.595 ≈ $4.0 (actual Tinker spend $2.33 with prefix caching).
Caps: probe < $0.05 · smoke `limit=10 budget_usd=1.5` · full `budget_usd=8`.

**Shared DB:** the Neo4j instance is shared with the Tinker side and has a 120 s transaction
timeout. Concurrent load from both frameworks could turn slow queries into timeout errors the
model sees, which would change behaviour. Run only when the Tinker side is idle.

**Plan:** probe (1 sample) → smoke (10) → full (186) → per-question diff vs
`~/bird_rl_runs/e2_full_qwen3_8_27b/results.jsonl`.

## 2026-09-22 00:14 — E2-FW probe (1 sample, ≈$0.002)

**Observed:** base-only route works — sampler model `accounts/<acct>/trainingSessions/ts-…`
with no checkpoint segment. Renderer `qwen3_8_xhigh_reasoning`, 922 prompt / 91 sampled
tokens, `stop_reason=stop`, last token 248046 `<|im_end|>` (the renderer's stop id),
logprobs aligned, `parse_response` ok, one well-formed `run_cypher` tool call
(`MATCH (u:User) RETURN count(u) AS users`). Harness blob matched `ed9e7f7`.
**Interpretation:** both open items from the preflight are closed: base-only sampling works
and the server returns the stop token, so her parser sees complete messages.
**Decision:** proceed to the 10-question smoke.

## 2026-09-22 00:16 — E2-FW smoke (first 10 questions, est. upper $0.19)

**Observed:** 10/10 scored, 0 harness errors, session closed. Strict 0.60, lenient 0.70,
no_query 0. Versus her Tinker smoke on the same 10: strict 0.50 vs 0.60, 9/10 questions
agree (Fireworks alone solved q531). Fireworks used more turns (4.5 vs 2.9), had 3 Cypher
errors vs 0 and 1 turn-cap hit vs 0, and sampled fewer tokens per turn (≈192 vs ≈390).
**Interpretation:** too small to separate framework from sampling noise. The turns/tokens
pattern is the thing to check on the full run: same renderer and prompt tokens, so a real
difference in thinking length would point at serving numerics, not the experiment.
**Decision:** run the full 186 with `budget_usd=8`, then a paired per-question comparison
(`compare_runs.py`, McNemar exact test on discordant questions).

## 2026-09-22 00:59 — E2-FW full run (186 questions) and paired comparison with Tinker

**Observed (Fireworks vs Tinker, same 186 questions, one sample each at T=1.0):**

| metric | Tinker | Fireworks |
|---|---|---|
| strict | 0.608 | 0.608 |
| lenient | 0.677 | 0.656 |
| no_query | 0 | 0 |
| mean turns | 4.31 | 4.18 |
| turn-cap hits | 14 | 13 |
| Cypher errors seen by model | 56 | 44 |
| prompt / sampled tokens | 1.53M / 0.21M | 1.50M / 0.22M |
| sampled tokens per turn | 262 | 286 |

By difficulty (strict): simple 0.675 vs 0.636 (n=151), moderate 0.367 vs 0.500 (n=30),
challenging 0.000 vs 0.400 (n=5). Paired strict: both right 92, both wrong 52, Tinker-only 21,
Fireworks-only 21 → McNemar exact p = 1.0; agreement 0.774. Paired lenient: 18 vs 14, p = 0.60.
0 harness errors, 0 budget skips, session closed. Wall time 12 min 20 s.

**Cost (billing API, 2026-09-21..23, probe + smoke + full):** $3.12. Billed tokens equal the
harness meter exactly: 1,577,817 prompt (743,000 cached / 834,817 uncached) + 231,187
completion; 834,817×1.86 + 743,000×0.372 + 231,187×5.595 per M = $3.12, so the documented
prices are the charged prices. Balance ≈ $147.88.

**Interpretation:** no detectable framework effect on accuracy. Identical strict score and a
perfectly symmetric 21/21 split of discordant questions is what two independent samples of the
same model look like. The 22% of questions that flip between runs is sampling variance at
T=1.0, and it is the noise floor any later RL gain must clear on this set. The smoke's
turns/tokens gap did not survive at n=186 (262 vs 286 tokens per turn). Per-difficulty gaps on
moderate and challenging are within noise at n=30 and n=5.
Cost difference is a caching effect, not price (inference from the arithmetic, assuming Tinker
meters the same way): Fireworks cached 47% of prompt tokens; Tinker's $2.33 on similar volume
implies roughly 74% cached.

**Decision:** baseline replicated. Report to the peer and the user; RL waits for the shared
reward/environment from the Tinker side. Artifacts: `~/bird_rl_runs/e2fw_full_qwen3_8_27b/`,
comparison `~/bird_rl_runs/e2fw_vs_tinker_compare.json`.

## 2026-09-22 — Peer verification and eval protocol for checkpoints

**Observed:** the Tinker-side session re-derived every E2-FW number from the two
results.jsonl files and confirmed blob 6390bc9 = `baseline_eval.py` @ ed9e7f7 (her JOURNAL
commit 4602d02). From the same paired table: 134/186 questions solved in at least one of the
two samples (92 + 21 + 21), 42 flip between samples, 52 never solved.
**Interpretation:** the 42 flippers are where group-relative RL has gradient signal; the 52
never-solved give none unless sampling finds a success. Single-shot accuracy is too noisy to
compare checkpoints (≈22% of questions flip between two samples).
**Decision (agreed with peer, pending the user):** compare checkpoints with k samples per
question on the 186, paired per question. k is the user's call because eval cost scales with
it: one full single-sample pass cost ≈ $3 here, so each checkpoint eval ≈ k × $3 per
platform (estimate; prefix caching may lower it). RL env + reward still to come from the
Tinker side, hash-pinned, before anyone trains.

## 2026-10-03 — Independent check of the partial-credit measurement

**Observed.** Re-executed all 372 stored `final_query` values from both E2 runs against the
live graph (`verify_partial_credit.py`, database reads only, no API spend) and scored
row-level F1 against the references, capped so 1.0 requires strict agreement.

| quantity | peer (Tinker side) | mine, independent |
|---|---|---|
| never-solved (strict 0 in both runs) | 52 | 52 |
| of those, non-zero partial credit | 13 | 13 |
| of those, still exactly zero | 39 | 39 |
| spread across the two samples, binary | 42 | 42 |
| spread across the two samples, partial | 47 | 47 |
| mean reward Tinker / Fireworks | 0.639 / 0.642 | 0.636 / 0.635 |

Rescore errors 0; strict drift versus stored values 0, so the graph is stable and stored
queries re-execute identically. Rescued ids: 538, 552, 587, 590, 610, 612, 618, 621, 638,
652, 681, 706, 708.

**Interpretation.** Every structural count reproduces exactly, so her finding stands: partial
credit grades 5 more questions and 39 of the 52 hardest return rows with zero overlap — those
queries are semantically wrong, not nearly right. The mean reward differs in the third decimal
between two independent implementations of "row-level F1", and the sign of the
Tinker-vs-Fireworks gap flips with it (hers favours Fireworks, mine favours Tinker). Both
gaps are noise, but it shows the reward cannot be specified in prose: the pinned spec must be
the code, and both arms must import the same function.
**Decision.** Ask for the reward as code, hash-pinned, and import it rather than reimplement.
Her caveat accepted: 52 is an upper bound on dead questions measured at k=2, to be
re-measured at the k the user sets.

## 2026-10-03 — Divergence explained; reward pinned; Jev mapping proposed

**Observed (all by tool today).** The reference file's truncated questions are exactly the
peer's 11 (532, 546, 590, 596, 602, 621, 638, 676, 706, 708, 715); 5 of them (590, 621, 638,
706, 708) are among the 13 rescued by partial credit; their full results total 60,527 rows.
Her `reward.py` on disk hashes to blob `1c658a14d2ea1b02bfcd1519dc71a8733123ddf0`, equal to
the blob at her commit `7f6864c`. Her 13 rescued ids are identical to mine, not merely the
same count.

**Interpretation.** The third-decimal mean-reward gap came from the truncated branch, and that
approximation lands on 5 of the 13 questions that carry the argument for partial credit, not
in a harmless tail. She is regenerating exact reference rows for those 11, which deletes the
approximation instead of choosing between two defensible readings.

**Decision.** Import `partial_credit` from her module and gate on the blob hash, the same way
the harness is gated; re-pin when the exact-rows version lands. I asked her to re-execute all
186 gold SQLs during regeneration and diff against the current reference file on the 175
unaffected questions: if any differ, the gold has drifted and both published numbers are
suspect; if none differ, the regeneration pipeline is proven faithful.

**Jev arm (read from `src/fw_jev/reward.py`, endpoint `api.typesafe.ai/v1/systemone`, model
`jev-1.13.0`).** Their pattern: choice mapped 1.0/0.5/0.0, score over levels 0-3 mapped
0/0.333/0.667/1.0, weighted 0.5/0.5, times a support factor, with guards that zero the reward.
Proposed for this task: state = question + hint + final Cypher + executed rows + true row
count, no gold; `answers_question` (choice) and `query_faithful` (score 0-3), equal weights,
bounded [0,1] like `partial_credit`; guards zero the reward on no query, query error, or zero
rows; judge sees at most 30 rows / 3,000 characters, pinned so it is not a silent free
parameter. Stated asymmetry, which is the experiment rather than a flaw: her reward reaches
1.0 only when strictly correct, mine can reach 1.0 while wrong.

**Two pre-training measurements proposed:** (1) judge-group spread — GRPO needs variance
within a group of 8, and a judge returning near-identical scores yields no gradient;
(2) judge-versus-checker agreement on the 186, where Jev says 1.0 and `partial_credit` says 0
and the reverse. Both need `TYPESAFE_API_KEY`. Jev cost is negligible at $0.042 per M input
tokens; the Fireworks sampling is the real spend.

**Blocked on the user:** TypeSafe key; k for checkpoint evals; push destination for this repo;
and his `gh` account switch, which is blocking her push (her pin is valid on disk regardless).

## 2026-10-03 — Plan received (hard stop 2026-10-17); new reward pin verified; objections filed

**Observed (who verified what).**
- Peer wrote `PLAN.md` and `datagen/SPEC.md` at her commit `e980d80` (pushed; `origin/main`
  contains it — checked with `git branch -r --contains`). Objective relayed from the user:
  baseline a smaller open model, post-train it on the graph-agent task, measure the gain;
  Fireworks post first. Judge arm demoted to stretch T9 and must use a self-hosted open-weights
  decision model, not Jev, so the TypeSafe key is no longer needed (relayed by peer, not yet
  confirmed to me by the user directly).
- New reward pin: `reward.py` blob `e326e77114d181f12f7fd95db6278b3c4cdb1cc2`. I verified disk
  blob = commit blob, imported `load_references_exact` + `partial_credit` behind a hash gate
  (`verify_reward_pin.py`) and re-executed all 372 stored final queries: mean reward
  0.640 Tinker / 0.643 Fireworks, 0 truncated references, 0 approximate scores, 52 / 13 / 39,
  spread 42 / 47, same 13 ids. Exact match to her re-measurement.
- `perplexity-ai/pplx-decider-v1-27b` exists (Hugging Face page read today): Apache 2.0, 26B,
  fine-tuned from Qwen3.8-27B, BF16 ≈ 49 GiB, custom inference script, choice and yes/no
  primitives with calibrated probabilities; no score primitive listed.
- Still open on her side: the full 186 gold-SQL drift diff. Verified so far only for the 11
  regenerated questions (row counts equal stored `n_rows`).

**Objections sent (plan).** P1 no budget line — measured sampling ≈ $0.016/rollout; training
estimated $0.013–$0.038/rollout at $4.103/M (estimate, datum structure unmeasured); a
3,200-rollout run ≈ $93–$173 vs $147.88 left; k=4 eval on 186 = $12.50, on ~360 held-out
instances ≈ $23. P2 cut order would drop the Fireworks RL run from a Fireworks-first post.
P3 checkpoint selection on the 186 biases the reported score; select on generated validation.
P4 name the paired test for k>1 and pre-register the bar. P5 T5 depends on T4 the same day;
smoke on pilot instances, never the 186. P6 T4 smoke must report datums per trajectory and
training tokens per rollout.
**Objections sent (spec).** S1 structure signature is defined only for generated queries, not
for the 186 or model-written Cypher. S2 inconsistent granularity (group-by key included,
filter property and aggregation target not). S3 canonicalisation gaps (branch order, direction,
:Question vs :Post+label). S4 split should also hold out whole components and tag held-out
structures novel-combination vs novel-component. S5 training prompts have no hint, evaluation
prompts do.

**Decision.** Accept T5, T6, T8, stretch T9. Build nothing billed until the environment is
pinned and the user approves a budget.

## 2026-10-03 — Student-model question: what Fireworks can actually train, and at what price

**Context.** Peer proposes changing the student from Qwen3.8-27B to Qwen3.5-9B or 3.5-4B: her
sub-agent's review of providers' own release notes puts the 27B among the strongest open
agentic models (released 2026-08-14), which contradicts the user's objective of post-training
a *smaller* model; and my P1 arithmetic showed one 27B run consumes either balance. She asked
me which candidates Fireworks can sample and fine-tune, and at what price.

**Observed (Fireworks registry and docs, today).**
- Qwen3.5-4B: READY in the registry but no LoRA flag, no tunable flags, no training shape,
  absent from the cost catalog. Not trainable on Fireworks.
- Qwen3.5-9B: LoRA-tunable (SFT and RL) but not on serverless training (`serverless: null`;
  deprecated 2026-08-26). Dedicated shapes only, each 2 × B200.
- GPU rates per GPU-hour, effective 2026-09-01: B200 $13, H100/H200 $8, B300 $15, GB300 $20.
  The 9B RL sampler shape `rft-qwen3p5-9b-v2` is 1 × B200 BF16. Trainer + sampler = $39/hour.
- 172 of 186 evaluation questions carry a hint, median 68 characters.
- Custom-model upload documents no mechanism for a separate read-out head.

**Interpretation.** The peer's "prices drop about 2.8x" holds on Tinker, which bills the 9B per
token. It does not transfer to Fireworks, where the 9B changes the billing model to hourly
dedicated: $147.88 ≈ 3.8 hours of cluster time for baseline, smoke, RL and evaluation, with
idle and provisioning billed. Wall time of an RL run on that shape is unknown and cannot be
measured without paying to provision. So the Fireworks choice is per-token/no-idle on a model
that is not small, versus hourly/idle-risk on one that is. I did not argue for keeping the 27B
as student; her frontier evidence is sound.

**Decision / proposals sent.** Options for the user: (A) 9B on both, Fireworks dedicated, one
short run under the rented-GPU preflight; (B) train on Tinker, deploy and evaluate the adapter
on Fireworks; (C) ask the Fireworks developer-relations contact for credits or 9B serverless access, citing the planned
post — do this regardless. Build T5 surface-agnostic: validate per token on 27B serverless,
then point at the 9B dedicated shape for the real run.
S5 preference: hints on generated instances at the evaluation rate (92%), written
deterministically from the query's own structure. P4 bar proposed: paired bootstrap over
questions on per-question mean strict at k; gain claimed only if the 95% lower bound > 0 and
the point estimate ≥ 5 points; headline = fraction of gap to reference closed; simple tier must
not fall more than 3 points; checkpoint chosen on `heldout_instance`, evaluated once.

**Ledger.** Appended F17–F24 to the shared ledger in the peer's repo. F17 supersedes the
Fireworks half of F09: $3.12 was 197 rollouts, one pass ≈ $2.98 pro-rated.

## 2026-10-03 — Small-model baselines reproduced; milestone 1 set; T5 design from source

**Observed.**
- Peer ran E3 on Tinker (same harness, 186 questions, 1 sample, T=1.0, renderer `qwen3_5`).
  I recomputed from her `results.jsonl`: Qwen3.5-9B strict 0.425 / lenient 0.591 (simple .483,
  moderate .167, challenging 1/5; 107 failures, 31 lenient-only); Qwen3.5-4B 0.376 / 0.489.
  All match her figures; 0 harness errors.
- Turn-cap counts are reported under two definitions. Cut off by the cap
  (`stop_reason == max_turns`): 14 / 36 / 70 for 27B / 9B / 4B. Used all 8 turns
  (`turns == 8`): 22 / 39 / 72. Her E3 message used the second, the 27B's "14" the first.
- Harness blob 9b8589c vs 6390bc9: exactly 5 added lines, price table only (`git diff`).
  Wrapper re-pinned to commit 4539665.
- User direction relayed by peer: decision model parked, T9 struck. Milestone 1 = Qwen3.5-9B
  from 0.425, RL on Tinker with `partial_credit`, judged by the pre-registered bar, which she
  counter-signed unchanged. S5 accepted as proposed.
- Read `tinker_cookbook/rl/train.py` against the Fireworks SDK: 12 distinct client methods
  called; all implemented on Fireworks except `save_weights_and_get_sampling_client[_async]`
  (NotImplementedError by design) and `training_client.create_sampling_client(path)` (needs a
  managed deployment). `rl.train` imports under tinker 0.23.0.

**Interpretation.** Headroom is real: 18.3 points for the 9B, single-sample, ±3.6. The 9B's
36 cap cut-offs against the 27B's 14 suggest part of the gap is efficiency, not only
correctness. T5 can reuse the baseline's method: run her RL loop unchanged and swap the
client, with a thin shim mapping the two unsupported calls onto `save_weights_for_sampler` +
`service.create_sampling_client(model_path=…)`. That shares loop code across arms, not just
env and reward. Not yet run; a source-level finding only.

**Decision.** Wait for T4's pinned env, the exact `rl.train.Config`, the loss_fn string, KL
setting, datums per trajectory and training tokens per rollout. Milestone 1 is Tinker-only, so
the Fireworks role stays undecided until the user picks A, B or C.

## 2026-10-03 — Review of the pinned RL environment (peer commit 7107dc1)

**Observed.**
- Pins verified, disk = commit: `rl_env.py` 5d8f30a42aae, `train.py` b210be5d80a4,
  `reward.py` e326e77114d1, `baseline_eval.py` 9b8589c2e21e.
- `"importance_sampling"` is in tinker 0.23.0's `LossFnType` and is the loss Fireworks' own
  serverless RL example passes to `forward_backward`. Source-level confirmation only.
- Fireworks SDK pins `lora_alpha = 32` regardless of rank and says in source this matches
  Tinker; default trains attention, MLP and unembed. Tinker half not checked by me.
- Fireworks LoRA upload supports `lm_head` only for Llama, Mistral, Qwen2, Qwen2-VL, Kimi K2.5.
  Tinker defaults `train_unembed=True`; `rl/train.py` does not expose the flag.
- Pinned `partial_credit` on the 9B baseline: mean reward 0.452 vs strict 0.425; 9 of 107
  failures get partial credit; 0 of the 31 lenient-only (shape) failures get any.
- `train.py` does not pass `env_file`; in-loop eval is 64 instances at one sample (SE ≈ 6
  points); `n_epochs=1` without `max_steps` gives ≈ 93 steps of 128 rollouts for 1,500
  instances, which I estimate at $100–$175 on Tinker against ≈ $129 (estimate from her E3
  token counts and listed prices).

**Interpretation.** Option B (train on Tinker, deploy the adapter on Fireworks) is probably
closed by the default `train_unembed=True` unless it is changed before run 1; the full-weight
route needs ≈ 19 GB and the laptop has ≈ 8 GB free. For the student, the reward is binary on
98 of 107 failures and blind to its largest identifiable failure class; whether that matters
depends on within-group spread, which is unmeasured. A fix exists in principle (a fixed tier
for an exact match on a column subset) but carries a wide-row gaming risk, so it waits for
data.

**Decision / asks sent.** Decide `train_unembed` before run 1. Smoke should report the
fraction of groups dropped as constant and per-group spread, split by lenient-only outcomes.
Keep parse/overflow rewards at 0.0. Select checkpoints post hoc on ≥ 200 `heldout_instance`
questions. Set `max_steps` from budget. Semaphore on tool execution or no concurrent arms; the
local database and an awake laptop are run dependencies. Naturalness filter: prefer shapes
attested in BIRD's training split (other databases), structure-level, before the split, with
removals reported by component; I must not derive rules from the 186 since I can read them.

## 2026-10-03 — New rule on numbers; costs regenerated; RL shim written and tested offline

**Rule (owner, relayed by peer, now in PLAN.md).** Every number stated must be computed with a
tool: measured, or calculated by a script from listed inputs and labelled an estimate. Unknown
is said as unknown.

**Observed.**
- `cost_estimates.py` regenerates every cost figure from listed inputs (output saved to
  `~/bird_rl_runs/cost_estimates_2026-10-03.json`). Upper bounds I had stated hold: $172.22 for
  a 3,200-rollout 27B run; $1.861 per 128-rollout step for the 9B on Tinker. Lower bounds I had
  stated ($93; $0.013 per rollout; $1.1 per step) rested on a final-sequence length I reasoned
  about and no file stores: withdrawn, now UNKNOWN, with a computed sampling-only floor ($50.68;
  $0.689). k=4 on the 186 is $11.93, not $12.50. Ledger F36 supersedes.
- Peer's pins at 550c7ff verified (disk = commit): `rl_env.py` e221addaa37c, `train.py`
  f962035c475d. Diff read: Neo4j pool capped at 24 with 1,800 s acquisition wait; `env_file`
  threaded through; `train_unembed=False` applied by patching
  `tinker.ServiceClient.create_lora_training_client_async`.
- Tinker's official adapter-export page names `lora_alpha` as exported metadata but gives no
  value or rule (page fetched today). A search summary claimed an example showed 32; the page
  did not bear that out. LoRA alpha parity stays unverified on the Tinker side.
- Owner ruled out option C (asking for credits now): an ask is stronger with a result.
- `rl_train_fireworks.py` written: runs her `cli_main` unchanged with `tinker.ServiceClient`
  replaced by a shim. Bridges the two unsupported calls, shortens checkpoint names to the
  17-character serverless limit, and adds a cost meter with a hard stop (`FW_BUDGET_USD` is
  mandatory). 9 of 9 offline tests pass against a fake service, including that her
  `train_unembed=False` pin reaches the Fireworks call.

**Interpretation.** My first test run failed on a budget value I had set by hand; computing it
from the prices in the test fixed it. A small instance of exactly what the rule is for.
The shim is untested against the real service. Known gaps, refused explicitly rather than
guessed: resume from saved state (the cookbook's resume path reads Tinker-style paths through a
REST client) and dedicated training. Resume matters for the rented-GPU rule: without it a
killed run restarts from zero.

**Decision.** No billed call until the owner approves one. When approved, the first is a
two-update smoke on the 27B serverless pool with a small `FW_BUDGET_USD`, reporting datums per
trajectory and training tokens per rollout from `fw_cost_meter.jsonl`.

## 2026-10-03 — Data v1 rejected by the peer; a stale fact of mine corrected

**Observed.**
- Peer rejected data v1 (469 structures, 5,590 instances, 60 pilot questions) after reading
  the pilot: invented vocabulary, instructions to name return aliases, path-counting questions;
  78.7% of filter uses anchored on a numeric id, 59.9% of instances returned one column aliased
  `value` (her script). Two causes were rules in her spec. Amendments B1–B6 at her commit
  0b77d59. Her queued messages to Codex had not been delivered, so my S1–S4 only took effect
  when she resumed that session.
- Free disk measured now: 127.1 GiB (`df -k /`). Earlier today I told her about 8 GB was free
  and that this blocked the full-weight merge route for option B. That number was measured on
  2026-09-22. I repeated dated evidence as a current fact; corrected to her and in the ledger
  (F38).
- BIRD official page: CC BY-SA 4.0; headline 12,751 pairs, 95 databases, 33.4 GB. `train.zip`
  is 8,919,543,554 bytes = 8.31 GiB by HTTP header.

**Interpretation.** B5's "about 92%" is my own measurement on the 186 (F22) written into a spec
the generator reads. It is a format property, not question content, but it is an exception to
"the generator never saw the evaluation set" and must be declared. B3's naturalness test is a
model's judgement, the agreed fallback; it needs a human-read audit of what was dropped.

**Decision.** Sent the correction, the B5 declaration request, the B3 audit request, and an
offer to build the SQL-to-coarse-shape mapper over BIRD's training questions as the
evidence-based check. Not starting it unasked: the data design is hers.

## 2026-10-03 — Human query shapes from BIRD's filtered training questions (check on B3)

**Why.** The generator's naturalness filter (peer's amendment B3) is a model's judgement.
The peer accepted my offer of an evidence-based check: which coarse query shapes people
actually asked. A check, not a target; the generated distribution is not tuned to it.

**Source (verified by me from the Hugging Face API).** `birdsql/bird23-train-filtered`,
author birdsql, CC BY-SA 4.0, 2,654,294 bytes, 6,601 rows; the card says 6,601 of 9,428
training questions kept by BIRD's own quality filter. File kept in the session scratch area,
not in a repo (owner's rule on staging datasets). The script stops if `codebase_community`
appears among `db_id` values or the row count is not 6,601: neither fired; 69 databases.

**Observed (`sql_shapes.py`, output `~/bird_rl_runs/bird_train_filtered_shapes.json`).**
Unclassified 0 of 6,601. Relational hops: 0 → 21.4%, 1 → 57.6%, 2 → 16.8%, 3 → 3.5%,
4+ → 0.67%. Aggregation none 54.6%, count 24.7%. GROUP BY 9.8%. Ordering none 83.0%, top-1
14.2%. No extras 72.1%. Single-column return 83.9%. 452 distinct coarse shapes, 216 seen once;
top 10 cover 60.6%, top 100 cover 91.0%. Equality anchors over 7,166 comparisons: string on a
non-id column 77.1%, number on an id column 6.4%.

**Validation.** 18 of 18 unit tests on hand-written SQL. Zero parse failures looked too good,
so five labels were recomputed from raw text by regex: 0 disagreements on four, 10 (0.15%) on
top-1 ordering. All 10 were SQLite `LIMIT offset, count`; the mapper was right and the regex
wrong, but the mapper had been filing them as plain top-1, so they now carry their own label
(n-th ranked, 17 queries in total once `OFFSET` forms are included).

**Interpretation.** Supports B4 on anchors from evidence that is not the 186. Argues against
pushing the generator toward wider returns: people ask for one column 83.9% of the time, so
v1's fault was the fixed alias, not the single column. Human questions are shallow, so deep
structures will mostly be judgement calls under B3, and those carry the novel_component
result. Limits: SQL on other databases, relational not graph hops, the filtered subset, and
shape ignores which columns are involved.

**Decision.** Delivered to the peer with ledger rows F40–F43. Offered to join against the
generator's output once it reports.

## 2026-10-03 — Peer's amendments C; is the depth rule a check or a target?

**Observed.** From my shape analysis the peer (commit fe76d42) withdrew her push for wider
returns (C1), added n-th ranked, conditional aggregates and DISTINCT projection to the
structure tuple (C3), and reversed her position on depth (C2): at least half of training
instances at ≤1 hop, 3–4 hop instances at most 15%. She asked whether that crosses from check
into target. Computed from the generator's v1 files on disk (modified 10:32:49): 469
structures at 0–4 hops = 16/56/151/122/124, train split 13/45/121/98/99; instances 0.153 at
≤1 hop and 0.523 at 3–4 hops; cap 12 instances per structure. So 58 shallow train structures
give at most 696 shallow instances, the floor caps the train set at 1,392, and the deep cap
leaves 208 instances over 197 deep train structures = 1.06 each. At my upper-bound cost her
balance covers 69 steps × 16 groups = 1,104 prompts.

**Interpretation.** It is a target and should be called one. It is legitimate: the source is
BIRD's training split on other databases, which leaks nothing from the 186. It does not weaken
the headline. It changes two things: the post must say the depth mix follows BIRD's training
split, and with about one instance per deep training structure the novel_combination result
for deep shapes would measure generalisation from single examples, a different test from the
one designed. Meeting the floor by lifting the per-structure cap would recreate template
memorisation.

**Decision.** Told her so, with the arithmetic; recommended fewer deep structures in train with
several instances each and the rest moved to held-out, decided before the split is re-run.
Flagged that C2's quoted 15.0% and 52.0% compute to 0.153 and 0.523. Ledger F44.

## 2026-10-03 — Corrections adopted by the peer; prices confirmed; one wording of mine corrected

**Observed.**
- Peer commit 939a1bf: her C2 figures recomputed from the files (0.153, 0.523; her 15.0% and
  52.0% came from an interim report); C2 retitled a target and declared in PLAN.md beside the
  hint rate; every training structure must have 6–12 instances, the shallow cap may not be
  raised, and deep structures that do not fit move to held-out. On v1 counts: at most 34 deep
  and 81 two-hop structures in training. The 1,500-instance target is dropped.
- The owner asked her whether "Qwen3.5 is trainable on Fireworks" was her claim or mine. She
  read Fireworks' fine-tuning models page: 9B LoRA and full-parameter on dedicated 2–4 B200
  shapes with no serverless option; 27B with serverless; no 4B listed. That agrees with my
  F18–F20. It gives no hourly prices.
- I fetched https://fireworks.ai/pricing: H100 $8.00, H200 $8.00, B200 $13.00, B300 $15.00,
  GB300 $20.00 per hour; dedicated training priced per GPU hour by reference to those;
  serverless 27B training at $1.86 / $0.372 / $5.595 / $4.103 per 1M tokens; "Pay per GPU
  second, with no extra charges for start-up times".

**Interpretation.** $13 per B200-hour now rests on two official sources. My F21 wording that
provisioning is billed was wrong as written: start-up is stated as not charged for on-demand
GPUs. Idle time while allocated is billed (inference from pay-per-GPU-second, supported by the
catalog calling its figure a floor). Whether a trainer's initialization is billed is not
settled by these texts.

**Decision.** Ledger F45 corrects F21. Before any dedicated run is proposed, read the
dedicated-training docs for what is billed during initialization.

## 2026-10-03 — Owner wants the same 9B experiment on Fireworks; timing probe scoped (no spend)

**Context.** The owner's preference, relayed by the peer: Fireworks runs the same Qwen3.5-9B
experiment as Tinker. He asked whether anything had been *run* that showed it was not
possible. Nothing has. The accurate statement is: possible, hourly-billed, unmeasured. An
earlier idea of his, a different student on Fireworks (Gemma 4), was also checked.

**Observed (registry, docs, pricing page, SDK, CLI help; all today).**
- Zero trainer jobs and zero deployments on the account (`firectl … list`, both empty).
- Dedicated training: trainer and deployment are "independently billed resources". SDK entry
  `FiretitanServiceClient.from_firetitan_config(... training_shape_id, deployment_id,
  cleanup_trainer_on_close, cleanup_deployment_on_close)`; `cleanup_trainer_on_close`
  defaults to False. `TrainerJobManager` / `DeploymentManager` expose `delete`, `get`,
  `scale_to_zero`.
- Trainer inactivity stop: 10 minutes by the docs, 60 minutes (max 3 h) by firectl 1.8.6
  help; heartbeats from an active session count as activity. Deployments scale to zero after
  1 hour idle by default, minimum window 5 minutes.
- Start-up billing: cost catalog "Queue wait is not billed", its floor excludes "allocated
  time for model initialization, checkpoints, and idle periods"; pricing page "no extra charges
  for start-up times". Not settled.
- Gemma 4: only 26B (MoE) and 31B are trainable, 4 × B200, no serverless.
- Six serverless-training models, all priced on the pricing page; the cost catalog has `null`
  for both GLMs, contradicting it. Sizes 27.4B to 2,780.9B; none smaller than the 27B. Her
  harness renders two of them: Qwen 3.8 27B and GLM-5.3.

**Computed (python, rates $13 per B200-hour).** Sampler $13/h, trainer $26/h, both $39/h.
Two-phase probe (10 min sampler only + 10 min both) $8.67, leaving 3.57 h; $15.17 if 10 min
of provisioning is billed. One evaluation hour on the sampler alone $13, leaving 3.46 h (the
peer's 2.79 h assumed $39/h). k=4 on the 186 is 744 rollouts; its duration is UNKNOWN until
rollouts per minute is measured. Worst-case leak if the laptop dies: $5.41 with a 10-minute
trainer timeout and 5-minute scale-to-zero; $39.00 with the 60-minute and 1-hour defaults.
`student_options.py`: baseline-pass estimates with the 9B run's tokens as a stand-in: 27B at
most $2.82, GLM-5.3 at most $6.93. At the upper-bound cost the balance covers 2,747 rollouts
of RL on the 27B, 2,304 after reserving two k=4 evaluations.

**Interpretation.** A probe is the right next step for his preference, after milestone 1 on
Tinker. It should be two-phase because the resources bill separately and sampling is probably
the dominant time. It cannot show a gain, steady-state speed, capacity wait, or the bill on the
day. There is no cheap, small, renderable per-token student on Fireworks; the only per-token
fallback is RL on the 27B itself, which answers a different question.

**Decision.** Sent the scoping to the peer: probe design, limits, the unsettled start-up
billing (budget as if billed), a three-layer watchdog (server-side timeouts set explicitly,
in-run wall-clock deadline with cleanup on close, an independent process deleting through the
SDK since firectl blocks mutations in agent sessions), confirmation of zero resources, and the
evaluation-cost correction. Free prerequisite work: build and offline-test the dedicated
branch of the shim; measure database time per query; write the rented-GPU preflight before the
probe. No billed action without the owner's approval. Ledger F46–F51.

## 2026-10-03 — Free prerequisite work: dedicated path, watchdog, database time

**Direction (owner, relayed by peer).** Qwen3.5-9B on Tinker now; Fireworks decided after that
result. Free work requested: the dedicated branch of the shim, tested offline, and the
database-time measurement.

**Observed.**
- Database time (`db_time_measure.py`, serial replay of the 9B baseline's 579 tool queries plus
  each final query's uncapped re-execution; output `~/bird_rl_runs/db_time_9b_serial.json`):
  per query median 0.0105 s, p90 0.0395 s, p99 0.3714 s, max 32.559 s; per rollout median
  0.031 s, p90 0.182 s, mean 0.728 s, max 122.07 s; total 135.4 s = 2.68% of rollout wall time
  in the source run. q614 alone is 90.2% of it (four Cartesian-product queries, about 30 s
  each); without it the mean is 0.0721 s. 0 replay mismatches; 0 of 19 interference checks.
- SDK `FiretitanProvisioningConfig` defaults: `trainer_pending_timeout_s` 172,800,
  `trainer_timeout_s` 3,600, `deployment_timeout_s` 5,400, `inactivity_timeout` None,
  `cleanup_trainer_on_close` False, `preemptible` False; no field for the deployment's
  scale-to-zero window.
- `rl_train_fireworks.py` now has a gated dedicated path (12 of 12 offline tests) and
  `fw_watchdog.py` is an independent teardown process (12 of 12). All suites together: 42 of 42.
  The watchdog's read-only listing shows zero trainer jobs and zero deployments.

**Interpretation.** The laptop's database is not the bottleneck for a typical rollout, so a
probe's step time will be GPU time. The risk is the tail: one bad model-written query can run
to the 120 s server timeout and gate a synchronous 128-rollout step; on hourly billing 122 s
is $1.32. No environment change proposed, since it would have to match on both arms.
The missing scale-to-zero field reopens the laptop-death worst case I gave earlier: up to
$4.33 for the trainer (10-minute inactivity) plus up to $13.00 for the deployment (1-hour
default), not $5.41, until I find the supported way to set the window.

**Decision.** Before any preflight: find how to set the deployment's scale-to-zero window on
the managed path; read what `preemptible` costs and risks; check whether a deployment keeps
serving after its trainer is deleted. Nothing here has touched real hardware. Ledger F52–F54.

## 2026-10-03 — Storage mirroring; an approval I did not act on

**Observed.** The peer relayed three instructions from the owner, who had stepped away: I may
use his AWS account for storage and logs; mirror everything to object storage under her
convention (`monitor.py`, her commit 16e938a); keep the shared ledger and journals current.
She had already mirrored the shared `~/bird_rl_runs` folder, including my run directories. I
scanned every file there for the literal Fireworks key, Tinker key and Neo4j password and for
generic key patterns: 0 files matched. She is starting the Tinker 9B runs on his go-ahead,
opening balance $128 (her figure), RL from base with no warm start, blocked on generated
questions.

**Interpretation.** A permission relayed by another session is not the owner's approval to
me, and writing to his cloud account is a new kind of action for this session. Nothing is lost
by waiting: my outputs land in the folder her tool already mirrors. The real durability gap is
this repository: 17 commits, no remote, one disk.

**Decision.** No AWS writes until the owner tells me directly. Raised the push destination
with him again. Noted to the peer that generated instances carry their gold Cypher, so a
supervised warm-start arm exists at no data cost if RL from base is weak, and that the
pre-registration should say whether such a result counts for milestone 1. Ledger F55.

**Correction (same day).** The entry above and my message to the peer say this repository has
17 commits. I had been keeping that count in my head across messages. `git log --oneline | wc -l`
gave 14 before that entry's commit and 15 after it. The count was never computed, which is the
rule the owner set today. Earlier commit counts I reported to him in summaries were likewise
uncounted and should not be relied on.

## 2026-10-03 — Tinker smoke run reviewed: datum count disputed, costs recomputed, shim re-pinned

**Observed (peer's smoke `~/bird_rl_runs/e4a_smoke`, 9B, 2 updates of 8 × 8).**
- Reproduced from `metrics.jsonl` and the rollout summaries: kept groups 7/8 and 5/8; wall time
  80.6 s and 91.5 s per 64-rollout step (sampling 66.5 and 76.7; train step 11.6 and 8.6); KL
  0.00024 and 0.00027; prompt 10,261 / 8,214 and sampled 1,768 / 1,560 tokens per rollout.
- Her datum figure (102 datums for 96 trajectories, 1.062) does not hold. The 102 markers in
  `logs.log` cover the 16 logged trajectories, 6.375 each. Per-turn lengths show 194 of 313 and
  171 of 254 transitions where the next prompt is shorter than previous prompt plus action, so
  merging is impossible there: at least 4.03 and 3.67 datums per trajectory (5.89 and 4.97
  turns). Training tokens per kept rollout: 11,558 and 11,205 per turn; 2,574 and 2,332 merged.
- At list prices: $1.606 and $1.202 per 64-rollout step, mean $1.404; $2.808 per 128-rollout
  step; 30 steps $84.24 (her figure $47.26). Training is 0.59 and 0.55 of a step.
- Her `train.py` pin is now e6d5b6048a4f (diff: loads the env file with override). All four
  blobs verify at her commit 4ab2639.

**Interpretation.** My upper bound was the right regime: the qwen3_5 renderer drops earlier
thinking, so each turn is its own training sequence. I had withdrawn the lower bound for lack
of evidence and she then read the evidence as supporting it; the files do not. Her point that
groups are rarely constant stands, and my F32 worry about a starved gradient was wrong in
practice. The billed training tokens for the smoke (about 1,095,448 per turn against about
237,424 merged) will settle the datum question.
For Fireworks: a dedicated 30-step 9B run fits if a 128-rollout step takes at most 368 s
including weight sync, unmeasured; the 27B serverless fallback affords 18 to 20 steps under
two stated assumptions.

**Decision.** Told her before run 1 launches, with the resize recommendation. Re-pinned the
shim to 4ab2639 and added `local_preflight`: database variables and the instances file are
checked before any Fireworks resource is requested, since her entry point reads them only
after the training client exists. 43 of 43 offline tests pass. Ledger F61–F63.

## 2026-10-03 — Peer's refinement of the datum model checked and refuted

**Observed.** The peer accepted that turns do not merge into one sequence, resized run 1 to
8 × 8 for 20 steps, and proposed a refinement: a datum ends wherever the next prompt is too
short to extend it, giving 444,444 and 317,374 trained tokens, 3.89 and 3.90 datums per
trajectory, $1.16 per 64-rollout step. Her arithmetic reproduces. But `rl/train.py` (lines
231–266) prints its "---- datum ----" lines from `assemble_training_data`, the trainer's own
function, so they are real datum counts. I identified the four printed groups by their printed
rewards: 16 of 16 printed trajectories are consistent with one datum per turn; 4 of 16 with her
segment model, all short trajectories where both models agree. Eight-turn trajectories print 8
datums; the segment model allows 3 to 7.

**Interpretation.** The length test is necessary for merging, not sufficient: removed thinking
shortens the prompt and an appended tool result lengthens it, so a prompt can be long enough
and still not contain the previous sequence. Length-based counts are a lower bound, which is
how I had stated them. Training is one datum per turn; $1.404 per 64-rollout step is the
estimate, and about 1,095,448 billed training tokens is my prediction for the smoke. The same
function runs under my shim, so the same regime applies on Fireworks.

**Decision.** Sent with the evidence; ledger F66 supersedes the datum counts in her F64 and the
cost in her F65. Practical effect on her run is small since she was already planning on the
higher figure.

## 2026-10-03 — Four-sample 9B baseline reproduced; what the bar can detect at each k

**Observed.** Peer's four-sample baseline of Qwen3.5-9B on the 186 (her ledger F68), reproduced
with `baseline_k_analysis.py` from the four pass folders: 79, 78, 78, 78 correct; per-question
mean strict 0.4207; bootstrap 95% over questions 0.3616–0.4798 (10,000 resamples, seed 0);
simple 0.4768, moderate 0.1833, challenging 0.15; solved in 0/1/2/3/4 of four samples
72/28/21/17/48. The bar needs 0.4707 after training.
Three equal totals looked like repeated samples. They are not: any two passes agree on
correctness for 143–157 of 186 questions, share a final query on 7–14 and a sampled-token count
on 0–1.
Standard error of the difference between two independent k-sample evaluations of an unchanged
model, from within-question variance with questions fixed: 0.0325 (k=1), 0.0230 (k=2), 0.0162
(k=4), 0.0115 (k=8). The bar's 0.05 is 1.5, 2.2, 3.1, 4.3 standard errors.

**Interpretation.** One sample cannot resolve the bar; two is marginal; four is the smallest of
these that puts a 5-point gain clearly outside sampling noise. This is evidence for k=4, the
owner's open decision. It is sampling noise only; generalisation to other questions is the
wider bootstrap interval. 66 questions are mixed, 72 never solved, 48 always solved; 0.05 on
the mean is 9.3 questions' worth, so a gain must come mostly from making mixed questions
reliable unless training unlocks some of the 72.
My "16 of 16" on printed trajectories was consistency under reward-matching, not unique
identification; stated so to the peer.

**Decision.** Sent to the peer with ledger F69; suggested reporting after run 1 which of the
three groups (never, mixed, always) moved.

## 2026-10-03 — Tinker run 1 stage 1 reviewed; timeout data; datagen_v3 join

**Observed (peer's run `e4b_rl_base`, 9B from base, 20 steps of 8 × 8; held-out evals
`e6_hi_*`).**
- Held-out instances (264 questions, one sample) reproduced: 0.504 base, 0.534 step 10, 0.602
  step 20; step 20 minus base +0.098 (paired bootstrap +0.034 to +0.163; 51 better, 25 worse).
  Turn-cap cut-offs 82, 74, 59; mean turns 4.60, 4.36, 4.19; query errors 229, 215, 206;
  sampled tokens per question flat.
- Database calls in stage 1: 6,424; none between 30 and 119 s; 9 at the 120 s timeout in six
  steps; median step 64.5 s; slow steps 763 s above it (37.2% of wall time). No in-loop
  evaluation lines in `metrics.jsonl`.
- Harness pin now `baseline_eval.py` d17b8ffba28e (adds `model_path`, `instances_path`,
  `load_generated`; prompt, tool, caps, scorer untouched). Both wrappers re-pinned to d600861;
  the baseline wrapper refuses `model_path` explicitly. 43 of 43 tests pass.
- datagen_v3 join (`datagen_join.py`): files consistent. Held-out structure is 0.600 at 3–4
  hops against 0.158 in training; only the 2-hop comparison (109 vs 93) is well powered. C2
  missed slightly on the 600 training questions (0.4933 shallow, 0.1583 deep). Stage 1 actually
  drew 0.506 shallow and 0.175 deep.

**Interpretation.** The gain on held-out instances is real on this evidence and corroborated by
three measures the scorer does not touch; it is still one sample, two looks and shared
structures. The timeout distribution is bimodal, so a shorter limit changes no outcome; 45 s
clears every successful call seen in either dataset (max 32.6 s). The depth gap between
held-out structure and training follows from the C2 change I recommended; reporting by hop
fixes the reading without new data.
If the dedicated shape matched Tinker's wall times, 37 steps of 8 × 8 would cost $41.13 at
$39 per hour ($25.85 at the median step time). That step time is unmeasured.
I nearly reported a curriculum between stages from file order in `train_300.jsonl` (0.400 then
0.603 shallow); the builder shuffles and the run's summaries show 0.506, so it was not sent as
a finding.

**Decision.** Sent all of the above; ledger F73–F75. Peer will adopt a 45 s training limit
from the next run, declared, evaluation unchanged at 120 s.

## 2026-10-03 — Preflight items closed from source and docs; a worse worst case found

**Observed.**
- The SDK builds the RL sampler deployment with min replicas = max replicas = 1
  (`managed.py`). The API reference: `scaleToZeroWindow` applies only "if
  min_replica_count==0", default 1h, at least 5 minutes; `expireTime` is "deprecated and no
  longer causes auto-deletion". So an abandoned sampler deployment has no automatic stop.
- `DeploymentManager.update(deployment_id, body, update_mask)` PATCHes a deployment.
- Preemptible deployments (docs): borrow idle reserved GPUs; "No charge to hold dedicated
  capacity"; reclaimable at any time; evaluation and batch only; work for promoted models
  including imported adapters; example targets a trained model, "not a base model". Serving
  cost not stated. SDK: the flag "requires an admin API key".
- After a dedicated run ends, evaluation is promote + deployment of the promoted model.

**Interpretation.** My earlier worst cases ($5.41, then $4.33 + up to $13) were both wrong: the
deployment would bill $13 per hour, $312 per day, until deleted. I had assumed scale-to-zero
applied without reading how the SDK sets replicas. Preemptible deployments may make evaluation
of a trained or imported adapter nearly free, which bears on both option A and option B; that
is inference from the docs, untested.

**Decision.** The shim now patches the deployment to min replicas 0 with a 300 s
scale-to-zero window immediately after provisioning and deletes everything if the patch fails
(45 of 45 offline tests). With it the laptop-death leak is $5.42. The provisioning window
before the patch is covered only by something off this machine; a scheduled deletion elsewhere
needs the owner's credentials there, so it is his decision. Peer's third pre-registration
addendum (a later run on the 186 is a separate labelled result; data changes only from
evidence that is not the 186) recorded on her side at my request. Ledger F76–F77.

## 2026-10-03 — Milestone 1 (Tinker, Qwen3.5-9B, RL from base) reproduced; bar not met

**Observed (`milestone1_check.py`, output `~/bird_rl_runs/milestone1_check_fireworks.json`).**
- Selection on 264 held-out instances, one sample: 0.5038 base, 0.5341, 0.6023, 0.6364, 0.6780
  at steps 10, 20, 30, 37. Step 37 minus base +0.1742 (+0.1061 to +0.2424). Mean turns 4.60 →
  3.67, turn-cap hits 82 → 45, query errors 229 → 176.
- The 186 at four samples: base 0.4207, step 37 0.4664, paired difference +0.0457, 95%
  bootstrap over questions +0.0094 to +0.0833. Pre-registered bar (point ≥ 0.05 and lower
  bound > 0): **not met**, short by 0.0043. Gap to 0.608 closed: 0.244. Simple 0.4768 → 0.5248,
  moderate 0.1833 → 0.2333, challenging 0.15 → 0.10 (5 questions). Lenient 0.5739 → 0.5766.
  The four step-37 passes are independent. 0 harness errors.
- Unseen structures (360, one sample): +0.1528 (+0.0944 to +0.2111); novel_combination +0.1854,
  novel_component +0.1292. Two hops: unseen +0.0917 (−0.0183 to +0.2018), seen +0.1505
  (+0.043 to +0.2581).
- Group breakdown, unbiased construction (groups on two baseline passes, baseline measured on
  the other two, mean over 6 splits): never 0.117 → 0.178 (+0.0293 of the mean), mixed 0.425 →
  0.511 (+0.0166), always 0.872 → 0.871 (−0.0002). With no training, the naive construction
  shows +0.117 and −0.128 for never and always.
- Lenient minus strict fell 0.1532 → 0.1102: 94.1% of the strict gain is form.
- The gain is 2.8 sampling standard errors; the miss is 0.27.

**Interpretation.** A gain on human questions was detected and is below the threshold we
signed. It is almost entirely answers becoming right in form; the policy did not answer more
questions. The gain on generated questions is 3.8 times larger, on seen and unseen structures
alike, so the result is a transfer gap. The breakdown I asked the peer for was biased by
regression to the mean; her reading that reliability was lost on previously solved questions
does not survive the unbiased version. The bar was to be read once; re-evaluating to clear it
is excluded.
Ledger ids F73–F77 were assigned twice by concurrent appends; I added a note and now number my
rows W1, W2, ….

**Decision.** Sent the reproduction and both corrections. Nothing to be acted on until the
owner has seen the result as it is. Ledger W1–W4.

## 2026-10-03 — Which suspect explains the transfer gap: a test with data on disk

**Context.** After milestone 1 the peer proposed that the two pre-registered suspects explain
different things: the question mix explains why generated-data skill does not carry to human
questions; the reward (blind to right values in the wrong shape) explains why what does carry
is form.

**Observed.** Splitting the strict gain at step 37 into form and substance with lenient
accuracy: generated, seen structures (264): strict +0.1742, lenient +0.1023 (+0.0417 to
+0.1629), substance 58.7%. Generated, unseen structures (360): strict +0.1528, lenient +0.1167
(+0.0556 to +0.1750), substance 76.4%. Human 186 at k=4: lenient +0.0027 (−0.0296 to +0.0349),
substance about 6%.

**Interpretation.** The same reward taught substance wherever the questions resemble the
training questions, including on unseen structures, and none on human questions. So the reward
is not what limits substance on the 186; the form habit generalised across question styles and
the substance did not. The suspects are not symmetric: the evidence points at the question
distribution. The generated-set percentages are single-sample point estimates.

**Decision.** Sent to the peer as evidence for the owner's decision: a second run changing the
reward would be aimed at the wrong thing; one changing the question mix is aimed at the right
thing, and its motivating measurement predates the result. Ledger W5.

## 2026-10-03 — Run 2 (human-like question mix) pre-registration reviewed

**Observed.** The owner returned, saw milestone 1 as it is, and approved a second Tinker run on
the question mix (relayed by the peer; her commit d0ab818). Tinker console balance $83.04
against $128: spend $44.96, inside her bounds of $40.79 to $60.91, implied cache share 0.793
(reproduced). She read LoRA alpha from the exported adapter: rank 32, alpha 32, no unembedding
tensors. `rl_env.py` is now 44f4db865e7c: a `TrainingCypherTool` with a 45 s transaction
timeout; evaluation unchanged at 120 s. Shim re-pinned to d0ab818; 45 of 45 tests pass.
Amendment D sets a 320-question training set to my measured human mix; held-out sets frozen.
`GRAPH_MODEL.md`: `TAGGED` is parsed from the `posts.Tags` column.

**Interpretation.** My "upper bound" claim about relational hops was wrong for this graph: a
column modelled as a node adds hops. About 70% at ≤1 hop is a reasonable target. The fixed
step-37 checkpoint is sound. Four weaknesses in the pre-registration: the hypothesis (the mix
matters) is tested against the baseline when it is a claim about run 2 against run 1; one run
per arm cannot separate mix from run-to-run variance; three things change, not one (file, 45 s
limit, checkpoint rule), and the limit also covers the reward's uncapped re-execution; and
count versus count distinct, which I had flagged as not comparable across SQL and Cypher,
became two separate targets.

**Decision.** Sent all of it before the data exists. Ledger W6 (my correction), W7.

## 2026-10-03 — Seed repeat or longer run: priced, and a third option

**Observed.** The peer accepted all four objections into a revised E9 before any run-2 data was
used. She asked whether a longer run for both arms is a better use of the Tinker budget than a
seed repeat. Priced from her run folders at her uncached estimates and at the measured billing
ratio 0.738: run 2 with all evaluations about $33.64; a seed repeat of run 1 on the 186 only
about $28.82; 37 more steps for one arm the same; for both arms about $57.65. From $83.04:
about $49.40 after run 2, $20.58 after a seed repeat, −$8.25 after a longer run of both arms.
600 old-mix training questions exist, 296 used; the new-mix file has 320.

**Interpretation.** A longer run of both arms does not fit. The right choice after run 2
depends on its result, so the rule should be fixed now: seed repeat if the run-2-minus-run-1
lenient interval touches zero; a longer run of the better arm if the difference is clear; a
longer run of the run 1 arm if run 2 is not better. A Fireworks replication of run 1 is itself a
seed repeat on a different platform, is what the owner said he wants, and is paid from credits
that do not compete with Tinker's; it settles the variance question if it lands near run 1 and
leaves seed and platform confounded if it does not.

**Decision.** Sent to the peer with the decision rule and the third option. What stands between
the Fireworks replication and a result: the timing probe, the deployment patch untested against
the real API, and an evaluation path for the trained adapter. Ledger W8.

## 2026-10-03 — Decision rule adopted; probe preflight drafted

**Observed.** The peer adopted the three-branch rule for what follows run 2, as an E9 addendum
recorded before run 2 exists, and cites my W8 figures as mine and not yet recomputed by her. She
records the Fireworks replication as outside the rule because it is not runnable yet, and has
told the owner nothing on my behalf.

**Decision.** Wrote `PROBE_PREFLIGHT.md` as a draft: what the probe measures, its cost
($8.67 planned, $15.17 if provisioning is billed, $20 cap requested), the four rented-GPU
questions answered from the code, the three layers that stop the meter, the known gaps, and the
condition that the first run is attended. Nothing is approved and nothing has been run. One gap
added while writing it: phase A needs a deployment-only sampling path for the 9B that the
baseline wrapper does not have yet.
