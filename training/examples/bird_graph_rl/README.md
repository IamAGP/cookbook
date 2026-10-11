# bird_graph_rl on Fireworks

RL post-training of an open model to answer questions by writing Cypher against a Neo4j graph
(BIRD-SQL `codebase_community`), run on Fireworks' serverless Training API.

**Start with the write-up:** [published post](https://adithyag73.github.io/first_principles/bird-graph-rl-fireworks/), source in [`blog/draft.md`](blog/draft.md). Result in one line: Qwen 3.8 27B,
trained for 37 updates on generated questions, gains 6.2 points of strict accuracy on 186
held-out human questions (95% interval 3.1 to 9.4) against an untrained adapter served the same
way.

A sibling experiment on a smaller model was run on another platform by a second agent; this
folder and its write-up cover the Fireworks run only.

**The idea of this folder: run the other side's code unchanged and swap only the client.** The
evaluation harness, RL environment, reward and training entry point live in the Tinker-side
repository. They are imported from a checkout of it and gated on git blob hashes, so the two
platforms cannot drift apart silently. This folder holds only what Fireworks needs that Tinker
does not, plus the checks and analyses done on this side.

## What is here

| File | Purpose |
|---|---|
| `baseline_eval_fireworks.py` | Runs the shared evaluation harness on Fireworks serverless, for the base model or for a saved training checkpoint loaded into a fresh session. |
| `rl_train_fireworks.py` | Runs the shared RL entry point on Fireworks. Bridges the two client calls Fireworks does not support, shortens checkpoint names to the serverless limit, and adds a cost meter with a hard stop. A dedicated (hourly) path exists behind explicit guards. |
| `fw_watchdog.py` | Independent process that deletes hourly resources on deadline, stall or orphaning, and verifies. |
| `PROBE_PREFLIGHT.md` | First preflight for a dedicated timing probe (superseded by the journal entry FW-P1). |
| `sql_shapes.py`, `sql_shapes_crosscheck.py` | Coarse shape of human-written SQL in BIRD's filtered training split, as a check on generated training questions. |
| `datagen_join.py` | Joins generated structures to questions and compares them with the human shapes. |
| `baseline_k_analysis.py`, `milestone1_check.py`, `compare_runs.py` | Independent reproductions of results and the noise floor of the success bar. |
| `cost_estimates.py`, `student_options.py`, `db_time_measure.py` | Every cost and timing figure, computed from listed inputs. |
| `verify_partial_credit.py`, `verify_reward_pin.py` | Reproductions of the shared reward on stored queries. |
| `launch_serverless.sh`, `eval_passes.sh`, `launch_dedicated.sh` | Detached launchers for a training run, a sequence of evaluation passes, and the guarded dedicated path. |
| `run_recorder.py` | Redraws curves and a per-update table during a run and mirrors the run folder. |
| `make_untrained_state.py` | Saves a never-trained adapter's state for the serving-route control. |
| `fw_r1_check.py`, `fw_e3_check.py` | The pre-registered tests: trained against base, and the three-arm test with the control. |
| `run2_check.py`, `hint_overlap.py`, `cypher_shapes.py` | Reproductions and checks done for the sibling experiment. |
| `record_transcripts.py` | Verbatim agent transcripts for the post's trajectory viewer (runs the peer's recorder with Fireworks sessions). |
| `blog/` | The write-up, its figures, the script that draws them, and the numbers behind each figure. |
| `*_test.py` | Offline tests (no network, no spend). |
| `JOURNAL.md` | Dated lab notebook: observed, interpretation, decision. |

## What has been run

All on the serverless (per-token) route; no dedicated hardware ever ran. Details, preflights
and every billed action are in `JOURNAL.md`, entries FW-T1 to FW-E3.

- A two-update plumbing trial of the RL loop.
- One training run: 37 updates of 8 questions x 8 rollouts on Qwen 3.8 27B.
- Twelve evaluation passes on the 186 benchmark questions: four of the base model, four of a
  never-trained adapter (serving-route control), four of the trained adapter.
- 24 verbatim transcripts of the post's three worked examples, for its trajectory viewer.
- A request for dedicated GPUs for a smaller model, which the account could not make without a
  payment method; nothing ran.

## Setup

```bash
export TINKER_COOKBOOK_DIR=/path/to/tinker-cookbook      # checkout at the pinned commit
export BIRD_REFERENCE_URI=s3://<bucket>/<prefix>/gold/reference_answers.json
export BIRD_FW_ENV_FILE=.env                             # holds FIREWORKS_API_KEY
PYTHONPATH=$TINKER_COOKBOOK_DIR python -m unittest \
    rl_train_fireworks_test.py fw_watchdog_test.py sql_shapes_test.py
```

The graph itself, the reference answers and the shared ledger of checkable facts are described
in the Tinker-side repository (`tinker_cookbook/recipes/bird_graph_rl/`).

Data: BIRD-SQL is CC BY-SA 4.0 and its content is Stack Exchange user content; attribution and
share-alike apply to anything derived from it.
