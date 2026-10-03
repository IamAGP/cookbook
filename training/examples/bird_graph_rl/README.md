# bird_graph_rl on Fireworks

The Fireworks side of a two-platform experiment: post-training a small open model to answer
questions by writing Cypher against a Neo4j graph (BIRD-SQL `codebase_community`), with the
identical experiment run on Tinker by a second agent.

**The idea of this folder: run the other side's code unchanged and swap only the client.** The
evaluation harness, RL environment, reward and training entry point live in the Tinker-side
repository. They are imported from a checkout of it and gated on git blob hashes, so the two
platforms cannot drift apart silently. This folder holds only what Fireworks needs that Tinker
does not, plus the checks and analyses done on this side.

## What is here

| File | Purpose |
|---|---|
| `baseline_eval_fireworks.py` | Runs the shared evaluation harness with a Fireworks serverless base-model sampler. |
| `rl_train_fireworks.py` | Runs the shared RL entry point on Fireworks. Bridges the two client calls Fireworks does not support, shortens checkpoint names to the serverless limit, and adds a cost meter with a hard stop. A dedicated (hourly) path exists behind explicit guards. |
| `fw_watchdog.py` | Independent process that deletes hourly resources on deadline, stall or orphaning, and verifies. |
| `PROBE_PREFLIGHT.md` | Draft preflight for a dedicated timing probe. Not approved; nothing has been run. |
| `sql_shapes.py`, `sql_shapes_crosscheck.py` | Coarse shape of human-written SQL in BIRD's filtered training split, as a check on generated training questions. |
| `datagen_join.py` | Joins generated structures to questions and compares them with the human shapes. |
| `baseline_k_analysis.py`, `milestone1_check.py`, `compare_runs.py` | Independent reproductions of results and the noise floor of the success bar. |
| `cost_estimates.py`, `student_options.py`, `db_time_measure.py` | Every cost and timing figure, computed from listed inputs. |
| `verify_partial_credit.py`, `verify_reward_pin.py` | Reproductions of the shared reward on stored queries. |
| `*_test.py` | Offline tests (no network, no spend). |
| `JOURNAL.md` | Dated lab notebook: observed, interpretation, decision. |

## What has and has not been run

Run on Fireworks: a zero-shot baseline of Qwen3.8-27B on the 186 benchmark questions through
the shared harness (197 rollouts in all, $3.12). Nothing else. No training step and no dedicated
hardware. The RL path and the watchdog are tested offline only and have never made a real
Fireworks call. `JOURNAL.md` records this as it changes.

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
