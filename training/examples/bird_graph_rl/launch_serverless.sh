#!/bin/zsh
# Launch one serverless (per-token) Fireworks RL run, detached from the calling shell.
#   launch_serverless.sh <name> <max_steps> <budget_usd> [extra train args...]
set -u
NAME=$1; STEPS=$2; BUDGET=$3; shift 3
HERE=${0:A:h}; ROOT=${HERE:h}
: ${TINKER_COOKBOOK_DIR:?set TINKER_COOKBOOK_DIR}
R=$HOME/bird_rl_runs
# RESUME=1 continues an interrupted run from the last record in its checkpoints.jsonl.
RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  if [ ! -s "$R/$NAME/checkpoints.jsonl" ]; then echo "refusing: nothing to resume in $R/$NAME"; exit 2; fi
  RESUME_ARGS=(behavior_if_log_dir_exists=resume); rm -f $R/$NAME.done
elif [ -e "$R/$NAME" ]; then echo "refusing: $R/$NAME exists"; exit 2; fi
cd $ROOT
echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch $NAME steps=$STEPS budget=$BUDGET resume=${RESUME:-0}" >> $R/$NAME.launch.log
FW_BUDGET_USD=$BUDGET PYTHONPATH=$TINKER_COOKBOOK_DIR caffeinate -i .venv/bin/python bird_graph_rl/rl_train_fireworks.py \
  model_name=Qwen/Qwen3.8-27B instances_path=$R/datagen_v3/train_300.jsonl max_steps=$STEPS batch_size=8 group_size=8 \
  test_split=None eval_every=0 env_file=$TINKER_COOKBOOK_DIR/.env log_path=$R/$NAME "${RESUME_ARGS[@]}" "$@" >> $R/$NAME.log 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] exit rc=$?" >> $R/$NAME.launch.log
touch $R/$NAME.done
