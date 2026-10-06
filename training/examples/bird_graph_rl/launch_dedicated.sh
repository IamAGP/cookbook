#!/bin/zsh
# Launch one dedicated (hourly) Fireworks RL run with its independent watchdog.
#   launch_dedicated.sh <name> <max_steps> <deadline_min> <budget_usd> [extra train args...]
# Both resource ids derive from <name> and are known before anything is requested.
# Requires the owner's approval and a written preflight in JOURNAL.md. Never run on defaults.
set -u
NAME=$1; STEPS=$2; DEADLINE_MIN=$3; BUDGET=$4; shift 4
HERE=${0:A:h}; ROOT=${HERE:h}
: ${TINKER_COOKBOOK_DIR:?set TINKER_COOKBOOK_DIR}
R=$HOME/bird_rl_runs; LOG=$R/$NAME
if [ -e "$LOG" ]; then echo "refusing: $LOG exists"; exit 2; fi
mkdir -p $LOG
DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_MIN * 60 ))
cd $ROOT
echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch $NAME steps=$STEPS deadline_min=$DEADLINE_MIN budget=$BUDGET deployment=$NAME trainer=$NAME-tr" | tee -a $R/$NAME.launch.log

FW_ALLOW_DEDICATED=1 FW_DEPLOYMENT_ID=$NAME FW_TRAINER_JOB_ID=$NAME-tr FW_DEADLINE_MIN=$DEADLINE_MIN \
FW_INACTIVITY_MIN=${FW_INACTIVITY_MIN:-20} FW_PENDING_MIN=${FW_PENDING_MIN:-10} FW_READY_MIN=${FW_READY_MIN:-20} \
FW_BUDGET_USD=$BUDGET PYTHONPATH=$TINKER_COOKBOOK_DIR \
  caffeinate -i .venv/bin/python bird_graph_rl/rl_train_fireworks.py model_name=Qwen/Qwen3.5-9B \
  instances_path=$R/datagen_v3/train_300.jsonl max_steps=$STEPS batch_size=8 group_size=8 \
  test_split=None eval_every=0 env_file=$TINKER_COOKBOOK_DIR/.env log_path=$LOG \
  behavior_if_log_dir_exists=resume "$@" > $R/$NAME.log 2>&1 &
TRAIN_PID=$!
PYTHONPATH=$TINKER_COOKBOOK_DIR caffeinate -i .venv/bin/python bird_graph_rl/fw_watchdog.py --log-dir $LOG \
  --deployment-id $NAME --trainer-job-id $NAME-tr --deadline-epoch $DEADLINE_EPOCH \
  --stall-min ${FW_STALL_MIN:-10} --pid $TRAIN_PID > $R/$NAME.watchdog.log 2>&1 &
WATCH_PID=$!
echo "train pid $TRAIN_PID, watchdog pid $WATCH_PID" | tee -a $R/$NAME.launch.log
wait $TRAIN_PID; RC=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] training exited rc=$RC" | tee -a $R/$NAME.launch.log
wait $WATCH_PID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] watchdog exited rc=$?" | tee -a $R/$NAME.launch.log
tail -3 $R/$NAME.watchdog.log
