#!/bin/zsh
# Run evaluation passes one after another, detached. Each pass is the peer's harness through
# baseline_eval_fireworks.py; a pass that already has a .done marker is skipped.
#   eval_passes.sh <group-name> <budget_usd_per_pass> <pass-name>=<extra args, comma separated> ...
# Example: eval_passes.sh fw27b_base 6 fw27b_base_s2=reference_uri=$REF fw27b_base_s3=reference_uri=$REF
set -u
GROUP=$1; BUDGET=$2; shift 2
HERE=${0:A:h}; ROOT=${HERE:h}
: ${TINKER_COOKBOOK_DIR:?set TINKER_COOKBOOK_DIR}
R=$HOME/bird_rl_runs
cd $ROOT
stamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $R/$GROUP.passes.log; }
stamp "start: $# passes, budget $BUDGET each"
for spec in "$@"; do
  name=${spec%%=*}; extra=(${(s:,:)${spec#*=}})
  if [ -e "$R/$name.done" ]; then stamp "$name already done, skipped"; continue; fi
  PYTHONPATH=$TINKER_COOKBOOK_DIR caffeinate -i .venv/bin/python bird_graph_rl/baseline_eval_fireworks.py \
    base_model=Qwen/Qwen3.8-27B env_file=$TINKER_COOKBOOK_DIR/.env log_path=$R/$name budget_usd=$BUDGET \
    concurrency=${EVAL_CONCURRENCY:-16} "${extra[@]}" > $R/$name.log 2>&1
  rc=$?; stamp "$name exited rc=$rc"
  [ $rc -eq 0 ] && touch $R/$name.done
done
stamp "all passes attempted"; touch $R/$GROUP.passes.done
