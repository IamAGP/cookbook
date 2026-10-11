"""Keep a durable record of a training run while it is in progress.

Every ``--every`` seconds, until ``<run>.done`` appears: redraw the curves from ``metrics.jsonl``
and ``fw_cost_meter.jsonl`` into ``<run>/curves.png``, rewrite ``<run>/curves.csv`` (one row per
update), and mirror the run folder and its console log to object storage.

What is and is not in the record: the loop logs reward, kept-group share, turns, sampling
entropy and the sampling-versus-training KL per update. It does not log a loss value. Adapter
weights stay on Fireworks (the SDK cannot download them); ``checkpoints.jsonl`` holds the paths
that load them back.

    python bird_graph_rl/run_recorder.py ~/bird_rl_runs/fw_rl_27b_run1 --s3 s3://bucket/prefix/runs
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = {
    "reward_kept_groups": "env/all/reward/total",
    "turns_per_episode": "env/all/turns_per_episode", "sampling_entropy": "optim/entropy",
    "kl_sample_train": "optim/kl_sample_train_v2", "learning_rate": "optim/lr",
    "seconds_total": "time/total", "seconds_sampling": "time/sampling", "seconds_train_step": "time/train_step",
}


def rows(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    cost: dict[int, float] = {}
    tokens: dict[int, int] = {}
    meter = run / "fw_cost_meter.jsonl"
    if meter.exists():
        for line in meter.read_text().splitlines():
            m = json.loads(line)
            if m["event"] == "forward_backward":
                cost[m["train_calls"]] = m["usd_upper"]
                tokens[m["train_calls"]] = m["tokens"]
    # Metrics of each forward/backward call, written by the wrapper's RecordingFuture. With one
    # training call per update (num_substeps=1) the call index is the update number.
    train: dict[int, dict] = {}
    extra = run / "fw_train_metrics.jsonl"
    if extra.exists():
        for line in extra.read_text().splitlines():
            m = json.loads(line)
            train[m["train_call"]] = m["metrics"]
    out = []
    for line in path.read_text().splitlines():
        m = json.loads(line)
        row = {"update": m["step"] + 1, **{name: m.get(key) for name, key in COLUMNS.items()}}
        row["meter_usd_upper_cumulative"] = cost.get(m["step"] + 1)
        fb = train.get(m["step"] + 1, {})
        row["trained_tokens"] = tokens.get(m["step"] + 1)
        row["loss_sum"] = fb.get("loss:sum")
        row["loss_per_trained_token"] = (fb["loss:sum"] / tokens[m["step"] + 1]
                                         if "loss:sum" in fb and tokens.get(m["step"] + 1) else None)
        row.update({f"fb/{k}": v for k, v in fb.items() if k != "loss:sum"})
        out.append(row)
    return out


def from_rollouts(run: Path, update: int) -> dict:
    """Figures over every rollout of an update, counted from the rollout summaries.

    The loop's own ``by_group`` metrics are computed after constant-reward groups have been
    dropped, so they always read "all groups mixed". The share actually kept for training has
    to be counted here: a group is dropped when all of its rollouts got the same reward.
    """
    path = run / f"iteration_{update - 1:06d}" / "train_rollout_summaries.jsonl"
    if not path.exists():
        return {}
    groups: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        groups.setdefault(r["group_idx"], []).append(r["total_reward"])
    if not groups:
        return {}
    rewards = [x for g in groups.values() for x in g]
    n = len(groups)
    constant = [g for g in groups.values() if len(set(g)) == 1]
    return {"reward_all_rollouts": sum(rewards) / len(rewards), "rollouts": len(rewards), "groups": n,
            "share_groups_kept": (n - len(constant)) / n,
            "share_groups_all_correct": sum(1 for g in constant if g[0] == 1.0) / n,
            "share_groups_all_wrong": sum(1 for g in constant if g[0] == 0.0) / n}


def record(run: Path) -> int:
    data = rows(run)
    if not data:
        return 0
    for row in data:
        row.update(from_rollouts(run, row["update"]))
    with (run / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in data for k in r}, key=lambda k: (k != "update", k)))
        writer.writeheader()
        writer.writerows(data)
    x = [r["update"] for r in data]
    panels = [("Loss (importance-sampling surrogate), sum per update", ["loss_sum"]),
              ("Reward per update", ["reward_all_rollouts", "reward_kept_groups"]),
              ("Share of question groups", ["share_groups_kept", "share_groups_all_correct", "share_groups_all_wrong"]),
              ("Turns per episode", ["turns_per_episode"]), ("Sampling entropy", ["sampling_entropy"]),
              ("KL, sampling vs training", ["kl_sample_train"]),
              ("Seconds per update", ["seconds_total", "seconds_sampling", "seconds_train_step"]),
              ("Meter, cumulative USD (upper bound)", ["meter_usd_upper_cumulative"])]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, names) in zip(axes.flat, panels):
        for name in names:
            ys = [r.get(name) for r in data]
            if any(y is not None for y in ys):
                ax.plot(x, ys, marker="o", markersize=3, label=name)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{run.name}: {len(data)} updates, drawn {time.strftime('%Y-%m-%d %H:%M:%S')}")
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=110)
    plt.close(fig)
    return len(data)


def mirror(run: Path, s3: str) -> str:
    target = f"{s3.rstrip('/')}/{run.name}"
    a = subprocess.run(["aws", "s3", "sync", str(run), target, "--only-show-errors"], capture_output=True, text=True)
    log = run.parent / f"{run.name}.log"
    b = subprocess.run(["aws", "s3", "cp", str(log), f"{target}.log", "--only-show-errors"], capture_output=True, text=True) if log.exists() else None
    errors = (a.stderr + (b.stderr if b else "")).strip()
    return errors[:200] or "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--s3", default="")
    ap.add_argument("--every", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    done = run.parent / f"{run.name}.done"
    while True:
        finished = done.exists()  # read before recording, so the last pass sees the final files
        n = record(run)
        status = mirror(run, args.s3) if args.s3 else "no mirror"
        print(f"[{time.strftime('%H:%M:%S')}] updates recorded: {n}; mirror: {status}", flush=True)
        if args.once or finished:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
