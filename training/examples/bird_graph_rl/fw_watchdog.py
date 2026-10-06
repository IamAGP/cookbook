"""Independent watchdog for a dedicated (hourly-billed) Fireworks run.

Runs as its own process, so it still works when the training process hangs, crashes or is
killed. It deletes the trainer job and the deployment through the SDK managers when any of
these holds, then verifies that nothing is left:

- deadline:  the wall-clock deadline has passed;
- stall:     resources exist and the run has written no progress line for ``--stall-min``;
- orphan:    resources exist and the training process is gone without having cleaned up.

Deletion goes through the SDK because ``firectl`` refuses mutating commands inside an agent
session. Listing with ``firectl`` is read-only and is used to find a trainer whose id was never
recorded (a hang during provisioning) and to confirm that zero resources remain.

What this cannot cover: if the laptop itself sleeps or dies, this process dies with it. That
case is covered only by the server-side settings the run requests (trainer inactivity timeout,
deployment scale-to-zero window), which is why those are set explicitly and never left to
defaults.

    python bird_graph_rl/fw_watchdog.py --log-dir <dir> --deployment-id <id> \\
        --deadline-epoch <unix seconds> --stall-min 5 --pid <training pid>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

PROGRESS_FILES = ("fw_cost_meter.jsonl", "metrics.jsonl")


def decide(*, now: float, deadline_epoch: float, resources_exist: bool, last_progress: float | None,
           stall_s: float, run_alive: bool, watch_started: float) -> str | None:
    """Return the reason to tear down now, or None to keep waiting."""
    if now >= deadline_epoch:
        return "deadline"
    if not resources_exist:
        return None
    if not run_alive:
        return "orphan"
    reference = last_progress if last_progress is not None else watch_started
    if now - reference >= stall_s:
        return "stall"
    return None


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True  # nothing to check against; never report an orphan on no evidence
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def last_progress(log_dir: Path) -> float | None:
    times = [(log_dir / name).stat().st_mtime for name in PROGRESS_FILES if (log_dir / name).exists()]
    return max(times) if times else None


def read_resources(log_dir: Path) -> dict[str, Any]:
    path = log_dir / "fw_resources.json"
    return json.loads(path.read_text()) if path.exists() else {}


def firectl_rows(*args: str) -> list[str]:
    """Data rows of a read-only ``firectl … list``; the header line is dropped."""
    out = subprocess.run(["firectl", *args], capture_output=True, text=True, timeout=60).stdout
    lines = [line for line in out.splitlines() if line.strip() and "firectl upgrade" not in line
             and "version" not in line.lower() and "updates available" not in line.lower()]
    return [line for line in lines if not line.lstrip().startswith("NAME")]


def remaining_resources(lister: Callable[..., list[str]] = firectl_rows) -> dict[str, list[str]]:
    return {"trainer_jobs": lister("rlor-trainer-job", "list"), "deployments": lister("deployment", "list")}


# A deleted resource can stay visible for a while in one of these states; it no longer bills.
GONE_STATES = frozenset({"DELETED", "DELETING", "JOB_STATE_DELETED", "JOB_STATE_DELETING", "JOB_STATE_ARCHIVED"})


def _state(resource: Any) -> str | None:
    if resource is None:
        return None
    value = resource.get("state") if isinstance(resource, dict) else getattr(resource, "state", None)
    return str(value) if value is not None else "UNKNOWN"


def _check(entry: dict[str, Any], fetch: Callable[[], Any], settle_s: float, poll_s: float) -> None:
    """Record whether the resource is gone, polling briefly because deletion is not instant."""
    waited = 0.0
    while True:
        try:
            state = _state(fetch())
        except Exception as e:  # noqa: BLE001
            entry["verify_error"] = f"{type(e).__name__}: {e}"[:200]
            return
        entry["state"] = state
        entry["still_exists"] = state is not None and state not in GONE_STATES
        if not entry["still_exists"] or waited >= settle_s:
            return
        time.sleep(poll_s)
        waited += poll_s


def delete_and_verify(trainer_mgr: Any, deploy_mgr: Any, trainer_job_ids: list[str], deployment_id: str | None,
                      settle_s: float = 60.0, poll_s: float = 5.0) -> dict[str, Any]:
    """Delete what was created and check it is gone. Never raises: every failure is reported."""
    report: dict[str, Any] = {"trainer": {}, "deployment": {}}
    for job_id in trainer_job_ids:
        entry: dict[str, Any] = {}
        try:
            trainer_mgr.delete(job_id=job_id)
            entry["deleted"] = True
        except Exception as e:  # noqa: BLE001
            entry["delete_error"] = f"{type(e).__name__}: {e}"[:200]
        _check(entry, lambda job_id=job_id: trainer_mgr.try_get(job_id=job_id), settle_s, poll_s)
        report["trainer"][job_id] = entry
    if deployment_id:
        entry = {}
        try:
            deploy_mgr.delete(deployment_id=deployment_id)
            entry["deleted"] = True
        except Exception as e:  # noqa: BLE001
            entry["delete_error"] = f"{type(e).__name__}: {e}"[:200]
        _check(entry, lambda: deploy_mgr.get(deployment_id=deployment_id), settle_s, poll_s)
        report["deployment"][deployment_id] = entry
    entries = list(report["trainer"].values()) + list(report["deployment"].values())
    report["clean"] = all(e.get("still_exists") is False for e in entries)
    return report


def write(log_dir: Path, event: dict[str, Any]) -> None:
    line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with (log_dir / "fw_watchdog.jsonl").open("a") as f:
        f.write(json.dumps(line, default=str) + "\n")
    print(json.dumps(line, default=str), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--deployment-id", required=True)
    ap.add_argument("--trainer-job-id", required=True)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--stall-min", type=float, required=True)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--poll-s", type=float, default=20.0)
    args = ap.parse_args()

    from dotenv import load_dotenv
    from fireworks.training.sdk import DeploymentManager, TrainerJobManager

    load_dotenv(os.environ.get("BIRD_FW_ENV_FILE", ".env"))
    key = os.environ["FIREWORKS_API_KEY"]
    trainer_mgr, deploy_mgr = TrainerJobManager(api_key=key), DeploymentManager(api_key=key)
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write(log_dir, {"event": "watching", "deployment_id": args.deployment_id, "trainer_job_id": args.trainer_job_id,
                    "deadline_epoch": args.deadline_epoch,
                    "stall_min": args.stall_min, "pid": args.pid})

    while True:
        res = read_resources(log_dir)
        finished = (log_dir / "fireworks_meta.json").exists()
        # The ids are known from the command line, so resources may exist before the run has
        # written anything. The stall clock only starts once provisioning has returned: waiting
        # for GPUs writes no progress lines and is bounded by the deadline instead.
        provisioned = bool(res.get("provisioned"))
        marks = [t for t in (last_progress(log_dir), (log_dir / "fw_resources.json").stat().st_mtime if provisioned else None) if t]
        reason = "run_finished" if finished else decide(
            now=time.time(), deadline_epoch=args.deadline_epoch, resources_exist=True,
            last_progress=max(marks) if marks else None,
            stall_s=args.stall_min * 60 if provisioned else float("inf"),
            run_alive=pid_alive(args.pid), watch_started=started)
        if reason is not None:
            jobs = sorted({args.trainer_job_id} | ({res["trainer_job_id"]} if res.get("trainer_job_id") else set()))
            report = delete_and_verify(trainer_mgr, deploy_mgr, jobs, args.deployment_id)
            write(log_dir, {"event": "teardown", "reason": reason, "report": report,
                            "remaining": remaining_resources()})
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
