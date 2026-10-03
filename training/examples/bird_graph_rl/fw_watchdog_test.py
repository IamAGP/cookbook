"""Offline tests for the watchdog's decisions and teardown. No network.

    .venv/bin/python -m unittest bird_graph_rl/fw_watchdog_test.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_watchdog as wd  # noqa: E402

BASE = dict(deadline_epoch=2000.0, stall_s=300.0, watch_started=1000.0)


class FakeTrainerMgr:
    def __init__(self, sticky: bool = False, fail: bool = False) -> None:
        self.jobs, self.sticky, self.fail, self.deleted = {"job-1"}, sticky, fail, []

    def delete(self, job_id: str) -> None:
        if self.fail:
            raise RuntimeError("503 from control plane")
        self.deleted.append(job_id)
        if not self.sticky:
            self.jobs.discard(job_id)

    def try_get(self, job_id: str) -> dict | None:
        return {"name": job_id} if job_id in self.jobs else None


class FakeDeployMgr:
    def __init__(self) -> None:
        self.deps, self.deleted = {"dep-1"}, []

    def delete(self, deployment_id: str) -> None:
        self.deleted.append(deployment_id)
        self.deps.discard(deployment_id)

    def get(self, deployment_id: str) -> dict | None:
        return {"id": deployment_id} if deployment_id in self.deps else None


class DecideTest(unittest.TestCase):
    def test_waits_while_healthy(self) -> None:
        self.assertIsNone(wd.decide(now=1100.0, resources_exist=True, last_progress=1090.0, run_alive=True, **BASE))

    def test_no_resources_means_nothing_to_tear_down_before_deadline(self) -> None:
        self.assertIsNone(wd.decide(now=1900.0, resources_exist=False, last_progress=None, run_alive=False, **BASE))

    def test_deadline_wins_even_without_recorded_resources(self) -> None:
        self.assertEqual(wd.decide(now=2000.0, resources_exist=False, last_progress=None, run_alive=True, **BASE), "deadline")

    def test_stall_is_measured_from_last_progress(self) -> None:
        self.assertIsNone(wd.decide(now=1389.0, resources_exist=True, last_progress=1090.0, run_alive=True, **BASE))
        self.assertEqual(wd.decide(now=1390.0, resources_exist=True, last_progress=1090.0, run_alive=True, **BASE), "stall")

    def test_stall_without_any_progress_is_measured_from_watch_start(self) -> None:
        self.assertEqual(wd.decide(now=1300.0, resources_exist=True, last_progress=None, run_alive=True, **BASE), "stall")

    def test_orphan_when_run_died_with_resources_up(self) -> None:
        self.assertEqual(wd.decide(now=1100.0, resources_exist=True, last_progress=1099.0, run_alive=False, **BASE), "orphan")

    def test_pid_alive(self) -> None:
        import os
        self.assertTrue(wd.pid_alive(os.getpid()))
        self.assertTrue(wd.pid_alive(None))
        self.assertFalse(wd.pid_alive(2 ** 22 + 12345))


class TeardownTest(unittest.TestCase):
    def test_deletes_both_and_reports_clean(self) -> None:
        t, d = FakeTrainerMgr(), FakeDeployMgr()
        report = wd.delete_and_verify(t, d, ["job-1"], "dep-1")
        self.assertEqual((t.deleted, d.deleted), (["job-1"], ["dep-1"]))
        self.assertTrue(report["clean"])

    def test_reports_not_clean_when_a_resource_survives(self) -> None:
        report = wd.delete_and_verify(FakeTrainerMgr(sticky=True), FakeDeployMgr(), ["job-1"], "dep-1")
        self.assertFalse(report["clean"])
        self.assertTrue(report["trainer"]["job-1"]["still_exists"])

    def test_a_failed_delete_does_not_stop_the_other_and_is_reported(self) -> None:
        d = FakeDeployMgr()
        report = wd.delete_and_verify(FakeTrainerMgr(fail=True), d, ["job-1"], "dep-1")
        self.assertIn("503", report["trainer"]["job-1"]["delete_error"])
        self.assertEqual(d.deleted, ["dep-1"])
        self.assertFalse(report["clean"])

    def test_unknown_trainer_id_still_removes_the_deployment(self) -> None:
        d = FakeDeployMgr()
        report = wd.delete_and_verify(FakeTrainerMgr(), d, [], "dep-1")
        self.assertEqual(d.deleted, ["dep-1"])
        self.assertTrue(report["clean"])

    def test_remaining_resources_uses_the_lister(self) -> None:
        calls = []
        out = wd.remaining_resources(lambda *a: calls.append(a) or [])
        self.assertEqual(out, {"trainer_jobs": [], "deployments": []})
        self.assertEqual(calls, [("rlor-trainer-job", "list"), ("deployment", "list")])


if __name__ == "__main__":
    unittest.main()
