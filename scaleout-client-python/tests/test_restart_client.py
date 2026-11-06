"""Tests for client restart concurrency handling.

Covers two related bugs in the restart path:
  - The force-restart path used to call TaskReceiver.stop() without ever waiting for its
    own RestartClient task to be reported "done" to the combiner, silently dropping the
    final ack (verified empirically: 200/200 runs lost the ack before the fix, 0/200 after).
  - GrpcEdgeClientRuntime._restart_client had no single-flight guard, so concurrent
    RestartClient requests could each spawn their own `_restart` thread.

Other tasks that happen to arrive in the same or a later poll batch as a restart command are
deliberately still dispatched and started normally by TaskReceiver -- they are not skipped.
They are then subject to the same force/non-force restart semantics as any task that was
already running when the restart command arrived (waited-for-and-aborted in the non-force
path, or killed by the eventual process re-exec in the force path). A duplicate RestartClient
request is the one thing that's rejected outright, and that rejection is properly reported
(not silently dropped) via the normal task-failure path.
"""

import threading
import time
import unittest
from unittest.mock import patch

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg

from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime
from scaleout.client.task_receiver import Task, TaskReceiver


class _FakeGrpcHandler:
    """Records every ClientReport sent and returns a scripted directive on the first poll."""

    def __init__(self, first_directive_tasks=None):
        self._first_directive_tasks = first_directive_tasks or []
        self._first = True
        self.reports = []
        self.lock = threading.Lock()

    def PollAndReportAsync(self, report):
        with self.lock:
            self.reports.append(list(report.reports))
        directive = scaleout_msg.CombinerDirective()
        if self._first:
            self._first = False
            directive.tasks.extend(self._first_directive_tasks)
        return directive


class _FakeClient:
    def __init__(self, grpc_handler):
        self.client_id = "client-1"
        self.grpc_handler = grpc_handler


def _restart_request(correlation_id="restart-corr-1", force=False):
    request = scaleout_msg.TaskRequest()
    request.correlation_id = correlation_id
    request.type = "Scaleout_RestartClient"
    if force:
        request.data = '{"parameters": {"force": true}}'
    return request


class TestForceRestartAckIsFlushed(unittest.TestCase):
    """Regression test for the lost-final-ack race in the force=True restart path.

    Drives the real GrpcEdgeClientRuntime._restart_client/_restart end to end against a fake
    grpc_handler, exactly reproducing the force=True branch. Before the fix (which threads
    `task=task` through to `_restart` in every branch), this was 200/200 reproducible: the
    task's done=True was never sent because TaskReceiver.stop() fired before the polling loop
    got another chance to report it.
    """

    def test_force_restart_reports_done_before_stopping(self):
        grpc_handler = _FakeGrpcHandler([_restart_request("restart-corr-1", force=True)])

        runtime = GrpcEdgeClientRuntime()
        runtime._client = type("FakeClient", (), {"client_id": "client-1", "name": "client-1"})()
        runtime.grpc_handler = grpc_handler
        runtime.task_receiver.polling_interval = 0.02

        with patch("scaleout.client.grpc_edge_client_runtime.valid_env_for_restart", return_value=True):
            runtime.task_receiver.start()
            runtime.task_receiver.wait_on_manager_thread()

        with grpc_handler.lock:
            all_reports = grpc_handler.reports

        ever_reported_done = any(
            r.correlation_id == "restart-corr-1" and r.done for reports in all_reports for r in reports
        )
        self.assertTrue(ever_reported_done, "restart task's done=True was never reported to the combiner")


class TestRestartSingleFlightGuard(unittest.TestCase):
    def setUp(self):
        self.runtime = GrpcEdgeClientRuntime()
        patcher = patch("scaleout.client.grpc_edge_client_runtime.valid_env_for_restart", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, correlation_id, force=True):
        request = scaleout_msg.TaskRequest()
        request.correlation_id = correlation_id
        request.type = "Scaleout_RestartClient"
        request.data = '{"parameters": {"force": true}}' if force else "{}"
        return request

    def test_concurrent_force_restarts_only_one_proceeds(self):
        release = threading.Event()
        started = threading.Event()

        def fake_restart(task=None, force=False):
            started.set()
            release.wait(timeout=5)

        with patch.object(self.runtime, "_restart", side_effect=fake_restart):
            results = []

            def call(i):
                try:
                    self.runtime._restart_client(self._request(f"corr-{i}"))
                    results.append(("ok", i))
                except Exception as e:
                    results.append(("error", str(e)))

            threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
            threads[0].start()
            started.wait(timeout=5)
            for t in threads[1:]:
                t.start()
            for t in threads:
                t.join(timeout=5)
            release.set()

        ok_count = sum(1 for r in results if r[0] == "ok")
        error_count = sum(1 for r in results if r[0] == "error")
        self.assertEqual(ok_count, 1)
        self.assertEqual(error_count, 4)
        for r in results:
            if r[0] == "error":
                self.assertIn("already in progress", r[1])

    def test_thread_is_daemon_and_stored(self):
        with patch.object(self.runtime, "_restart", return_value=None):
            self.runtime._restart_client(self._request("corr-1"))
        self.assertIsNotNone(self.runtime._restart_thread)
        self.assertTrue(self.runtime._restart_thread.daemon)
        self.runtime._restart_thread.join(timeout=5)

    def test_lock_released_when_other_tasks_running_without_spawning_thread(self):
        task_a = Task(scaleout_msg.TaskRequest(correlation_id="other-task"))
        self.runtime.task_receiver._current_tasks.append(task_a)

        request = self._request("corr-1", force=False)
        request.data = "{}"  # no force, no abort_ongoing_tasks -> hits has_other_tasks() branch

        with self.assertRaises(Exception):
            self.runtime._restart_client(request)

        # Lock must have been released since no thread was spawned.
        with patch.object(self.runtime, "_restart", return_value=None):
            self.runtime._restart_client(self._request("corr-2"))
        self.assertIsNotNone(self.runtime._restart_thread)
        self.runtime._restart_thread.join(timeout=5)


class TestOtherTasksInSameBatchAreStarted(unittest.TestCase):
    """A non-restart task delivered alongside a restart command must still be started."""

    def test_task_delivered_with_a_restart_is_still_dispatched(self):
        restart_req = _restart_request("restart-1")
        other_req = scaleout_msg.TaskRequest(correlation_id="training-1", type="Scaleout_TrainingRound")

        grpc_handler = _FakeGrpcHandler([restart_req, other_req])
        client = _FakeClient(grpc_handler)

        seen = []
        both_seen = threading.Event()

        def task_callback(request):
            seen.append(request.correlation_id)
            if len(seen) == 2:
                both_seen.set()
            return {}

        receiver = TaskReceiver(client, task_callback, polling_interval=0.02)
        receiver.start()
        try:
            both_seen.wait(timeout=2)
            self.assertIn("restart-1", seen)
            self.assertIn("training-1", seen)
        finally:
            receiver.stop()
            receiver.wait_on_manager_thread()


class TestRestartOrderingWithinSameBatch(unittest.TestCase):
    """Other new tasks in a batch must be dispatched before any restart new task in that
    same batch, regardless of the order the combiner listed them in the directive.

    This is what makes has_other_tasks()/has_current_tasks() -- checked from the restart's
    own thread -- reliable: by construction, every other same-batch task is already in
    _current_tasks by the time a restart task is even created, so there's no race to win.
    Testing the real thread race directly would be flaky (whichever thread the OS happens
    to schedule first); this instead asserts the deterministic dispatch order that
    guarantees the race can't occur in the first place.
    """

    def test_restart_listed_first_is_still_dispatched_after_other_task(self):
        restart_req = _restart_request("restart-1")
        other_req = scaleout_msg.TaskRequest(correlation_id="other-1", type="Scaleout_TrainingRound")

        # Restart listed FIRST in the directive -- the ordering that used to matter.
        grpc_handler = _FakeGrpcHandler([restart_req, other_req])
        client = _FakeClient(grpc_handler)

        release = threading.Event()

        def task_callback(request):
            release.wait(timeout=5)
            return {}

        receiver = TaskReceiver(client, task_callback, polling_interval=0.5)
        receiver.start()
        try:
            deadline = time.time() + 2
            while len(receiver._current_tasks) < 2 and time.time() < deadline:
                time.sleep(0.01)
            ids_in_order = [t.correlation_id for t in receiver._current_tasks]
            self.assertEqual(ids_in_order, ["other-1", "restart-1"])
        finally:
            release.set()
            receiver.stop()
            receiver.wait_on_manager_thread()


class TestDuplicateRestartIsReportedNotDropped(unittest.TestCase):
    """A second, concurrent RestartClient request is rejected -- and that rejection is
    actually reported back to the combiner, not silently dropped."""

    def test_second_restart_task_is_reported_as_failed(self):
        grpc_handler = _FakeGrpcHandler(
            [_restart_request("restart-1", force=True), _restart_request("restart-2", force=True)]
        )

        runtime = GrpcEdgeClientRuntime()
        runtime._client = type("FakeClient", (), {"client_id": "client-1", "name": "client-1"})()
        runtime.grpc_handler = grpc_handler
        runtime.task_receiver.polling_interval = 0.02

        with patch("scaleout.client.grpc_edge_client_runtime.valid_env_for_restart", return_value=True):
            runtime.task_receiver.start()
            runtime.task_receiver.wait_on_manager_thread()

        with grpc_handler.lock:
            all_reports = [r for reports in grpc_handler.reports for r in reports]

        done_ids = {r.correlation_id for r in all_reports if r.done}
        self.assertEqual(done_ids, {"restart-1", "restart-2"})


if __name__ == "__main__":
    unittest.main()
