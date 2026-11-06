"""Unit tests for TaskReceiver.start_autonomous_task and client-reported timestamps."""

import json
import threading
import time
import unittest
from datetime import datetime, timezone

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.grpc.statustype import StatusType

from scaleout.client.task_receiver import MAX_SELF_REGISTRATION_ATTEMPTS, StoppedException, Task, TaskReceiver


class _StubGrpcHandler:
    """Returns a canned directive and stops the polling loop after one iteration."""

    def __init__(self, directive, stop_event):
        self.directive = directive
        self.stop_event = stop_event
        self.received_reports = []

    def PollAndReportAsync(self, report):
        self.received_reports.append(report)
        self.stop_event.set()
        return self.directive


class _StubClient:
    def __init__(self, grpc_handler=None):
        self.client_id = "client-1"
        self.grpc_handler = grpc_handler


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestStartAutonomousTask(unittest.TestCase):
    def test_task_appears_in_current_tasks_and_completes(self):
        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {}, polling_interval=0)

        task = receiver.start_autonomous_task(lambda: {"ok": True}, "Inference", {"model_id": "m1"})

        self.assertIn(task, receiver._current_tasks)
        self.assertTrue(_wait_until(lambda: task.done))
        self.assertEqual(task.status, StatusType.COMPLETED)
        self.assertEqual(task.response, {"ok": True})
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.completed_at)
        self.assertIsNotNone(task.runner_thread)
        self.assertNotEqual(task.runner_thread, threading.current_thread())

    def test_activities_include_command_type_until_acked(self):
        release = threading.Event()
        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {}, polling_interval=0)
        task = receiver.start_autonomous_task(lambda: release.wait(timeout=2), "Inference", {"model_id": "m1"})

        self.assertTrue(_wait_until(lambda: task.started_at is not None))
        activities = receiver._get_current_activities()
        self.assertEqual(len(activities), 1)
        report = activities[0]
        self.assertEqual(report.correlation_id, task.correlation_id)
        self.assertEqual(report.command_type, "Inference")
        self.assertEqual(json.loads(report.parameters), {"model_id": "m1"})
        self.assertTrue(report.HasField("started_at"))

        task.acked_by_server = True
        activities_after_ack = receiver._get_current_activities()
        report_after_ack = activities_after_ack[0]
        self.assertFalse(report_after_ack.HasField("command_type"))
        self.assertFalse(report_after_ack.HasField("parameters"))

        release.set()
        self.assertTrue(_wait_until(lambda: task.done))

    def test_command_type_stops_being_sent_when_never_acked(self):
        release = threading.Event()
        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {}, polling_interval=0)
        task = receiver.start_autonomous_task(lambda: release.wait(timeout=2), "Inference", {"model_id": "m1"})

        for _ in range(MAX_SELF_REGISTRATION_ATTEMPTS):
            self.assertTrue(receiver._get_current_activities()[0].HasField("command_type"))
        for _ in range(3):
            report = receiver._get_current_activities()[0]
            self.assertFalse(report.HasField("command_type"))
            self.assertFalse(report.HasField("parameters"))

        release.set()
        self.assertTrue(_wait_until(lambda: task.done))

    def test_non_serialisable_response_fails_task_instead_of_polling(self):
        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {}, polling_interval=0)
        task = receiver.start_autonomous_task(lambda: {1, 2}, "Inference", None)

        self.assertTrue(_wait_until(lambda: task.done))
        self.assertEqual(task.status, StatusType.FAILED)
        json.loads(receiver._get_current_activities()[0].response)

    def test_check_abort_raises_after_simulated_interrupted_directive(self):
        proceed = threading.Event()
        aborted = threading.Event()

        def callback():
            proceed.wait(timeout=2)
            try:
                receiver.check_abort()
            except StoppedException:
                aborted.set()
                raise

        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {}, polling_interval=0)
        task = receiver.start_autonomous_task(callback, "Inference", None)
        self.assertTrue(_wait_until(lambda: task.runner_thread is not None))

        directive = scaleout_msg.CombinerDirective()
        directive_task = directive.tasks.add()
        directive_task.correlation_id = task.correlation_id
        directive_task.status = StatusType.INTERRUPTED.value

        stub_handler = _StubGrpcHandler(directive, receiver._task_manager_stop_event)
        receiver.client = _StubClient(grpc_handler=stub_handler)
        receiver._run_task_polling()

        self.assertTrue(task.acked_by_server)
        self.assertTrue(task.interrupted)

        proceed.set()
        self.assertTrue(_wait_until(lambda: aborted.is_set()))
        self.assertTrue(_wait_until(lambda: task.done))
        self.assertEqual(task.status, StatusType.INTERRUPTED)


class TestRunTaskTimestamps(unittest.TestCase):
    def test_run_task_sets_started_and_completed_at(self):
        request = scaleout_msg.TaskRequest(correlation_id="corr-4")
        task = Task(request)
        receiver = TaskReceiver(client=_StubClient(), task_callback=lambda req: {"result": "ok"}, polling_interval=0)
        receiver._current_tasks.append(task)

        before = datetime.now(timezone.utc)
        receiver._run_task(task)
        after = datetime.now(timezone.utc)

        self.assertEqual(task.status, StatusType.COMPLETED)
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.completed_at)
        self.assertTrue(before <= task.started_at <= after)
        self.assertTrue(task.started_at <= task.completed_at <= after)


if __name__ == "__main__":
    unittest.main()
