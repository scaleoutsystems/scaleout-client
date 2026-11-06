import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.grpc.statustype import StatusType
from scaleoututil.grpc.tasktype import TaskType
from scaleoututil.logging import ScaleoutLogger
import traceback

if TYPE_CHECKING:
    from scaleout.client.grpc_edge_client_runtime import GrpcEdgeClientRuntime  # not-floating-import

# Polls to keep sending command_type for an unacked autonomous task before assuming the combiner lacks support.
MAX_SELF_REGISTRATION_ATTEMPTS = 10


class StoppedException(Exception):
    pass


class UnknownTaskType(Exception):
    pass


class Task:
    def __init__(self, request: scaleout_msg.TaskRequest):
        self.request = request
        self.runner_thread = None
        self.lock = threading.Lock()
        self.status = StatusType.PENDING
        self.interrupted = False
        self.interrupted_reason = None
        self.response = None
        self.correlation_id = request.correlation_id
        self.done = False
        self.command_type = None
        self.parameters_json = None
        self.acked_by_server = False
        self.registration_attempts = 0
        self.started_at = None
        self.completed_at = None


class TaskReceiver:
    def __init__(self, client: "GrpcEdgeClientRuntime", task_callback: callable, polling_interval: int = 5):
        self.client = client
        self.task_callback = task_callback

        self.polling_interval = polling_interval

        self._current_tasks: List[Task] = []

        self._task_manager_thread = None
        self._task_manager_stop_event = threading.Event()
        self._skip_polling_tasks = False

        # Protects access to current_task and task manager thread
        self._lock = threading.RLock()

    def start(self):
        self._skip_polling_tasks = False
        if self._task_manager_thread is not None:
            if self._task_manager_thread.is_alive():
                ScaleoutLogger().error("TaskReceiver: Task polling thread is already running.")
                raise RuntimeError("Task polling thread is already running.")
            if not self._task_manager_stop_event.is_set():
                ScaleoutLogger().error("TaskReceiver: Task polling thread is already running.")
                raise RuntimeError("Task polling thread is already running.")
        self._task_manager_thread = threading.Thread(
            target=self._run_task_polling,
            name="TaskReceiver",
            daemon=True,
        )
        self._task_manager_stop_event.clear()
        self._task_manager_thread.start()

    def stop(self):
        """Nonblocking stop of the task polling thread."""
        self._task_manager_stop_event.set()

    def get_current_task(self) -> Task:
        """Get the current task for the current thread.

        This function should be called from the task callback to get the current task.
        If called from another thread, this function returns None.
        """
        with self._lock:
            # We lock to ensure that the current task is not finished while we check it
            for task in self._current_tasks:
                with task.lock:
                    if task.runner_thread == threading.current_thread():
                        return task
        return None

    def check_abort(self):
        """Check if the current task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from another thread, this function is a no-op.
        """
        task = self.get_current_task()
        if task is None:
            return
        with task.lock:
            if task.interrupted:
                raise StoppedException(task.interrupted_reason)

    def abort_all_current_tasks(self):
        """Abort all current tasks."""
        with self._lock:
            # We lock to ensure that the current task is not finished while we check it
            for task in self._current_tasks:
                with task.lock:
                    # We lock to ensure that the current task does not receive updates while we set the interrupted flag
                    if not task.interrupted:
                        task.interrupted = True
                        task.interrupted_reason = "Aborted by client"
                ScaleoutLogger().info("TaskReceiver: Aborting current task... ")

    def start_autonomous_task(self, callback: Callable[[], Any], command_type: str, parameters: Optional[dict]) -> Task:
        """Start a task the client is initiating on its own accord.

        Runs `callback` on a new thread and starts reporting on it through the
        existing poll/report cycle immediately, so the combiner can lazily create
        a stoppable `commands` row for it - no synchronous registration
        round-trip needed before the task can start.
        """
        correlation_id = str(uuid.uuid4())
        parameters_json = json.dumps(parameters) if parameters is not None else None
        request = scaleout_msg.TaskRequest()
        request.correlation_id = correlation_id
        request.type = command_type
        if parameters_json is not None:
            request.data = parameters_json

        task = Task(request)
        task.command_type = command_type
        task.parameters_json = parameters_json
        with self._lock:
            self._current_tasks.append(task)
        threading.Thread(target=self._execute_task, args=(task, callback)).start()
        return task

    def _run_task_polling(self):
        # This method runs in the task manager thread
        while True:
            try:
                tic = time.time()
                if self._task_manager_stop_event.is_set():
                    ScaleoutLogger().info("TaskReceiver: Stopping task polling thread.")
                    break

                activities = self._get_current_activities()
                report = scaleout_msg.ClientReport()
                report.client_id = self.client.client_id
                report.poll_interval_ms = int(self.polling_interval * 1000)
                report.reports.extend(activities)
                report.skip_polling_tasks = self._skip_polling_tasks
                if len(activities) == 0:
                    ScaleoutLogger().debug("TaskReceiver: Nothing to report, Polling for task")
                else:
                    ScaleoutLogger().debug("TaskReceiver: Reporting %s tasks", len(activities))

                directive: scaleout_msg.CombinerDirective = self.client.grpc_handler.PollAndReportAsync(report)

                with self._lock:
                    # Lock when removing current task
                    for activity in activities:
                        for task in self._current_tasks:
                            if task.correlation_id == activity.correlation_id:
                                if activity.done:
                                    self._current_tasks.remove(task)
                                break

                new_task_requests = []
                new_restart_task_requests = []
                for task_request in directive.tasks:
                    task = self._get_task_by_correlation_id(task_request.correlation_id)
                    if task is not None:
                        # Update to existing task
                        with task.lock:
                            task.acked_by_server = True
                            if StatusType.matches(task_request.status, StatusType.INTERRUPTED):
                                if not task.interrupted:
                                    task.interrupted = True
                                    task.interrupted_reason = "Aborted by server"
                                ScaleoutLogger().info("TaskReceiver: Received interrupt message for task %s.", task.correlation_id)
                            elif StatusType.matches(task_request.status, StatusType.TIMEOUT):
                                if not task.interrupted:
                                    task.interrupted = True
                                    task.interrupted_reason = "Timed out by server"
                                ScaleoutLogger().info("TaskReceiver: Received timeout message for task %s.", task.correlation_id)
                    elif task_request.type == TaskType.RestartClient.value:
                        new_restart_task_requests.append(task_request)
                    else:
                        new_task_requests.append(task_request)

                # Dispatch restart requests last: this guarantees that any other task
                # delivered in this same batch is already in _current_tasks by the time a
                # restart's own thread checks has_other_tasks() / has_current_tasks() --
                # closing an ordering race where a restart arriving before another task in
                # the same poll response could otherwise conclude "no other tasks running".
                for task_request in new_task_requests + new_restart_task_requests:
                    with self._lock:
                        # Lock to add new task
                        ScaleoutLogger().info("TaskReceiver: Got task %s", task_request.correlation_id)
                        new_task = Task(task_request)
                        new_task.acked_by_server = True
                        self._current_tasks.append(new_task)
                        # Run the task in a separate thread
                        threading.Thread(target=self._run_task, args=(new_task,)).start()

                toc = time.time()
                if toc - tic < self.polling_interval:
                    time.sleep(self.polling_interval - (toc - tic))
            except Exception as e:
                # Unexpected error -- log and stop polling
                ScaleoutLogger().error("TaskReceiver: Error in task polling: %s", e)
                ScaleoutLogger().error(traceback.format_exc())
                self._task_manager_stop_event.set()
                break
        self._task_manager_stop_event.set()

    def _get_current_activities(self) -> List[scaleout_msg.ActivityReport]:
        activities = []
        with self._lock:
            for task in self._current_tasks:
                with task.lock:
                    report = scaleout_msg.ActivityReport()
                    report.node_id = self.client.client_id
                    report.status = task.status.value
                    if task.response:
                        report.response = json.dumps(task.response)
                    report.correlation_id = task.correlation_id
                    report.done = task.done
                    if task.started_at is not None:
                        report.started_at.FromDatetime(task.started_at)
                    if task.completed_at is not None:
                        report.completed_at.FromDatetime(task.completed_at)
                    if not task.acked_by_server and task.command_type is not None:
                        if task.registration_attempts < MAX_SELF_REGISTRATION_ATTEMPTS:
                            task.registration_attempts += 1
                            report.command_type = task.command_type
                            if task.parameters_json is not None:
                                report.parameters = task.parameters_json
                        elif task.registration_attempts == MAX_SELF_REGISTRATION_ATTEMPTS:
                            # Older combiners don't support self-registered tasks: keep running locally, stop resending.
                            task.registration_attempts += 1
                            ScaleoutLogger().warning(
                                "TaskReceiver: Combiner did not acknowledge autonomous task %s; it will run untracked and cannot be stopped by the server.",
                                task.correlation_id,
                            )
                    activities.append(report)
        return activities

    def _get_task_by_correlation_id(self, correlation_id: str) -> Optional[Task]:
        with self._lock:
            for task in self._current_tasks:
                if task.correlation_id == correlation_id:
                    return task
        return None

    def _run_task(self, task: Task):
        self._execute_task(task, lambda: self.task_callback(task.request))

    def _execute_task(self, task: Task, runner: Callable[[], Any]):
        # This method runs in the task runner thread (Not the task manager thread)
        # It only affects the current task and hence does not need to lock the task receiver
        with task.lock:
            task.runner_thread = threading.current_thread()
            task.status = StatusType.RUNNING
            task.started_at = datetime.now(timezone.utc)
        try:
            response = runner()
            # Try to serialize
            # Fail here rather than in the polling thread
            json.dumps(response)
            ScaleoutLogger().info("TaskReceiver: Task completed: %s", task.correlation_id)
            with task.lock:
                task.response = response
                task.status = StatusType.COMPLETED
        except StoppedException as e:
            with task.lock:
                ScaleoutLogger().info("TaskReceiver: Task interrupted: %s", e)
                task.status = StatusType.INTERRUPTED
                task.response = {"msg": str(e)}
        except UnknownTaskType as e:
            with task.lock:
                ScaleoutLogger().error("TaskReceiver: Task failed: %s", e)
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        except Exception as e:
            with task.lock:
                ScaleoutLogger().error("TaskReceiver: Task failed: %s", e)
                ScaleoutLogger().error(traceback.format_exc())
                task.status = StatusType.FAILED
                task.response = {"error": str(e)}
        finally:
            with task.lock:
                task.done = True
                task.completed_at = datetime.now(timezone.utc)

    def wait_on_manager_thread(self):
        if self._task_manager_thread is not None:
            # Use wait with timeout to catch control-C properly
            while self._task_manager_stop_event.wait(1) is False:
                pass
            self._task_manager_thread.join()

    def has_current_tasks(self):
        with self._lock:
            return len(self._current_tasks) > 0

    def has_other_tasks(self):
        with self._lock:
            this_task = self.get_current_task()
            return len([t for t in self._current_tasks if t != this_task]) > 0

    def stop_recieving_new_tasks(self):
        """Used to let the combiner know that the client only reports already started tasks before shutting down."""
        self._skip_polling_tasks = True

    def resume_recieving_new_tasks(self):
        """Undo stop_recieving_new_tasks, e.g. when a restart fails before completing."""
        self._skip_polling_tasks = False

    def is_task_running(self, task):
        with self._lock:
            return task in self._current_tasks
