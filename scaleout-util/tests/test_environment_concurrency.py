"""Concurrency regression tests for PythonEnv.create_virtualenv.

Creating a managed environment is not atomic: the existence check and the
virtualenv + pip work are separated by minutes. Several clients sharing a
project directory (e.g. the siblings spawned by
``scaleoututil.launchers.launch_clients``) resolve to the same env directory,
so creation must be serialized -- otherwise they build concurrently and the
loser's cleanup deletes the winner's environment mid-install.
"""

import shutil
import threading
from pathlib import Path

from scaleoututil.utils.environment import PythonEnv


def _make_env(tmp_path: Path) -> PythonEnv:
    env = PythonEnv(name=".testenv", build_dependencies=[], dependencies=[])
    env.set_base_path(tmp_path / "venvs")
    return env


def test_existing_env_short_circuits(tmp_path, monkeypatch):
    """An environment that already exists is reused, never rebuilt."""
    env = _make_env(tmp_path)
    env.path.mkdir(parents=True)

    def fail(*args, **kwargs):
        raise AssertionError("should not rebuild an existing environment")

    monkeypatch.setattr(PythonEnv, "_create_virtualenv_locked", fail)

    assert env.create_virtualenv() is True


def test_lock_file_lives_outside_env_dir(tmp_path, monkeypatch):
    """The lock is held during the build and sits outside the env dir.

    ``remove_on_error`` rmtree's the environment directory on failure. A lock
    kept inside it would be destroyed by exactly the failure it has to guard,
    so the lock file must live beside the directory, not within it.
    """
    env = _make_env(tmp_path)
    lock_path = env.path.parent / f"{env.path.name}.lock"
    seen = {}

    def build(self, env_dir, capture_output=False):
        # The lock is held for the duration of the build, so its file exists
        # here (filelock unlinks it again on release).
        seen["held_during_build"] = lock_path.exists()
        seen["outside_env_dir"] = env_dir not in lock_path.parents
        env_dir.mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(PythonEnv, "_create_virtualenv_locked", build)
    assert env.create_virtualenv() is True

    assert seen["held_during_build"] is True
    assert seen["outside_env_dir"] is True

    # The env dir can be removed without taking the lock with it.
    shutil.rmtree(env.path)
    assert not env.path.exists()


def test_concurrent_creation_builds_exactly_once(tmp_path, monkeypatch):
    """Racing callers serialize: one builds, the rest observe the finished env."""
    env_base = tmp_path / "venvs"
    calls = []
    calls_lock = threading.Lock()

    def build(self, env_dir, capture_output=False):
        with calls_lock:
            calls.append(env_dir)
        # Hold the lock long enough that the other threads are definitely
        # queued behind us before the directory appears.
        threading.Event().wait(0.2)
        env_dir.mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(PythonEnv, "_create_virtualenv_locked", build)

    results = {}
    errors = {}

    def worker(i):
        try:
            env = PythonEnv(name=".testenv", build_dependencies=[], dependencies=[])
            env.set_base_path(env_base)
            results[i] = env.create_virtualenv()
        except Exception as e:  # noqa: BLE001 - surfaced via the assertion below
            errors[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == {}
    assert results == {i: True for i in range(4)}
    assert len(calls) == 1, f"environment built {len(calls)} times, expected exactly 1"
