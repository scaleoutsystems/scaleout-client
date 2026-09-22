"""Unit tests for the protocol-agnostic queue core.

Uses an in-memory fake TransportAdapter. No gRPC imports.
"""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace
from typing import Optional

from scaleoututil.queue.backends.memory import InMemoryBackend
from scaleoututil.queue.priority_queue import PriorityQueue
from scaleoututil.queue.record import (
    DEFAULT_CLASS_NAME,
    Acked,
    ClassOrder,
    DecayPolicy,
    PermanentFailure,
    PriorityClass,
    QueueRecord,
    RecordState,
    SendResult,
    Sent,
    TransientFailure,
)


class _FakeAdapter:
    """Captures every send call. Configurable per record behaviour."""

    def __init__(self) -> None:
        self.sent: list[QueueRecord] = []
        self._lock = threading.Lock()
        self._scripted: dict[str, list[SendResult]] = {}

    def script(self, record_id: str, results: list[SendResult]) -> None:
        self._scripted[record_id] = list(results)

    def send(self, record: QueueRecord) -> SendResult:
        with self._lock:
            self.sent.append(record)
            scripted = self._scripted.get(record.record_id)
            if scripted:
                return scripted.pop(0)
            return Acked()


class _FakeClock:
    """Controllable wall-clock stand-in. Backend reads `clock()` per check."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _record(
    priority: str,
    payload: bytes = b"x",
    message_type: str = "Test",
    enqueued_at: Optional[float] = None,
) -> QueueRecord:
    rec = QueueRecord.create(message_type=message_type, payload=payload, priority=priority)
    if enqueued_at is not None:
        rec = replace(rec, enqueued_at=enqueued_at)
    return rec


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# Convenience class set used by most ordering tests
def _basic_classes() -> list[PriorityClass]:
    return [
        PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO),
        PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO),
        PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO),
    ]


class TestInMemoryBackendOrdering(unittest.TestCase):
    def test_strict_priority_across_classes(self):
        backend = InMemoryBackend(classes=_basic_classes())
        backend.append(_record("P5"))
        backend.append(_record("P5"))
        backend.append(_record("P1"))  # highest priority added last
        first = backend.next_record()
        self.assertIsNotNone(first)
        self.assertEqual(first.priority, "P1")

    def test_fifo_within_class(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)])
        backend.append(_record("P4", payload=b"a"))
        backend.append(_record("P4", payload=b"b"))
        backend.append(_record("P4", payload=b"c"))
        self.assertEqual(backend.next_record().payload, b"a")
        self.assertEqual(backend.next_record().payload, b"b")
        self.assertEqual(backend.next_record().payload, b"c")
        self.assertIsNone(backend.next_record())

    def test_lifo_within_class(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO)])
        backend.append(_record("P5", payload=b"a"))
        backend.append(_record("P5", payload=b"b"))
        backend.append(_record("P5", payload=b"c"))
        self.assertEqual(backend.next_record().payload, b"c")
        self.assertEqual(backend.next_record().payload, b"b")
        self.assertEqual(backend.next_record().payload, b"a")

    def test_drop_oldest_when_class_capped(self):
        backend = InMemoryBackend(
            classes=[PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO, capacity=2)]
        )
        backend.append(_record("P5", payload=b"a"))
        backend.append(_record("P5", payload=b"b"))
        backend.append(_record("P5", payload=b"c"))  # should evict 'a'

        stats = backend.stats()
        self.assertEqual(stats["P5"].depth, 2)
        self.assertEqual(stats["P5"].dropped, 1)

        # LIFO: pulls newest first, then the survivor 'b'
        self.assertEqual(backend.next_record().payload, b"c")
        self.assertEqual(backend.next_record().payload, b"b")
        self.assertIsNone(backend.next_record())

    def test_state_transitions_in_flight_and_sent(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)])
        backend.append(_record("P4"))
        picked = backend.next_record()
        self.assertEqual(picked.state, RecordState.IN_FLIGHT)
        backend.mark_sent(picked.record_id)
        self.assertIsNone(backend.next_record())

    def test_mark_failed_requeues_for_retry(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)])
        backend.append(_record("P4"))
        picked = backend.next_record()
        backend.mark_failed(picked.record_id, retry=True)

        again = backend.next_record()
        self.assertIsNotNone(again)
        self.assertEqual(again.record_id, picked.record_id)
        self.assertEqual(again.attempts, 1)

    def test_mark_failed_no_retry_drops(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)])
        backend.append(_record("P4"))
        picked = backend.next_record()
        backend.mark_failed(picked.record_id, retry=False)
        self.assertIsNone(backend.next_record())

    def test_requeue_inflight_recovers_on_startup(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)])
        backend.append(_record("P4", payload=b"a"))
        picked = backend.next_record()
        self.assertEqual(picked.state, RecordState.IN_FLIGHT)
        # simulate a restart: requeue_inflight should move IN_FLIGHT back to QUEUED
        backend.requeue_inflight()
        recovered = backend.next_record()
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.record_id, picked.record_id)

    def test_duplicate_class_names_rejected(self):
        with self.assertRaises(ValueError):
            InMemoryBackend(
                classes=[
                    PriorityClass(name="X", priority=10, order=ClassOrder.FIFO),
                    PriorityClass(name="X", priority=20, order=ClassOrder.FIFO),
                ]
            )


class TestDefaultClass(unittest.TestCase):
    def test_unknown_priority_routes_to_default(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)])
        backend.append(_record("does-not-exist", payload=b"orphan"))
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.payload, b"orphan")
        self.assertEqual(record.priority, DEFAULT_CLASS_NAME)

    def test_default_loses_to_real_class(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)])
        backend.append(_record("does-not-exist", payload=b"orphan"))
        backend.append(_record("P1", payload=b"real"))
        # P1 (priority=100) beats default (priority=0)
        self.assertEqual(backend.next_record().payload, b"real")
        self.assertEqual(backend.next_record().payload, b"orphan")

    def test_default_settings_overridable(self):
        custom_default = PriorityClass(
            name=DEFAULT_CLASS_NAME,
            priority=5,
            order=ClassOrder.LIFO,
            capacity=2,
        )
        backend = InMemoryBackend(classes=[], default=custom_default)
        backend.append(_record("?", payload=b"a"))
        backend.append(_record("?", payload=b"b"))
        backend.append(_record("?", payload=b"c"))  # evicts 'a' under capacity=2

        self.assertEqual(backend.stats()[DEFAULT_CLASS_NAME].dropped, 1)
        # LIFO drain
        self.assertEqual(backend.next_record().payload, b"c")
        self.assertEqual(backend.next_record().payload, b"b")

    def test_default_class_not_removable(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)])
        with self.assertRaises(ValueError):
            backend.remove_class(DEFAULT_CLASS_NAME)


class TestDynamicClassRegistry(unittest.TestCase):
    def test_add_class_then_enqueue_and_drain(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)])
        backend.add_class(PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO))
        backend.append(_record("P5", payload=b"new-class"))
        self.assertEqual(backend.next_record().payload, b"new-class")

    def test_add_class_duplicate_rejected(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)])
        with self.assertRaises(ValueError):
            backend.add_class(PriorityClass(name="P1", priority=50, order=ClassOrder.FIFO))

    def test_add_class_with_default_name_rejected(self):
        backend = InMemoryBackend(classes=[])
        with self.assertRaises(ValueError):
            backend.add_class(PriorityClass(name=DEFAULT_CLASS_NAME, priority=99, order=ClassOrder.FIFO))

    def test_remove_empty_class(self):
        backend = InMemoryBackend(
            classes=[
                PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO),
                PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO),
            ]
        )
        migrated = backend.remove_class("P5")
        self.assertEqual(migrated, 0)
        # subsequent enqueue with P5 falls through to default
        backend.append(_record("P5", payload=b"orphan"))
        self.assertEqual(backend.next_record().priority, DEFAULT_CLASS_NAME)

    def test_remove_non_empty_requires_force(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO)])
        backend.append(_record("P5"))
        with self.assertRaises(ValueError):
            backend.remove_class("P5", force=False)

    def test_remove_with_force_migrates_to_default(self):
        backend = InMemoryBackend(classes=[PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO)])
        backend.append(_record("P5", payload=b"a"))
        backend.append(_record("P5", payload=b"b"))
        migrated = backend.remove_class("P5", force=True)
        self.assertEqual(migrated, 2)
        # records now live under the default class
        rec = backend.next_record()
        self.assertEqual(rec.priority, DEFAULT_CLASS_NAME)
        self.assertEqual(backend.stats()[DEFAULT_CLASS_NAME].depth, 1)

    def test_remove_unknown_class_raises(self):
        backend = InMemoryBackend(classes=[])
        with self.assertRaises(KeyError):
            backend.remove_class("nope")


class TestDecayPolicy(unittest.TestCase):
    def test_drop_after_discards_aged_tail(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="P5",
                    priority=40,
                    order=ClassOrder.LIFO,
                    decay=DecayPolicy(drop_after=60.0),
                )
            ],
            clock=clock,
        )
        # three records spaced over time, oldest first
        backend.append(_record("P5", payload=b"old", enqueued_at=900.0))   # age 100 > 60 → drop
        backend.append(_record("P5", payload=b"mid", enqueued_at=980.0))   # age 20  → keep
        backend.append(_record("P5", payload=b"new", enqueued_at=995.0))   # age 5   → keep

        # LIFO: returns newest first
        self.assertEqual(backend.next_record().payload, b"new")
        self.assertEqual(backend.next_record().payload, b"mid")
        self.assertIsNone(backend.next_record())
        self.assertEqual(backend.stats()["P5"].dropped, 1)

    def test_demote_after_reassigns_to_target_class(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="P4",
                    priority=70,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(30.0, "P5")),
                ),
                PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO),
            ],
            clock=clock,
        )
        # record in P4, aged 50s (> 30s threshold)
        backend.append(_record("P4", payload=b"aged", enqueued_at=950.0))
        # fresh record in P5
        backend.append(_record("P5", payload=b"fresh", enqueued_at=999.0))

        # P4 record gets demoted to P5; then P5 (priority=40) is the winning class.
        # LIFO inside P5: fresh is newer than aged, so fresh drains first.
        first = backend.next_record()
        self.assertEqual(first.payload, b"fresh")
        second = backend.next_record()
        self.assertEqual(second.payload, b"aged")
        self.assertEqual(second.priority, "P5")  # priority field updated on reassign

    def test_demote_to_missing_class_falls_back_to_default(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="P4",
                    priority=70,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(30.0, "does-not-exist")),
                ),
            ],
            clock=clock,
        )
        backend.append(_record("P4", payload=b"aged", enqueued_at=950.0))
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.priority, DEFAULT_CLASS_NAME)

    def test_demote_cascade_chain(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="A",
                    priority=80,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(10.0, "B")),
                ),
                PriorityClass(
                    name="B",
                    priority=50,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(20.0, "C")),
                ),
                PriorityClass(name="C", priority=20, order=ClassOrder.FIFO),
            ],
            clock=clock,
        )
        # ancient record will cascade A → B → C in one next_record() call
        backend.append(_record("A", payload=b"ancient", enqueued_at=900.0))
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.payload, b"ancient")
        self.assertEqual(record.priority, "C")

    def test_demote_cycle_protected_by_cascade_cap(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="A",
                    priority=80,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(10.0, "B")),
                ),
                PriorityClass(
                    name="B",
                    priority=50,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(10.0, "A")),
                ),
            ],
            clock=clock,
        )
        backend.append(_record("A", payload=b"loop-victim", enqueued_at=900.0))
        # cascade cap kicks in: must return a record rather than spin or return None
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.payload, b"loop-victim")
        self.assertIn(record.priority, {"A", "B"})

    def test_bulk_demote_in_one_call(self):
        """Many aged records in one class must all be demoted within a single
        next_record() — not paced one-per-call. Regression for the inner-loop
        continue bug where each demote skipped to the next class."""
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[
                PriorityClass(
                    name="P4",
                    priority=70,
                    order=ClassOrder.FIFO,
                    decay=DecayPolicy(demote_after=(10.0, "P5")),
                ),
                PriorityClass(name="P5", priority=40, order=ClassOrder.FIFO),
            ],
            clock=clock,
        )
        # 20 aged records in P4 — far more than the cascade depth cap (5)
        for i in range(20):
            backend.append(_record("P4", payload=f"old{i}".encode(), enqueued_at=900.0 + i))
        # one fresh record in P4 that should NOT be demoted
        backend.append(_record("P4", payload=b"fresh", enqueued_at=999.0))

        # First call to next_record must demote all 20 stale heads to P5 (so the
        # fresh P4 record becomes the head) and then return the highest-priority
        # candidate — which is the fresh P4 record, at P4's priority (70).
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.payload, b"fresh")
        self.assertEqual(record.priority, "P4")
        # all 20 aged records now live in P5, none still in P4
        self.assertEqual(backend.stats()["P4"].depth, 0)
        self.assertEqual(backend.stats()["P5"].depth, 20)

    def test_no_decay_means_no_demote_no_drop(self):
        clock = _FakeClock(t=1000.0)
        backend = InMemoryBackend(
            classes=[PriorityClass(name="P4", priority=70, order=ClassOrder.FIFO)],
            clock=clock,
        )
        backend.append(_record("P4", payload=b"old", enqueued_at=1.0))  # very old
        # no decay → still returned in its original class
        record = backend.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(record.priority, "P4")


class TestPriorityQueueDrainer(unittest.TestCase):
    def _make(self, *, classes=None) -> tuple[PriorityQueue, _FakeAdapter, InMemoryBackend]:
        backend = InMemoryBackend(classes=classes or _basic_classes())
        adapter = _FakeAdapter()
        queue = PriorityQueue(transport=adapter, backend=backend)
        return queue, adapter, backend

    def test_enqueue_and_drain_simple(self):
        queue, adapter, _ = self._make()
        queue.start()
        try:
            r = _record("P4", payload=b"hello")
            queue.enqueue(r)
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 1))
            self.assertEqual(adapter.sent[0].payload, b"hello")
        finally:
            queue.stop(timeout=2.0)

    def test_strict_priority_when_drained(self):
        queue, adapter, _ = self._make()
        queue.enqueue(_record("P5", payload=b"low-a"))
        queue.enqueue(_record("P5", payload=b"low-b"))
        queue.enqueue(_record("P1", payload=b"high"))

        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 3))
            # P1 always drains before any P5, regardless of arrival order
            self.assertEqual(adapter.sent[0].payload, b"high")
        finally:
            queue.stop(timeout=2.0)

    def test_transient_failure_retries_same_record(self):
        queue, adapter, _ = self._make()
        r = _record("P4", payload=b"flaky")
        adapter.script(r.record_id, [TransientFailure("net"), TransientFailure("net"), Acked()])
        queue.enqueue(r)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) >= 3, timeout=5.0))
            self.assertEqual({s.record_id for s in adapter.sent[:3]}, {r.record_id})
            self.assertEqual(adapter.sent[1].attempts, 1)
            self.assertEqual(adapter.sent[2].attempts, 2)
        finally:
            queue.stop(timeout=2.0)

    def test_permanent_failure_drops_record(self):
        queue, adapter, backend = self._make()
        r = _record("P4", payload=b"bad")
        adapter.script(r.record_id, [PermanentFailure("schema mismatch")])
        queue.enqueue(r)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 1))
            self.assertTrue(_wait_for(lambda: backend.stats()["P4"].depth == 0 and backend.stats()["P4"].in_flight == 0))
        finally:
            queue.stop(timeout=2.0)

    def test_drop_oldest_under_load(self):
        queue, adapter, backend = self._make(
            classes=[PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO, capacity=5)]
        )
        for i in range(50):
            queue.enqueue(_record("P5", payload=f"t{i}".encode()))
        self.assertEqual(backend.stats()["P5"].depth, 5)
        self.assertEqual(backend.stats()["P5"].dropped, 45)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 5))
            payloads = [r.payload for r in adapter.sent]
            self.assertEqual(payloads[0], b"t49")
            self.assertEqual(payloads[-1], b"t45")
        finally:
            queue.stop(timeout=2.0)

    def test_requeue_inflight_on_restart(self):
        queue, adapter, backend = self._make()
        r = _record("P4", payload=b"recovered")
        backend.append(r)
        backend.next_record()  # moves to IN_FLIGHT
        self.assertEqual(backend.stats()["P4"].in_flight, 1)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 1))
            self.assertEqual(adapter.sent[0].payload, b"recovered")
        finally:
            queue.stop(timeout=2.0)

    def test_stop_interrupts_idle_drainer(self):
        queue, _, _ = self._make()
        queue.start()
        t0 = time.monotonic()
        queue.stop(timeout=2.0)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_pass_through_add_remove_class(self):
        queue, adapter, backend = self._make(
            classes=[PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO)]
        )
        queue.add_class(PriorityClass(name="P5", priority=40, order=ClassOrder.LIFO))
        self.assertIn("P5", backend.stats())

        queue.enqueue(_record("P5", payload=b"x"))
        migrated = queue.remove_class("P5", force=True)
        self.assertEqual(migrated, 1)
        self.assertNotIn("P5", backend.stats())
        # record landed in default
        self.assertEqual(backend.stats()[DEFAULT_CLASS_NAME].depth, 1)


class TestStreamingAckStateMachine(unittest.TestCase):
    """Backend state machine for streaming ack semantics (mark_sent / mark_acked / sweep_unacked)."""

    def _backend(self, *, unacked_ttl_s: float = 30.0, clock: Optional[_FakeClock] = None) -> InMemoryBackend:
        clock = clock or _FakeClock()
        return InMemoryBackend(classes=_basic_classes(), clock=clock, unacked_ttl_s=unacked_ttl_s)

    def test_mark_sent_keeps_record_inflight_with_sent_state(self):
        backend = self._backend()
        r = _record("P5", payload=b"t")
        backend.append(r)
        popped = backend.next_record()
        self.assertEqual(popped.record_id, r.record_id)
        backend.mark_sent(r.record_id)
        # still tracked in-flight, but state == SENT
        self.assertEqual(backend.stats()["P5"].in_flight, 1)
        self.assertEqual(backend._in_flight[r.record_id].state, RecordState.SENT)
        self.assertIn(r.record_id, backend._sent_at)

    def test_mark_acked_after_mark_sent_clears_both_tables(self):
        backend = self._backend()
        r = _record("P5", payload=b"t")
        backend.append(r)
        backend.next_record()
        backend.mark_sent(r.record_id)
        backend.mark_acked(r.record_id)
        self.assertEqual(backend.stats()["P5"].in_flight, 0)
        self.assertNotIn(r.record_id, backend._in_flight)
        self.assertNotIn(r.record_id, backend._sent_at)

    def test_requeue_inflight_covers_inflight_and_sent(self):
        backend = self._backend()
        a = _record("P5", payload=b"a")
        b = _record("P5", payload=b"b")
        backend.append(a)
        backend.append(b)
        # a stays IN_FLIGHT, b moves on to SENT
        backend.next_record()
        backend.next_record()
        backend.mark_sent(b.record_id)
        backend.requeue_inflight()
        # both are back in the class queue, _sent_at cleared
        self.assertEqual(backend.stats()["P5"].in_flight, 0)
        self.assertEqual(backend.stats()["P5"].depth, 2)
        self.assertEqual(backend._sent_at, {})

    def test_sweep_unacked_requeues_only_aged_sent_records(self):
        clock = _FakeClock()
        backend = self._backend(unacked_ttl_s=10.0, clock=clock)
        a = _record("P5", payload=b"a")
        b = _record("P5", payload=b"b")
        backend.append(a)
        backend.append(b)
        backend.next_record()  # popped a (LIFO → newest first is b)
        backend.next_record()  # popped the other
        # Only the second pop goes through mark_sent: it's the "on wire" one.
        # Identify which one is still IN_FLIGHT vs SENT by stamping a only.
        backend.mark_sent(a.record_id)  # a in SENT
        # b still IN_FLIGHT (no sent stamp)
        # advance past TTL
        clock.advance(11.0)
        requeued = backend.sweep_unacked()
        self.assertEqual(requeued, [a.record_id])
        # a went back to QUEUED with attempts += 1; b stays in flight
        self.assertEqual(backend.stats()["P5"].in_flight, 1)
        self.assertEqual(backend.stats()["P5"].depth, 1)
        self.assertNotIn(a.record_id, backend._sent_at)

    def test_sweep_unacked_no_op_when_under_ttl(self):
        clock = _FakeClock()
        backend = self._backend(unacked_ttl_s=10.0, clock=clock)
        r = _record("P5", payload=b"t")
        backend.append(r)
        backend.next_record()
        backend.mark_sent(r.record_id)
        clock.advance(5.0)
        self.assertEqual(backend.sweep_unacked(), [])
        self.assertEqual(backend.stats()["P5"].in_flight, 1)


class TestDrainerSentSemantics(unittest.TestCase):
    """Drainer distinguishes Acked (unary, terminal) from Sent (streaming, deferred ack)."""

    def _make(self) -> tuple[PriorityQueue, _FakeAdapter, InMemoryBackend]:
        backend = InMemoryBackend(classes=_basic_classes())
        adapter = _FakeAdapter()
        queue = PriorityQueue(transport=adapter, backend=backend)
        return queue, adapter, backend

    def test_acked_result_marks_record_acked(self):
        queue, adapter, backend = self._make()
        r = _record("P4", payload=b"u")
        queue.enqueue(r)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 1))
            self.assertTrue(_wait_for(lambda: backend.stats()["P4"].in_flight == 0))
            self.assertNotIn(r.record_id, backend._in_flight)
        finally:
            queue.stop(timeout=2.0)

    def test_sent_keeps_record_inflight_as_sent(self):
        queue, adapter, backend = self._make()
        r = _record("P5", payload=b"s")
        adapter.script(r.record_id, [Sent()])
        queue.enqueue(r)
        queue.start()
        try:
            self.assertTrue(_wait_for(lambda: len(adapter.sent) == 1))
            # record remains until adapter calls mark_acked
            self.assertTrue(_wait_for(lambda: backend.stats()["P5"].in_flight == 1))
            self.assertEqual(backend._in_flight[r.record_id].state, RecordState.SENT)
            # external ack arrives
            backend.mark_acked(r.record_id)
            self.assertEqual(backend.stats()["P5"].in_flight, 0)
        finally:
            queue.stop(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
