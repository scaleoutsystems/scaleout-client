"""In-memory PersistentBackend. v1 default; lost on process exit.

Holds a dict of priority classes (configurable, dynamic, plus an always-present
default). Scheduling: walk classes, peek heads, apply tail-drop / head-demote
from each class's DecayPolicy, pick the head with the highest class.priority.

Per-class storage is a deque kept sorted ascending by (enqueued_at, record_id).
Oldest record is always at index 0; newest at index -1. FIFO classes pop the
front; LIFO classes pop the back; tail-drop always discards from the front.
Sorted storage means retried and demoted records re-enter at their *true* age
position — no leapfrog, no LIFO drift. The fast path is O(1) (append-if-newest);
out-of-order inserts use bisect.insort_right which is O(log n) search +
O(min(i, len-i)) shift via deque.insert.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Callable, Optional

from scaleoututil.queue.record import (
    DEFAULT_CLASS_NAME,
    ClassOrder,
    PriorityClass,
    QueueRecord,
    QueueStats,
    RecordState,
)


# Cap how many demotion reassignments a single next_record() call may do before
# it gives up and returns whatever record it has. Bounds work per call and
# guarantees forward progress on pathological cycles (A → B → A).
_MAX_CASCADE_DEPTH = 5


class InMemoryBackend:
    """Threadsafe in-memory backend.

    Construct with a list of PriorityClass entries. Optionally override the
    default class settings; otherwise a low-priority unbounded FIFO is injected.

    Example:
        InMemoryBackend(
            classes=[
                PriorityClass(name="P1", priority=100, order=ClassOrder.FIFO),
                PriorityClass(name="P4", priority=70,  order=ClassOrder.FIFO),
                PriorityClass(name="P5", priority=40,  order=ClassOrder.LIFO,
                              capacity=10_000,
                              decay=DecayPolicy(drop_after=300)),
            ]
        )
    """

    def __init__(
        self,
        classes: list[PriorityClass],
        default: Optional[PriorityClass] = None,
        *,
        clock: Callable[[], float] = time.time,
        unacked_ttl_s: float = 30.0,
    ) -> None:
        if default is None:
            default = PriorityClass(name=DEFAULT_CLASS_NAME, priority=0, order=ClassOrder.FIFO)
        all_classes = [*classes, default]
        names = [c.name for c in all_classes]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate priority class names: {names}")

        self._lock = threading.Lock()
        self._clock = clock
        self._default_name = default.name
        self._classes: dict[str, PriorityClass] = {c.name: c for c in all_classes}
        self._queues: dict[str, deque[QueueRecord]] = {c.name: deque() for c in all_classes}
        self._in_flight: dict[str, QueueRecord] = {}
        self._sent_at: dict[str, float] = {}
        self._unacked_ttl_s = unacked_ttl_s
        self._dropped: dict[str, int] = dict.fromkeys(self._classes, 0)

    # ---- producer-facing -----------------------------------------------------

    def append(self, record: QueueRecord) -> None:
        with self._lock:
            cls = self._resolve_class(record.priority)
            if cls.name != record.priority:
                record = replace(record, priority=cls.name)
            self._enqueue_to_class(record, cls)

    # ---- drainer-facing ------------------------------------------------------

    def next_record(self) -> Optional[QueueRecord]:
        with self._lock:
            # Phase 1: stabilise the queues. Each outer pass walks every class; within
            # a class we drain all aged heads (drop_after discards, demote_after
            # reassigns to the target class) before moving on. The outer cascade only
            # exists to bound true cross-class cycles (A→B→A); within-class bulk-aging
            # is handled by the inner while in one pass.
            for cascade in range(_MAX_CASCADE_DEPTH + 1):
                did_demote = False
                for cls in list(self._classes.values()):
                    while True:
                        head = self._drop_expired_and_peek(cls)
                        if head is None:
                            break
                        if cascade < _MAX_CASCADE_DEPTH and cls.decay is not None and cls.decay.demote_after is not None:
                            age_threshold, target_name = cls.decay.demote_after
                            if self._age(head) > age_threshold:
                                target = self._classes.get(target_name) or self._classes[self._default_name]
                                if target.name != cls.name:
                                    self._reassign_head(cls, target)
                                    did_demote = True
                                    continue  # re-peek the same class's new head
                        break  # head is fresh, or class has no demote policy, or at cap
                if not did_demote:
                    break  # decay stable; proceed to selection
            # Phase 2: select the highest-priority class with a non-empty queue.
            best: Optional[tuple[int, PriorityClass]] = None
            for cls in self._classes.values():
                if not self._queues[cls.name]:
                    continue
                if best is None or cls.priority > best[0]:
                    best = (cls.priority, cls)
            if best is None:
                return None
            return self._pop_and_mark_inflight(best[1])

    def mark_sent(self, record_id: str) -> None:
        # Streaming-transport hook: the record is on the wire but not yet acked.
        # Keep it in _in_flight (so requeue_inflight / sweep_unacked can still find
        # it) and stamp wall-time for the TTL sweeper. Terminal removal happens in
        # mark_acked.
        with self._lock:
            record = self._in_flight.get(record_id)
            if record is None:
                return
            self._in_flight[record_id] = replace(record, state=RecordState.SENT)
            self._sent_at[record_id] = self._clock()

    def mark_acked(self, record_id: str) -> None:
        with self._lock:
            self._in_flight.pop(record_id, None)
            self._sent_at.pop(record_id, None)

    def mark_failed(self, record_id: str, retry: bool) -> None:
        with self._lock:
            record = self._in_flight.pop(record_id, None)
            self._sent_at.pop(record_id, None)
            if record is None or not retry:
                return
            cls = self._resolve_class(record.priority)
            if cls.name != record.priority:
                record = replace(record, priority=cls.name)
            requeued = replace(
                record,
                state=RecordState.QUEUED,
                attempts=record.attempts + 1,
            )
            # Sorted re-insert by (enqueued_at, record_id): the retried record returns
            # to its true age position. Drainer backoff prevents tight spinning.
            self._sorted_insert(requeued, cls, skip_capacity=True)

    def requeue_inflight(self) -> None:
        # Covers both IN_FLIGHT (handed to adapter, not yet on wire) and SENT
        # (on wire, awaiting ack) — both states live in _in_flight.
        with self._lock:
            for record in list(self._in_flight.values()):
                cls = self._resolve_class(record.priority)
                if cls.name != record.priority:
                    record = replace(record, priority=cls.name)
                self._enqueue_to_class(replace(record, state=RecordState.QUEUED), cls)
            self._in_flight.clear()
            self._sent_at.clear()

    def sweep_unacked(self) -> list[str]:
        """Requeue SENT records whose ack is overdue.

        Returns the list of record_ids that were requeued. Records in IN_FLIGHT
        are left alone (no _sent_at stamp until the adapter hands them off).
        """
        requeued_ids: list[str] = []
        with self._lock:
            now = self._clock()
            stale = [rid for rid, ts in self._sent_at.items() if (now - ts) > self._unacked_ttl_s]
            for rid in stale:
                record = self._in_flight.pop(rid, None)
                self._sent_at.pop(rid, None)
                if record is None:
                    continue
                cls = self._resolve_class(record.priority)
                if cls.name != record.priority:
                    record = replace(record, priority=cls.name)
                requeued = replace(
                    record,
                    state=RecordState.QUEUED,
                    attempts=record.attempts + 1,
                )
                self._sorted_insert(requeued, cls, skip_capacity=True)
                requeued_ids.append(rid)
        return requeued_ids

    def oldest_pending_age_ms(self) -> Optional[float]:
        """Age in ms of the oldest queued-but-not-yet-sent record, or None if empty."""
        with self._lock:
            oldest_age: Optional[float] = None
            now = self._clock()
            for q in self._queues.values():
                if q:
                    # deque is sorted ascending by enqueued_at; front is always oldest
                    age = now - q[0].enqueued_at
                    if oldest_age is None or age > oldest_age:
                        oldest_age = age
            return oldest_age * 1000.0 if oldest_age is not None else None

    def stats(self) -> dict[str, QueueStats]:
        with self._lock:
            depth_by_class: dict[str, int] = {p: len(q) for p, q in self._queues.items()}
            inflight_by_class: dict[str, int] = dict.fromkeys(self._queues, 0)
            for r in self._in_flight.values():
                inflight_by_class[r.priority] = inflight_by_class.get(r.priority, 0) + 1
            return {
                p: QueueStats(
                    depth=depth_by_class[p],
                    in_flight=inflight_by_class.get(p, 0),
                    dropped=self._dropped.get(p, 0),
                )
                for p in self._queues
            }

    # ---- dynamic class registry ----------------------------------------------

    def add_class(self, cls: PriorityClass) -> None:
        if cls.name == self._default_name:
            raise ValueError(f"cannot re-add default class {cls.name!r}")
        with self._lock:
            if cls.name in self._classes:
                raise ValueError(f"class {cls.name!r} already exists")
            self._classes[cls.name] = cls
            self._queues[cls.name] = deque()
            self._dropped[cls.name] = 0

    def remove_class(self, name: str, force: bool = False) -> int:
        """Remove a class. Returns the number of records migrated to the default class.

        Refuses (raises ValueError) if the class has queued or in-flight records and
        force is False. With force=True, queued records are physically moved to the
        default class, and in-flight records' priority is rewritten to the default
        name so they land in default if/when they fail-retry.
        """
        if name == self._default_name:
            raise ValueError(f"cannot remove default class {name!r}")
        with self._lock:
            cls = self._classes.get(name)
            if cls is None:
                raise KeyError(name)
            q = self._queues[name]
            in_flight_in_class = [rid for rid, r in self._in_flight.items() if r.priority == name]
            if not force and (q or in_flight_in_class):
                raise ValueError(f"class {name!r} not empty (queued={len(q)}, in_flight={len(in_flight_in_class)}); pass force=True to migrate to default")
            migrated = 0
            default = self._classes[self._default_name]
            while q:
                record = q.popleft()
                record = replace(record, priority=self._default_name)
                self._enqueue_to_class(record, default)
                migrated += 1
            for rid in in_flight_in_class:
                self._in_flight[rid] = replace(self._in_flight[rid], priority=self._default_name)
                migrated += 1
            del self._classes[name]
            del self._queues[name]
            del self._dropped[name]
            return migrated

    def replace_class(self, cls: PriorityClass) -> None:
        """Replace the definition of an existing class in-place.

        Queued records are not moved — they reference the class by name and pick up
        the new parameters on the next scheduling pass. Raises KeyError if the class
        does not exist; raises ValueError for the default class.
        """
        if cls.name == self._default_name:
            raise ValueError(f"cannot replace default class {cls.name!r}")
        with self._lock:
            if cls.name not in self._classes:
                raise KeyError(cls.name)
            self._classes[cls.name] = cls

    # ---- internal helpers (caller holds self._lock) --------------------------

    def _resolve_class(self, name: str) -> PriorityClass:
        return self._classes.get(name) or self._classes[self._default_name]

    def _age(self, record: QueueRecord) -> float:
        return self._clock() - record.enqueued_at

    def _enqueue_to_class(self, record: QueueRecord, cls: PriorityClass) -> None:
        self._sorted_insert(replace(record, state=RecordState.QUEUED), cls, skip_capacity=False)

    def _sorted_insert(
        self,
        record: QueueRecord,
        cls: PriorityClass,
        *,
        skip_capacity: bool,
    ) -> None:
        """Insert record into the class deque maintaining ascending (enqueued_at, record_id) order.

        Fast path: if record is newer than the current tail (the common case for fresh
        enqueues, since wall clock grows monotonically), append in O(1) and skip bisect.
        Otherwise use bisect.insort_right with an explicit key on the deque — O(log n)
        search + O(min(i, len-i)) shift via deque.insert.

        skip_capacity: True for retries (the record was already counted against capacity
        when first enqueued; we don't want to drop a different record just because the
        class is briefly full). False for fresh enqueues.
        """
        q = self._queues[cls.name]
        if not skip_capacity and cls.capacity is not None and len(q) >= cls.capacity:
            # drop-oldest within class: front of deque is always oldest
            q.popleft()
            self._dropped[cls.name] += 1
        if not q or (record.enqueued_at, record.record_id) >= (q[-1].enqueued_at, q[-1].record_id):
            q.append(record)
            return
        bisect.insort_right(q, record, key=lambda r: (r.enqueued_at, r.record_id))

    def _drop_expired_and_peek(self, cls: PriorityClass) -> Optional[QueueRecord]:
        """Discard tail records older than drop_after, then return the scheduling head.

        Front of the deque is always the oldest record (sorted invariant). The
        scheduling head differs by class order: FIFO returns the front, LIFO the back.
        """
        q = self._queues[cls.name]
        drop_after = cls.decay.drop_after if cls.decay else None
        if drop_after is not None:
            while q and self._age(q[0]) > drop_after:
                q.popleft()
                self._dropped[cls.name] += 1
        if not q:
            return None
        return q[0] if cls.order == ClassOrder.FIFO else q[-1]

    def _reassign_head(self, source: PriorityClass, target: PriorityClass) -> None:
        src_q = self._queues[source.name]
        if source.order == ClassOrder.FIFO:
            record = src_q.popleft()
        else:
            record = src_q.pop()
        record = replace(record, priority=target.name)
        # demoted/migrated records carry their original enqueued_at, so sorted insert
        # places them at the correct age position in the target class — no leapfrog.
        self._sorted_insert(record, target, skip_capacity=False)

    def _pop_and_mark_inflight(self, cls: PriorityClass) -> QueueRecord:
        q = self._queues[cls.name]
        record = q.popleft() if cls.order == ClassOrder.FIFO else q.pop()
        record = replace(record, state=RecordState.IN_FLIGHT)
        self._in_flight[record.record_id] = record
        return record
