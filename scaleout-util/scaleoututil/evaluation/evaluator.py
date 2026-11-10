"""Continual-learning evaluator.

A framework-agnostic helper for federated continual-learning workloads. The
client registers domains over time via :meth:`ContinualLearningEvaluator.add_domain`,
supplies an ``eval_fn`` that scores a model against one domain's dataset, and
calls :meth:`ContinualLearningEvaluator.evaluate` once per round. The evaluator
runs ``eval_fn`` across every registered domain, records the primary metrics,
and returns a flat metric dict suitable for ``client.log_metric``.

Emission contract per round:

* **Per-domain raw**: ``{label}_{primary}`` for each primary metric. Non-primary
  keys returned by ``eval_fn`` are NOT emitted per-domain.
* **Cross-domain mean**: ``avg_{key}`` for every key returned by ``eval_fn``
  (primary or not). So ``avg_loss``, ``avg_num_examples`` etc. are useful
  aggregates even though ``loss`` is not a primary metric.
* **CL aggregates**: ``plasticity_{primary}``, ``fwt_{primary}``,
  ``forgetting_{primary}``, ``bwt_{primary}`` -- only for primary metrics,
  since the definitions don't generalize to arbitrary ``eval_fn`` outputs.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

_STATE_VERSION = 2


class ContinualLearningEvaluator:
    """Stateful evaluator for federated continual learning.

    This evaluator is built for the **domain-incremental continual-learning**
    setting and computes its CL aggregates (plasticity, FWT, forgetting, BWT)
    under the following assumptions. They are baked into the metric
    definitions, so read them before use -- a different setup likely wants a
    different evaluator class rather than this one bent to fit.

    1. **Stream of domains, arbitrary labels.** Domains (a.k.a. tasks) are
       encountered over time as a stream. Labels are arbitrary strings
       (``"night"``, ``"day"``, ...) -- no ordering or numbering is implied,
       and the evaluator never interprets the label itself. The only temporal
       signal it uses is *when* each domain is first registered/evaluated.
    2. **Lazy registration.** Each domain is registered (via
       :meth:`add_domain`) in the round you first train on it -- not up front.
       Registering the full domain set eagerly at startup is NOT supported:
       every domain's baseline would collapse to round 0 and forgetting/BWT
       would be meaningless.
    3. **Evaluate after training.** Within the round a domain first appears,
       evaluation runs *after* training on it, so the domain's first recorded
       eval is its post-training baseline -- the anchor for forgetting/BWT.
    4. **One shared model, evaluated across all domains seen so far** each
       round.

    Out of scope: joint / multi-task training (all domain data available
    simultaneously and trained together) is not continual learning -- the
    forgetting/transfer aggregates still compute but no longer mean anything.

    :param eval_fn: Callable ``(model, domain_dataset) -> dict``. Must include
        every name listed in ``primary`` in its returned dict. May return
        additional keys for the user's own bookkeeping; they are ignored by
        the evaluator (not emitted, not stored in history).
    :param primary: Name of the primary metric, or an iterable of names when
        multiple metrics should each receive the full CL aggregate panel.
        Default ``"accuracy"``.

    Persistence is explicit: call :meth:`save_state` to atomically write the
    serializable state (history matrix, per-domain first-eval round, primary
    metric names, captured pre-train values) to a path, and :meth:`load_state`
    to restore it so CL history survives a client restart. ``eval_fn`` and the
    per-domain dataset objects themselves are NOT serialized; the caller must
    re-pass ``eval_fn`` and re-call :meth:`add_domain` for each previously
    known label on the new run.

    Example::

        ev = ContinualLearningEvaluator(eval_fn=score_fn, primary="accuracy")
        ev.add_domain(test_set_A, label="A")
        m = ev.evaluate(model)               # round 1, one domain: raw only
        ev.add_domain(test_set_B, label="B")
        m = ev.evaluate(model)               # round 2, two domains: full panel
        ev.save_state("eval_state.json")     # persist when the caller chooses
    """

    def __init__(
        self,
        eval_fn: Callable[[Any, Any], Dict[str, float]],
        primary: Union[str, Iterable[str]] = "accuracy",
    ) -> None:
        if isinstance(primary, str):
            primary_metrics: Tuple[str, ...] = (primary,)
        else:
            primary_metrics = tuple(primary)
        if not primary_metrics:
            raise ValueError("primary must name at least one metric")
        self.primary_metrics = primary_metrics
        self.eval_fn = eval_fn

        # Ordered list of (label, dataset). Order is the registration order.
        self._domains: List[Tuple[str, Any]] = []
        # label -> 0-indexed round in which the domain was first evaluated.
        # Evaluation is assumed to run AFTER training on a domain within the
        # round it first appears, so a domain's first-eval entry is already a
        # post-train measurement and serves as the baseline for forgetting/BWT.
        self._domain_first_eval: Dict[str, int] = {}
        # label -> dict of eval_fn outputs captured BEFORE training on this
        # domain (via ``add_domain(evaluate_with=model)``). Off-history side
        # channel: FWT reads from here directly, so it doesn't depend on the
        # client/server dispatch order (train-first vs validate-first).
        self._pretrain_value: Dict[str, Dict[str, float]] = {}
        # One dict per round: {label: {metric_name: value}}.
        self._history: List[Dict[str, Dict[str, float]]] = []

    # -- registration ----------------------------------------------------------
    # .add_domain() - when new domains come in
    # .pre_train_evaluate() - enables FWT by evaluating before training
    # .evaluate() - after training
    def add_domain(
        self,
        dataset: Any,  # noqa: ANN401 - generic by design
        label: Optional[str] = None,
        pre_evaluate_with: Any = None,  # noqa: ANN401 - generic by design
    ) -> str:
        """Register a new domain. Returns the (possibly auto-generated) label.

        Subsequent ``evaluate`` calls will include this domain. Re-using an
        existing label raises ``ValueError`` so registrations are explicit.

        :param dataset: Opaque object passed verbatim to ``eval_fn`` whenever
            this domain is evaluated.
        :param label: Optional stable identifier used as the prefix in emitted
            metric keys (``{label}_{metric}``). If omitted, an auto-numbered
            label ``"domain{i}"`` is generated.
        :param pre_evaluate_with: Optional model object. When provided, the
            evaluator runs ``eval_fn(pre_evaluate_with, dataset)`` immediately and
            stores the result as the domain's *pre-train* measurement -- the
            measurement Forward Transfer is computed from. This is an
            off-history side channel; it does not advance the round counter
            and does not show up in :meth:`history`. Useful when registering
            a domain just before training on it, to capture the "what does
            the model already know about this domain?" value in a way that
            doesn't depend on dispatch order.
        """
        if label is None:
            label = f"domain{len(self._domains)}"
        if any(existing == label for existing, _ in self._domains):
            raise ValueError(f"Domain label {label!r} is already registered.")
        self._domains.append((label, dataset))

        if pre_evaluate_with is not None:
            raw = self.eval_fn(pre_evaluate_with, dataset)
            for p in self.primary_metrics:
                if p not in raw:
                    raise KeyError(
                        f"eval_fn did not return required primary metric {p!r} for pre-train evaluation of domain {label!r} (got: {sorted(raw.keys())})"
                    )
                value = float(raw[p])
                if not math.isfinite(value):
                    raise ValueError(f"eval_fn returned non-finite pre-train value {p}={raw[p]!r} for domain {label!r}.")
            # Store only primary metrics: non-primary eval_fn outputs aren't
            # surfaced anywhere downstream, so keeping them here would just
            # bloat the persisted state file.
            self._pretrain_value[label] = {p: float(raw[p]) for p in self.primary_metrics}

        return label

    @property
    def domains(self) -> List[str]:
        """Labels of all currently registered domains, in registration order."""
        return [label for label, _ in self._domains]

    @property
    def round(self) -> int:
        """Number of completed ``evaluate`` calls (0 before the first one)."""
        return len(self._history)

    def history(self) -> List[Dict[str, Dict[str, float]]]:
        """A deep-ish copy of the recorded per-round, per-domain history."""
        return [{label: dict(metrics) for label, metrics in r.items()} for r in self._history]

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, model: Any) -> Dict[str, float]:  # noqa: ANN401 - generic by design
        """Run ``eval_fn`` against every registered domain and return a flat metric dict.

        Emits per-domain primary values, cross-domain means, and primary-only
        CL aggregates:

        * ``{label}_{p}`` for every domain and every primary metric ``p``.
        * ``avg_{key}`` for every key returned by ``eval_fn`` -- primary or
          not (so ``avg_loss``, ``avg_num_examples`` etc. show up too).
        * ``plasticity_{p}`` -- mean of each domain's value of ``p`` at its
          first post-train eval (its first-eval round). Emitted whenever ≥1
          domain is registered.
        * ``fwt_{p}`` -- mean across domains whose pre-train value was
          captured via :meth:`add_domain` with ``pre_evaluate_with=...``.
          Emitted whenever ≥1 such domain exists.
        * ``forgetting_{p}`` -- mean over domains seen in ≥1 round past
          their baseline of ``max(past) - current`` of ``p``.
        * ``bwt_{p}`` -- mean over the same defined-set of
          ``current - baseline`` of ``p``.

        CL aggregates are intentionally cross-domain only -- per-domain
        time-series of ``p`` are already visible via the ``{label}_{p}``
        keys. Aggregates are silently omitted in rounds where they are
        undefined (e.g. ``forgetting`` in a round where no domain has
        post-baseline history yet).
        """
        if not self._domains:
            return {}

        round_index = self.round  # 0-indexed; equal to len before the append below.
        round_metrics: Dict[str, Dict[str, float]] = {}
        for label, dataset in self._domains:
            raw = self.eval_fn(model, dataset)
            for p in self.primary_metrics:
                if p not in raw:
                    raise KeyError(f"eval_fn did not return required primary metric {p!r} for domain {label!r} (got keys: {sorted(raw.keys())})")
                value = float(raw[p])
                # Reject non-finite primaries up front -- a NaN here would
                # otherwise infect every downstream aggregate via the history
                # matrix (max(NaN, ...) == NaN, etc.) and silently mislabel
                # future rounds.
                if not math.isfinite(value):
                    raise ValueError(f"eval_fn returned non-finite primary metric {p}={raw[p]!r} for domain {label!r} at round {round_index}.")
            # Store every key returned by eval_fn: non-primary values are
            # not emitted per-domain, but their cross-domain ``avg_{key}`` IS,
            # so we need them in history. (CL aggregates stay primary-only.)
            round_metrics[label] = {k: float(v) for k, v in raw.items()}
            self._domain_first_eval.setdefault(label, round_index)

        self._history.append(round_metrics)
        return self._build_output()

    # -- output assembly -------------------------------------------------------

    def _build_output(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        current_idx = self.round - 1
        current = self._history[current_idx]

        # 1. Per-domain raw values, primary metrics only. Non-primary keys
        # returned by eval_fn (loss, num_examples, ...) are NOT surfaced
        # per-domain -- the dashboard would otherwise be 3-5x larger for
        # information that isn't usually inspected per-domain.
        for label, metrics in current.items():
            for primary in self.primary_metrics:
                out[f"{label}_{primary}"] = float(metrics[primary])

        # 2. Cross-domain mean for every key (primary or not). Useful for
        # things like total loss / total sample count without forcing the
        # user to inspect every domain individually.
        all_keys = set()
        for metrics in current.values():
            all_keys.update(metrics.keys())
        for key in all_keys:
            values = [m[key] for m in current.values() if key in m]
            if values:
                out[f"avg_{key}"] = sum(values) / len(values)

        # 3. CL aggregates (plasticity, fwt, forgetting, bwt) only for
        # primary metrics -- these definitions don't generalize to arbitrary
        # ``eval_fn`` outputs.
        for primary in self.primary_metrics:
            self._aggregate_primary(out, current, current_idx, primary)

        return out

    def _baseline_round(self, label: str) -> int:
        """Round used as the baseline for post-training CL metrics.

        This is the domain's first-eval round. Evaluation is assumed to run
        after training on a domain, so that first eval is already a post-train
        measurement -- the correct anchor for forgetting/BWT.
        """
        return self._domain_first_eval[label]

    def _aggregate_primary(
        self,
        out: Dict[str, float],
        current: Dict[str, Dict[str, float]],
        current_idx: int,
        primary: str,
    ) -> None:
        # CL aggregates are emitted as cross-domain means only. Per-domain
        # raw values from eval_fn (e.g. ``domain0_accuracy``) are already
        # emitted in step 1 of _build_output and are sufficient to inspect
        # individual domains in a dashboard; per-domain CL aggregates would
        # just create N times more keys without adding information not
        # already derivable from the raw per-domain time series.
        # ``avg_{primary}`` is handled in step 2 of _build_output along with
        # all the other per-key averages.

        # Plasticity: each domain contributes its value at the FIRST post-train
        # eval (its first-eval round). Bounded by the current history length so
        # a domain registered at the boundary still contributes something
        # sensible.
        plasticity_values: List[float] = []
        for label, _ in self._domains:
            anchor = min(self._baseline_round(label), current_idx)
            plasticity_values.append(self._history[anchor][label][primary])
        out[f"plasticity_{primary}"] = sum(plasticity_values) / len(plasticity_values)

        # Forward Transfer: defined for domains whose pre-train value was
        # explicitly captured at registration via add_domain(pre_evaluate_with=...).
        # Independent of dispatch order.
        fwt_terms: List[float] = []
        for label, _ in self._domains:
            pretrain = self._pretrain_value.get(label)
            if pretrain is None or primary not in pretrain:
                continue
            fwt_terms.append(pretrain[primary])
        if fwt_terms:
            out[f"fwt_{primary}"] = sum(fwt_terms) / len(fwt_terms)

        # Forgetting / BWT are post-training metrics: they only consider the
        # history strictly after the baseline round.
        forgetting_terms: List[float] = []
        bwt_terms: List[float] = []
        for label, _ in self._domains:
            baseline = self._baseline_round(label)
            if current_idx <= baseline:
                continue  # no post-baseline history yet
            past_values = [self._history[k][label][primary] for k in range(baseline, current_idx)]
            current_value = current[label][primary]
            forgetting_terms.append(max(past_values) - current_value)
            bwt_terms.append(current_value - self._history[baseline][label][primary])

        if forgetting_terms:
            out[f"forgetting_{primary}"] = sum(forgetting_terms) / len(forgetting_terms)
        if bwt_terms:
            out[f"bwt_{primary}"] = sum(bwt_terms) / len(bwt_terms)

    # -- persistence -----------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Atomically write the serializable state to ``path`` as JSON.

        Call this whenever you want to checkpoint -- e.g. once per round after
        :meth:`evaluate`. Uses a temp-file-and-rename so a crash mid-write
        can't corrupt an existing checkpoint. ``eval_fn`` and the per-domain
        ``dataset`` objects are intentionally NOT written -- on resume the
        caller is responsible for re-passing ``eval_fn`` and re-registering
        each previously known label via :meth:`add_domain`.
        """
        data = {
            "version": _STATE_VERSION,
            "primary_metrics": list(self.primary_metrics),
            "history": self._history,
            "domain_first_eval": self._domain_first_eval,
            "pretrain_value": self._pretrain_value,
        }
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".evaluator-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                # Best-effort cleanup only; don't mask the original save failure.
                pass
            raise

    def load_state(self, path: str) -> None:
        """Load state previously written by :meth:`save_state`.

        Call this after construction (and after re-passing ``eval_fn`` /
        re-registering domains) to resume CL history across a restart.
        Validates that ``primary_metrics`` haven't changed; otherwise the
        recorded history's CL aggregates would silently mean something
        different than what the caller expects.
        """
        with open(path) as f:
            data = json.load(f)
        version = data.get("version")
        if version not in (1, _STATE_VERSION):
            raise ValueError(f"Evaluator state at {path} has version {version!r}; this code understands versions 1 and {_STATE_VERSION}.")
        stored_primaries = tuple(data.get("primary_metrics", ()))
        if stored_primaries != self.primary_metrics:
            raise ValueError(
                f"Evaluator state at {path} was saved with primary_metrics="
                f"{stored_primaries}, but this ContinualLearningEvaluator was constructed with "
                f"primary_metrics={self.primary_metrics}. Refusing to load."
            )
        self._history = [{label: dict(metrics) for label, metrics in r.items()} for r in data.get("history", [])]
        self._domain_first_eval = {k: int(v) for k, v in data.get("domain_first_eval", {}).items()}
        # Older files may carry a now-unused "trained_round" key; it is ignored.
        # v1/v2 files may have no pretrain_value; treat as "FWT not captured".
        self._pretrain_value = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in data.get("pretrain_value", {}).items()}
