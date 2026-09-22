import pytest

from scaleoututil.evaluation import ContinualLearningEvaluator


def make_eval_fn(matrix):
    """Build an eval_fn driven by a hand-built results matrix.

    The fake "model" is the round int and the per-domain "dataset" is the
    domain label, so ``eval_fn(model, dataset)`` looks up
    ``matrix[round][label]``. This lets each test assert CL aggregates against
    a known accuracy table. ``loss`` is returned as a non-primary key so the
    avg-only emission path is exercised too.
    """

    def eval_fn(model, dataset):
        acc = matrix[model][dataset]
        return {"accuracy": acc, "loss": 1.0 - acc}

    return eval_fn


def test_first_round_returns_per_domain_and_plasticity_only():
    matrix = {0: {"d0": 0.9}}
    ev = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")
    ev.add_domain("d0", label="d0")
    metrics = ev.evaluate(model=0)

    # primary is emitted per-domain; non-primary (loss) is avg-only.
    assert metrics["d0_accuracy"] == pytest.approx(0.9)
    assert "d0_loss" not in metrics
    assert metrics["avg_accuracy"] == pytest.approx(0.9)
    assert metrics["avg_loss"] == pytest.approx(0.1)
    assert metrics["plasticity_accuracy"] == pytest.approx(0.9)
    # forgetting / bwt are undefined when no domain has post-baseline history.
    assert "forgetting_accuracy" not in metrics
    assert "bwt_accuracy" not in metrics


def test_forgetting_bwt_against_worked_example():
    # Three domains, registered lazily one per round (the round each is first
    # trained, so its first eval is the post-train baseline):
    #   round 0: d0 -> 0.9
    #   round 1: d0 -> 0.6, d1 -> 0.8
    #   round 2: d0 -> 0.5, d1 -> 0.7, d2 -> 0.85
    matrix = {
        0: {"d0": 0.9},
        1: {"d0": 0.6, "d1": 0.8},
        2: {"d0": 0.5, "d1": 0.7, "d2": 0.85},
    }
    ev = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")

    ev.add_domain("d0", label="d0")
    ev.evaluate(model=0)
    ev.add_domain("d1", label="d1")
    ev.evaluate(model=1)
    ev.add_domain("d2", label="d2")
    final = ev.evaluate(model=2)

    # plasticity = mean of each domain's value at its first post-train eval:
    #   mean(d0@0=0.9, d1@1=0.8, d2@2=0.85)
    assert final["plasticity_accuracy"] == pytest.approx((0.9 + 0.8 + 0.85) / 3)
    # avg accuracy after final round = mean(0.5, 0.7, 0.85)
    assert final["avg_accuracy"] == pytest.approx((0.5 + 0.7 + 0.85) / 3)
    # forgetting = mean over domains with post-baseline history of
    # (max past acc) - current acc:
    #   d0: max(0.9, 0.6) - 0.5 = 0.4
    #   d1: max(0.8)      - 0.7 = 0.1   (d2 has no post-baseline history)
    assert final["forgetting_accuracy"] == pytest.approx((0.4 + 0.1) / 2)
    # bwt = mean over the same set of (current acc) - (baseline acc):
    #   d0: 0.5 - 0.9 = -0.4
    #   d1: 0.7 - 0.8 = -0.1
    assert final["bwt_accuracy"] == pytest.approx((-0.4 + -0.1) / 2)


def test_forward_transfer_from_pretrain_value():
    # eval_fn keys on the "model" marker: a pre-train pass scores 0.2, the
    # post-train round scores 0.7.
    def eval_fn(model, dataset):
        return {"accuracy": {"pre": 0.2, "post": 0.7}[model]}

    ev = ContinualLearningEvaluator(eval_fn=eval_fn, primary="accuracy")
    # pre_evaluate_with captures the FWT measurement at registration time.
    ev.add_domain("dA", label="A", pre_evaluate_with="pre")
    metrics = ev.evaluate(model="post")

    assert metrics["A_accuracy"] == pytest.approx(0.7)
    assert metrics["fwt_accuracy"] == pytest.approx(0.2)


def test_arbitrary_labels_are_not_interpreted():
    matrix = {0: {"night": 0.4, "day": 0.6}}

    def eval_fn(model, dataset):
        return {"accuracy": matrix[model][dataset]}

    ev = ContinualLearningEvaluator(eval_fn=eval_fn, primary="accuracy")
    ev.add_domain("night", label="night")
    ev.add_domain("day", label="day")
    metrics = ev.evaluate(model=0)

    assert metrics["night_accuracy"] == pytest.approx(0.4)
    assert metrics["day_accuracy"] == pytest.approx(0.6)
    assert metrics["avg_accuracy"] == pytest.approx(0.5)


def test_missing_primary_metric_raises():
    def bad_fn(model, dataset):
        return {"loss": 0.1}

    ev = ContinualLearningEvaluator(eval_fn=bad_fn, primary="accuracy")
    ev.add_domain("d0", label="d0")
    with pytest.raises(KeyError):
        ev.evaluate(model=0)


def test_non_finite_primary_metric_raises():
    def nan_fn(model, dataset):
        return {"accuracy": float("nan")}

    ev = ContinualLearningEvaluator(eval_fn=nan_fn, primary="accuracy")
    ev.add_domain("d0", label="d0")
    with pytest.raises(ValueError):
        ev.evaluate(model=0)


def test_history_returned_is_a_copy():
    matrix = {0: {"d0": 0.9}}
    ev = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")
    ev.add_domain("d0", label="d0")
    ev.evaluate(model=0)

    h = ev.history()
    h[0]["d0"]["accuracy"] = -1.0
    assert ev.history()[0]["d0"]["accuracy"] == pytest.approx(0.9)


def test_non_primary_keys_are_avg_only_not_per_domain():
    def eval_fn(model, dataset):
        return {"accuracy": 0.7, "loss": 0.3, "samples": 42.0}

    ev = ContinualLearningEvaluator(eval_fn=eval_fn, primary="accuracy")
    ev.add_domain("a", label="a")
    metrics = ev.evaluate(model=None)

    # primary is per-domain; everything else is cross-domain average only.
    assert metrics["a_accuracy"] == pytest.approx(0.7)
    assert "a_loss" not in metrics
    assert "a_samples" not in metrics
    assert metrics["avg_loss"] == pytest.approx(0.3)
    assert metrics["avg_samples"] == pytest.approx(42.0)


def test_save_and_load_state_roundtrip(tmp_path):
    matrix = {
        0: {"d0": 0.9},
        1: {"d0": 0.6, "d1": 0.8},
    }
    path = str(tmp_path / "state.json")

    ev = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")
    ev.add_domain("d0", label="d0")
    ev.evaluate(model=0)
    ev.add_domain("d1", label="d1")
    ev.evaluate(model=1)
    assert ev.round == 2
    ev.save_state(path)

    # Resume: fresh evaluator, re-pass eval_fn, re-register domains, then load.
    resumed = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")
    resumed.add_domain("d0", label="d0")
    resumed.add_domain("d1", label="d1")
    resumed.load_state(path)

    assert resumed.round == 2
    assert resumed.history() == ev.history()


def test_load_state_rejects_mismatched_primary(tmp_path):
    matrix = {0: {"d0": 0.9}}
    path = str(tmp_path / "state.json")

    ev = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="accuracy")
    ev.add_domain("d0", label="d0")
    ev.evaluate(model=0)
    ev.save_state(path)

    other = ContinualLearningEvaluator(eval_fn=make_eval_fn(matrix), primary="f1")
    with pytest.raises(ValueError):
        other.load_state(path)
