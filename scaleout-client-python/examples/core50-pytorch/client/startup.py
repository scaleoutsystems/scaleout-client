import math
import os
import sys

import torch

from data import (
    NUM_DOMAINS,
    load_domain,
    prepare_data,
)
from model import load_parameters, save_parameters

from scaleout import EdgeClient
from scaleoututil.evaluation import ContinualLearningEvaluator
from scaleoututil.utils.model import ScaleoutModel

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def _required_int_env(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            f"Launch this client via run.py at the project root."
        )
    return int(value)


def startup(client: EdgeClient):
    """Entry point called by Scaleout Edge."""
    prepare_data()
    MyClient(client)


class MyClient:
    def __init__(self, client: EdgeClient):
        self.client = client
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.client_number = _required_int_env("CLIENT_NUMBER")
        self.total_n_clients = _required_int_env("TOTAL_N_CLIENTS")

        # Domains are registered lazily, one per round, as the client first
        # sees each one. This mirrors a real continual-learning deployment
        # where the set of known domains grows over time. Registration is
        # driven entirely by train(), which carries a reliable round_id;
        # validate() only evaluates what train() has already registered.
        self.evaluator = ContinualLearningEvaluator(eval_fn=self._eval_fn, primary="accuracy")
        # Domain index of the most recent train() call, used to label
        # validation metrics (validation tasks carry no usable round_id).
        self._last_trained_domain = None

        client.set_train_callback(self.train)
        client.set_validate_callback(self.validate)

    def _ensure_domain_registered(self, domain_index: int, pretrain_model=None) -> str:
        """Register the test slice for ``domain_index`` once. Returns the stable label.

        Called only from train() (which carries a reliable round_id). If
        ``pretrain_model`` is supplied AND the domain hasn't been registered
        yet, the evaluator runs a pre-training measurement on that model right
        at registration time -- giving the CL panel a Forward Transfer signal
        captured just before this round trains on the domain.
        """
        label = f"domain{domain_index}"
        if label not in self.evaluator.domains:
            test_slice = load_domain(
                domain_index,
                self.client_number,
                self.total_n_clients,
                is_train=False,
            )
            self.evaluator.add_domain(test_slice, label=label, pre_evaluate_with=pretrain_model)
        return label

    def _round_to_domain(self):
        """Map server-side round_id (1-based string) to a domain index, round-robin.

        Rounds cycle through the domains: round 1 -> domain 0, ..., round
        NUM_DOMAINS -> domain NUM_DOMAINS-1, round NUM_DOMAINS+1 -> domain 0,
        and so on. This lets training continue past NUM_DOMAINS rounds,
        revisiting domains in order (domain1, 2, ..., 8, 1, 2, ...).

        Returns ``None`` when the task carries no usable round_id (missing or
        non-numeric). ``round_id`` is a protobuf string that defaults to ""
        for tasks not tied to a numbered FL round; mapping those to a domain
        would silently register/evaluate ``domain0`` and pollute the
        continual-learning metrics. Callers must skip such tasks.
        """
        ctx = self.client.current_logging_context
        round_id = ctx.round_id if ctx is not None else None
        try:
            r = int(round_id) - 1
        except (TypeError, ValueError):
            return None
        return max(0, r) % NUM_DOMAINS

    def train(
        self,
        scaleout_model: ScaleoutModel,
        settings,
        data_path=None,
        batch_size=32,
        epochs=1,
        lr=0.01,
    ):
        """Train on this client's slice of the current round's domain only."""
        domain_index = self._round_to_domain()
        if domain_index is None:
            raise RuntimeError(
                "train() received a task with no usable round_id; cannot determine "
                "which domain to train on. Training tasks are expected to carry a "
                "numeric round_id from the server."
            )
        self._last_trained_domain = domain_index
        # Register this domain with the evaluator, passing the *incoming*
        # model so the evaluator can capture a pre-train (FWT) measurement
        # if this is the first time we see this domain. The pretrain pass
        # is skipped if the domain was already registered by an earlier
        # validate() in this FL round.
        pretrain_model = load_parameters(scaleout_model).to(self.device).eval()
        self._ensure_domain_registered(domain_index, pretrain_model=pretrain_model)
        x_train, y_train = load_domain(
            domain_index,
            self.client_number,
            self.total_n_clients,
            is_train=True,
        )
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device).long()

        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        n_samples = x_train.shape[0]
        n_batches = max(1, int(math.ceil(n_samples / batch_size)))

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # Reshuffle each epoch (seeded by epoch for reproducibility).
            perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(epoch))

            for b in range(n_batches):
                self.client.check_task_abort()

                idx = perm[b * batch_size : (b + 1) * batch_size]
                batch_x = x_train[idx]
                batch_y = y_train[idx]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

                if b % 50 == 0:
                    print(
                        f"[client {self.client_number}] domain {domain_index} "
                        f"epoch {epoch}/{epochs - 1} batch {b}/{n_batches - 1} "
                        f"loss={loss.item():.4f}"
                    )

            epoch_loss = running_loss / max(total, 1)
            epoch_acc = correct / total if total > 0 else 0.0

            self.client.log_metric(
                {
                    "training_loss": float(epoch_loss),
                    "training_accuracy": float(epoch_acc),
                    "domain_index": float(domain_index),
                }
            )
            print(
                f"[client {self.client_number}] domain {domain_index} "
                f"epoch {epoch} done loss={epoch_loss:.4f} acc={epoch_acc:.4f}"
            )

        metadata = {
            "num_examples": int(n_samples),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "domain_index": int(domain_index),
        }

        result_model = save_parameters(model)
        return result_model, {"training_metadata": metadata}

    def validate(self, scaleout_model: ScaleoutModel, data_path=None):
        """Evaluate the global model against every domain seen so far.

        Validation tasks are dispatched by the server without a usable
        round_id, so the domain is NOT derived here. Domain registration (and
        the FWT pre-train capture) is handled entirely by train(); validate()
        simply scores the incoming global model against whatever domains
        train() has registered so far.
        """
        if not self.evaluator.domains:
            # No domain has been trained yet -> nothing to evaluate.
            return None

        model = load_parameters(scaleout_model)
        model.to(self.device)
        model.eval()

        metrics = self.evaluator.evaluate(model)
        if self._last_trained_domain is not None:
            metrics["domain_index"] = float(self._last_trained_domain)
        return metrics

    def _eval_fn(self, model, dataset) -> dict:
        """Evaluator-facing eval function: runs no-grad inference over one domain slice."""
        x, y = dataset
        x = x.to(self.device)
        y = y.to(self.device).long()

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        total = x.shape[0]
        if total == 0:
            return {"accuracy": 0.0, "loss": 0.0, "num_examples": 0.0}

        batch_size = 128
        loss_sum = 0.0
        correct = 0
        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch_x = x[start : start + batch_size]
                batch_y = y[start : start + batch_size]
                outputs = model(batch_x)
                loss_sum += criterion(outputs, batch_y).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()

        return {
            "accuracy": float(correct) / total,
            "loss": loss_sum / total,
        }
