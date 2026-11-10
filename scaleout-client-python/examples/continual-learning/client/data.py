"""CORe50 (NI, mini) data plumbing for the continual-learning example.

Layout on disk:

    client/data/core50/             # Avalanche download cache
    client/data/processed/          # per-domain (X, y) tensor pickles
    client/data/.prepare.lock       # cross-process download lock

Every client at runtime calls ``load_domain(d, client_number, total_n_clients,
is_train)`` which returns this client's deterministic slice of domain ``d``'s
train or test split. The slice depends only on ``(client_number,
total_n_clients)`` and the domain seed, so two clients with the same numbers
get the same slice and disjoint clients never overlap.
"""

import os
from math import floor

import numpy as np
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

DATA_DIR = os.path.join(abs_path, "data")
AVALANCHE_DIR = os.path.join(DATA_DIR, "core50")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PREPARE_LOCK = os.path.join(DATA_DIR, ".prepare.lock")

NUM_DOMAINS = 8           # CORe50 NI has 8 training batches
TRAIN_FRACTION = 0.8      # within each domain, fraction reserved for training
SHUFFLE_SEED_PREFIX = "core50-ni-mini"

# ImageNet normalization for the pretrained ResNet-18 backbone.
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _domain_seed(domain_index: int) -> int:
    # Stable seed independent of Python's hash randomization.
    key = f"{SHUFFLE_SEED_PREFIX}-domain-{domain_index}".encode()
    return int.from_bytes(key, "big") % (2**32 - 1)


def _processed_path(domain_index: int) -> str:
    return os.path.join(PROCESSED_DIR, f"domain_{domain_index}.pt")


def _materialize_domain(domain_index: int, avalanche_dataset) -> None:
    """Convert one Avalanche experience to a single (X, y) tensor pickle."""
    from torch.utils.data import DataLoader

    loader = DataLoader(avalanche_dataset, batch_size=512, num_workers=0)
    xs, ys = [], []
    for batch in loader:
        # Avalanche returns (x, y) or (x, y, t) depending on version.
        x, y = batch[0], batch[1]
        xs.append(x)
        ys.append(y)
    X = torch.cat(xs).contiguous()
    y = torch.cat(ys).to(torch.long).contiguous()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save({"x": X, "y": y}, _processed_path(domain_index))


def prepare_data() -> None:
    """Download CORe50 mini (if needed) and materialize per-domain tensors.

    Idempotent. Safe to call from many client subprocesses concurrently when a
    ``FileLock`` is in use upstream; otherwise the Avalanche download race is
    handled by ``filelock.FileLock`` here.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(AVALANCHE_DIR, exist_ok=True)

    # Fast path: every domain already materialized.
    if all(os.path.exists(_processed_path(d)) for d in range(NUM_DOMAINS)):
        return

    from filelock import FileLock  # noqa: PLC0415 - optional dep, deferred import

    with FileLock(PREPARE_LOCK):
        # Re-check inside the lock to skip work another process just finished.
        if all(os.path.exists(_processed_path(d)) for d in range(NUM_DOMAINS)):
            return

        from avalanche.benchmarks.classic import CORe50  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415

        tx = transforms.Compose([transforms.ToTensor()])
        benchmark = CORe50(
            scenario="ni",
            mini=True,
            run=0,
            object_lvl=False,
            train_transform=tx,
            eval_transform=tx,
            dataset_root=AVALANCHE_DIR,
        )

        for d, experience in enumerate(benchmark.train_stream):
            if d >= NUM_DOMAINS:
                break
            if os.path.exists(_processed_path(d)):
                continue
            _materialize_domain(d, experience.dataset)


def _load_domain_full(domain_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(_processed_path(domain_index), weights_only=True)
    return payload["x"], payload["y"]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if x.max() > 1.5:
        # Avalanche's ToTensor scales to [0,1]; guard against raw uint8 caches.
        x = x / 255.0
    return (x - _NORM_MEAN) / _NORM_STD


def _client_slice(n: int, client_number: int, total_n_clients: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    per_client = floor(n / total_n_clients)
    start = client_number * per_client
    end = start + per_client
    return order[start:end]


def load_domain(
    domain_index: int,
    client_number: int,
    total_n_clients: int,
    is_train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this client's deterministic slice of one domain's train or test split.

    Within each domain we first shuffle by the domain seed to split off a
    fixed test fraction, then shuffle the chosen half by a second domain-stable
    seed and slice by ``client_number``.
    """
    if domain_index < 0 or domain_index >= NUM_DOMAINS:
        raise IndexError(f"domain_index {domain_index} out of range [0, {NUM_DOMAINS})")
    if client_number < 0 or client_number >= total_n_clients:
        raise IndexError(f"client_number {client_number} out of range [0, {total_n_clients})")

    X, y = _load_domain_full(domain_index)
    n_total = X.shape[0]
    split_seed = _domain_seed(domain_index)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n_total)
    n_train = int(floor(TRAIN_FRACTION * n_total))
    half = order[:n_train] if is_train else order[n_train:]

    slice_seed = _domain_seed(domain_index) ^ (0x9E3779B9 if is_train else 0x85EBCA77)
    idx_within_half = _client_slice(len(half), client_number, total_n_clients, slice_seed)
    indices = half[idx_within_half]

    X_slice = _normalize(X[indices])
    y_slice = y[indices]
    return X_slice, y_slice


def get_all_test_domains(client_number: int, total_n_clients: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return this client's per-domain test slices (length NUM_DOMAINS)."""
    return [load_domain(d, client_number, total_n_clients, is_train=False) for d in range(NUM_DOMAINS)]


if __name__ == "__main__":
    prepare_data()
