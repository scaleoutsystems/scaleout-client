"""Export one session's per-round continual-learning metrics to a JSON file.

Maintainer tool. The committed export lets ``tools/make_assets.py`` rebuild the
metric figures without a running deployment.

Where the numbers live
----------------------
This example writes to **two different stores**, and the interesting ones are
not where you would guess:

* ``client.log_metric({...})`` in ``train()`` -> the ``metrics`` table, one row
  per key, carrying a ``round_id``. This is only ``training_loss``,
  ``training_accuracy`` and ``domain_index``. ``round_id`` is a UUID, so round
  order is recovered from first appearance in time, not by parsing it.
* the dict **returned** by ``validate()`` -> the ``validations`` table, the
  whole dict serialized into one ``data`` JSON string. **All** the
  continual-learning metrics (``domain{i}_accuracy``, ``plasticity_*``,
  ``fwt_*``, ``forgetting_*``, ``bwt_*``, ``avg_*``) arrive this way.

Validations carry no ``round_id``. They do carry ``model_id`` -- the aggregate
a round produced, which becomes the next round's input model -- so ranking the
distinct validated ``model_id``s by their first timestamp recovers round order.
``domain_index`` inside the payload is used to cross-check that mapping.

Neither the CLI nor the SDK has a metrics accessor, so both stores are read
over REST, reusing the CLI's auth plumbing (a plain ``scaleout login`` is
enough).

What gets written
-----------------
Per-round **fleet means**, one line per round -- not the raw per-client rows.
Every committed figure is captioned "mean over N clients" and consumes exactly
that, so writing all N per-client validations would triple the file with
numbers nothing reads. Training metrics are fetched (to anchor round numbering)
but not written, for the same reason.

Usage
-----
    python tools/export_run.py --session-id <SID> --out assets/data/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

from scaleout.cli.shared import build_client

LIMIT = "100000"
NUM_DOMAINS = 8


DOMAIN_SESSIONS = ["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8"]
# Values are rounded before writing: the figures render 2-3 decimals, so the
# rest is diff noise. 9 was chosen empirically as the shortest precision that
# still reproduces every committed figure byte-for-byte -- at 6, matplotlib's
# anti-aliasing shifts a few hundred edge pixels on the line chart.
PRECISION = 9


def aggregate(cl_rows, rounds, clients, session_id):
    """Reduce per-client validations to the per-round fleet means the figures use.

    Only the mean is written. Every committed figure is captioned "mean over N
    clients" and consumes exactly that, so storing all N per-client rows would
    triple the file for numbers nothing reads. Re-run this exporter if you ever
    need the per-client spread.
    """
    means, trained = {}, {}
    for rnd in rounds:
        rows = [r for r in cl_rows if r["round_id"] == rnd]
        buckets = {}
        for r in rows:
            if r["domain_index"] is not None:
                trained[str(rnd)] = r["domain_index"]
            for k, v in r["metrics"].items():
                buckets.setdefault(k, []).append(v)
        means[str(rnd)] = {k: round(sum(v) / len(v), PRECISION) for k, v in sorted(buckets.items())}
    return {
        "session_id": session_id,
        "n_clients": len(clients),
        "rounds": rounds,
        "trained": trained,
        "domain_sessions": DOMAIN_SESSIONS,
        "mean": means,
    }


def write_compact(payload, path):
    """One line per round, so a regenerated export produces a readable diff."""
    rounds = payload["rounds"]
    lines = [
        "{",
        f'  "session_id": {json.dumps(payload["session_id"])},',
        f'  "n_clients": {payload["n_clients"]},',
        f'  "rounds": {json.dumps(rounds)},',
        f'  "trained": {json.dumps(payload["trained"])},',
        f'  "domain_sessions": {json.dumps(payload["domain_sessions"])},',
        '  "mean": {',
    ]
    for i, rnd in enumerate(rounds):
        comma = "," if i < len(rounds) - 1 else ""
        lines.append(f'    "{rnd}": {json.dumps(payload["mean"][str(rnd)], sort_keys=True)}{comma}')
    lines += ["  }", "}", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _get(client, path, params=None, sort_key="committed_at"):
    headers = {
        **client._get_headers(),
        "X-Sort-Key": sort_key,
        "X-Sort-Order": "asc",
        "X-Limit": LIMIT,
    }
    resp = requests.get(client._get_url_api_v1(path), headers=headers, params=params or {}, verify=client.verify)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    sid = args.session_id

    _, client = build_client(None, args.host, None, args.token)

    # client_id -> human name, so the export is readable and stable.
    names = {}
    for c in _get(client, "clients/").get("result", []):
        if c.get("client_id"):
            names[c["client_id"]] = c.get("name") or c["client_id"]

    # --- training metrics (have round_id) ---
    metrics = [m for m in _get(client, "metrics/", {"session_id": sid}).get("result", []) if m.get("round_id") is not None]

    # round_id is a UUID and carries no ordering, so rank the distinct round_ids
    # by first appearance -> 1-indexed round numbers within this session, which
    # is what the server's own round_number counts.
    round_first_seen = {}
    for m in metrics:
        round_first_seen.setdefault(m["round_id"], m["timestamp"])
    round_number = {rid: i + 1 for i, rid in enumerate(sorted(round_first_seen, key=lambda r: round_first_seen[r]))}

    train_rows = []
    for m in metrics:
        train_rows.append(
            {
                "round_id": round_number[m["round_id"]],
                "client": (m.get("sender") or {}).get("name") or "?",
                "key": m["key"],
                "value": m["value"],
            }
        )

    # --- validations (CL metrics; no round_id) ---
    vals = [v for v in _get(client, "validations/", {"session_id": sid}).get("result", []) if v.get("session_id") == sid]
    if not vals:
        sys.exit(f"No validations found for session {sid}. Has a round finished yet?")

    # Rank distinct validated models by first appearance -> round order.
    first_seen = {}
    for v in vals:
        first_seen.setdefault(v["model_id"], v["timestamp"])
    model_order = sorted(first_seen, key=lambda m: first_seen[m])
    model_rank = {m: i for i, m in enumerate(model_order)}

    # Anchor to the real round numbers from the training metrics when we can,
    # so the export's round_ids line up with what the dashboard shows.
    base_round = min((r["round_id"] for r in train_rows), default=1)

    cl_rows = []
    mismatches = 0
    for v in vals:
        rank = model_rank[v["model_id"]]
        round_id = base_round + rank
        data = json.loads(v["data"]) if isinstance(v["data"], str) else v["data"]
        # Cross-check: the client reports which domain it last trained on.
        di = data.get("domain_index")
        if di is not None and int(di) != (round_id - 1) % NUM_DOMAINS:
            mismatches += 1
        cl_rows.append(
            {
                "round_id": round_id,
                "client": names.get(v.get("node_id"), v.get("node_id") or "?"),
                "domain_index": None if di is None else int(di),
                "metrics": {k: float(x) for k, x in data.items()},
            }
        )
    cl_rows.sort(key=lambda r: (r["round_id"], r["client"]))

    if mismatches:
        print(
            f"WARNING: {mismatches}/{len(cl_rows)} validations disagree with (round-1) % 8; check the round mapping before trusting the figures.",
            file=sys.stderr,
        )

    rounds = sorted({r["round_id"] for r in cl_rows})
    clients = sorted({r["client"] for r in cl_rows})

    payload = aggregate(cl_rows, rounds, clients, sid)
    write_compact(payload, args.out)

    print(f"-> {args.out}")
    print(
        f"rounds {rounds[0]}-{rounds[-1]} ({len(rounds)})  clients {len(clients)}  "
        f"from {len(cl_rows)} per-client validations ({len(train_rows)} training rows fetched, not written)"
    )
    print("keys:", ", ".join(sorted(payload["mean"][str(rounds[0])])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
