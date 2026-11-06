"""Generate the README's figures and animations.

Maintainer tool, not part of the client runtime. Deterministic: same inputs ->
byte-identical output, so re-running does not churn git.

Requires ``pillow``, ``matplotlib`` and ``numpy``. These are deliberately NOT
in ``client/python_env.yaml`` -- clients never need them.

Inputs
------
* CORe50 frames: ``client/data/core50/core50_32x32/s{N}/o{M}/*.png``, created
  by ``python client/data.py``.
* Run metrics: ``assets/data/run.json``, written by ``tools/export_run.py``.

Every asset is rendered twice, once per theme, as ``<name>-light.<ext>`` and
``<name>-dark.<ext>``; the README pairs them with ``<picture>`` +
``prefers-color-scheme``, which GitHub honours. Rasters cannot follow the
viewer's theme on their own.

Usage
-----
    python tools/make_assets.py                 # everything, both themes
    python tools/make_assets.py --only 04       # one asset
    python tools/make_assets.py --theme light   # one theme
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: these assets are rendered to files, never shown

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
FRAMES_DIR = os.path.join(PROJECT, "client", "data", "core50", "core50_32x32")
ASSETS = os.path.join(PROJECT, "assets")
RUN_JSON = os.path.join(ASSETS, "data", "run.json")

# CORe50 "new instances" maps each of the 8 training batches onto exactly one
# recording session. Derived from LUP.pkl + paths.pkl in the Avalanche
# download; see the project CLAUDE.md for the derivation.
DOMAIN_SESSIONS = ["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8"]

# CORe50 has 50 objects in 10 categories, 5 instances each, so category ==
# object_index // 5 and o1/o6/... is the first instance of each category.
# `object_lvl=False` means the example classifies the 10 categories.
CATEGORY_OBJECTS = ["o1", "o6", "o11", "o16", "o21", "o26", "o31", "o36", "o41", "o46"]
CATEGORY_NAMES = [
    "plug adapter",
    "mobile phone",
    "scissors",
    "light bulb",
    "can",
    "glasses",
    "ball",
    "marker",
    "cup",
    "remote control",
]

# Sequential blue ramp, steps 100..700 of the data-viz reference palette.
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


@dataclass(frozen=True)
class Theme:
    """One rendering theme. Values come from the data-viz reference palette."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    gridline: str
    baseline: str
    accent: str  # categorical slot 1
    orange: str  # categorical slot 2, reserved for the headline series
    ordinal: List[str]  # 8 ordered steps: index 0 = earliest domain seen
    seq: List[str]  # continuous magnitude; seq[0] sits nearest the surface
    seq_flip: float  # value above which cell text switches to `seq_ink`
    seq_ink: str

    def seq_color(self, value: float) -> str:
        i = int(round(max(0.0, min(1.0, value)) * (len(self.seq) - 1)))
        return self.seq[i]

    def cell_ink(self, value: float) -> str:
        return self.seq_ink if value >= self.seq_flip else self.ink


# On light, "near zero" recedes toward the surface by getting lighter; on dark
# it recedes by getting darker, so the ramp is reversed rather than reused.
# The ordinal ramp keeps its documented contrast floors: no lighter than step
# 250 on light, no darker than step 600 on dark.
LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    accent="#2a78d6",
    orange="#eb6834",
    ordinal=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"],
    seq=SEQ_STEPS,
    seq_flip=0.46,
    seq_ink="#ffffff",
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    accent="#3987e5",
    orange="#d95926",
    ordinal=["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef"],
    seq=list(reversed(SEQ_STEPS)),
    seq_flip=0.54,
    seq_ink="#0b0b0b",
)
THEMES = {"light": LIGHT, "dark": DARK}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    prop = fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    return ImageFont.truetype(fm.findfont(prop), size)


def _tw(dr: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return dr.textlength(text, font=font)


def _out(name: str, theme: Theme, ext: str) -> str:
    os.makedirs(ASSETS, exist_ok=True)
    return os.path.join(ASSETS, f"{name}-{theme.name}.{ext}")


def _save_gif(frames: List[Image.Image], path: str, durations) -> str:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    return path


# ------------------------------------------------------------------ CORe50 frames


def _require_frames() -> None:
    if not os.path.isdir(FRAMES_DIR):
        sys.exit(f"CORe50 frames not found at {FRAMES_DIR}\nRun: python client/data.py")


def _frame(session: str, obj: str, size: int, pick: float = 0.5) -> Image.Image:
    """One frame of ``obj`` from ``session``, upscaled to ``size``.

    Nearest-neighbour on purpose: the example trains on CORe50 mini at 32x32,
    and the blocky upscale keeps that honest rather than implying the model
    sees a sharp image.
    """
    d = os.path.join(FRAMES_DIR, session, obj)
    files = sorted(os.listdir(d))
    img = Image.open(os.path.join(d, files[int(len(files) * pick)])).convert("RGB")
    return img.resize((size, size), Image.NEAREST)


def asset_01_domain_stream(theme: Theme) -> str:
    """Hero animation: ten fixed categories, the recording session swapping under them."""
    _require_frames()
    tile, gap, margin, label_h = 88, 8, 24, 15
    cols, rows, header, footer = 5, 2, 62, 34
    f_title, f_sub, f_label = _font(15, True), _font(13), _font(9)
    title = "CORe50 · the same ten categories, eight recording sessions"

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    grid_w = cols * tile + (cols - 1) * gap
    width = int(max(grid_w, _tw(probe, title, f_title))) + 2 * margin
    row_h = tile + label_h + gap
    height = header + rows * row_h + footer

    frames = []
    for domain, session in enumerate(DOMAIN_SESSIONS):
        im = Image.new("RGB", (width, height), theme.surface)
        dr = ImageDraw.Draw(im)
        dr.text((margin, 16), title, font=f_title, fill=theme.ink)
        dr.text((margin, 38), f"domain {domain}  ·  session {session}", font=f_sub, fill=theme.accent)
        for i, (obj, name) in enumerate(zip(CATEGORY_OBJECTS, CATEGORY_NAMES)):
            x = margin + (i % cols) * (tile + gap)
            y = header + (i // cols) * row_h
            im.paste(_frame(session, obj, tile), (x, y))
            dr.rectangle([x, y, x + tile - 1, y + tile - 1], outline=theme.gridline)
            dr.text((x, y + tile + 3), name, font=f_label, fill=theme.muted)
        strip_y = height - 20
        seg = (grid_w - 7 * 6) / 8
        for k in range(8):
            x0 = margin + k * (seg + 6)
            dr.rectangle([x0, strip_y, x0 + seg, strip_y + 5], fill=theme.accent if k == domain else theme.gridline)
        frames.append(im)

    return _save_gif(frames, _out("01-domain-stream", theme, "gif"), 950)


def asset_02_same_object(theme: Theme) -> str:
    """Static figure: one physical object, all eight sessions, drift isolated."""
    _require_frames()
    tile, gap, margin, header, label_h = 96, 8, 24, 58, 16
    rows_spec = [("o6", "mobile phone 1"), ("o21", "can 1")]
    f_title, f_sub, f_label, f_row = _font(15, True), _font(13), _font(9), _font(11)
    title = "The same physical object, eight times"
    sub = "Labels never change. Background, lighting, pose and occlusion do."

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_label_w = 96
    grid_w = 8 * tile + 7 * gap
    width = int(max(grid_w + row_label_w, _tw(probe, sub, f_sub))) + 2 * margin
    height = header + len(rows_spec) * (tile + gap + label_h) + 14

    im = Image.new("RGB", (width, height), theme.surface)
    dr = ImageDraw.Draw(im)
    dr.text((margin, 14), title, font=f_title, fill=theme.ink)
    dr.text((margin, 35), sub, font=f_sub, fill=theme.secondary)
    for r, (obj, obj_name) in enumerate(rows_spec):
        y = header + r * (tile + gap + label_h)
        dr.text((margin, y + tile // 2 - 6), obj_name, font=f_row, fill=theme.secondary)
        for d, session in enumerate(DOMAIN_SESSIONS):
            x = margin + row_label_w + d * (tile + gap)
            im.paste(_frame(session, obj, tile), (x, y))
            dr.rectangle([x, y, x + tile - 1, y + tile - 1], outline=theme.gridline)
            if r == len(rows_spec) - 1:
                dr.text((x, y + tile + 4), f"domain {d} · {session}", font=f_label, fill=theme.muted)

    out = _out("02-same-object", theme, "png")
    im.save(out, optimize=True)
    return out


# ------------------------------------------------------- the evaluator's inputs

def _mono(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans Mono")), size)


# Real counts for CORe50 NI domain 2 at TOTAL_N_CLIENTS=4, from client/data.py.
PIPE = {
    "session": "s2",
    "domain": 2,
    "total": 14986,
    "train": 11988,
    "test": 2998,
    "clients": 4,
    "train_slice": 2997,
    "test_slice": 749,
}


def asset_03_domain_pipeline(theme: Theme) -> str:
    """How one domain becomes an evaluator input.

    The point of the figure: a domain splits into a *train* slice the round
    trains on and a *test* slice that is handed to the evaluator once and kept
    for the rest of the run. Both are deterministic functions of the domain
    index and the client number, so no client has to coordinate with any other.
    """
    f_title, f_sub, f_box, f_small, f_num = _font(15, True), _font(12), _font(11, True), _font(9), _font(13, True)
    f_code = _mono(9)

    W, H = 800, 262
    margin = 24
    im = Image.new("RGB", (W, H), theme.surface)
    dr = ImageDraw.Draw(im)
    dr.text((margin, 14), "How one domain becomes an evaluator input", font=f_title, fill=theme.ink)
    dr.text((margin, 36), "deterministic from (domain index, client number) alone \u2014 clients never coordinate",
            font=f_sub, fill=theme.secondary)

    def stage(x, y, w, h, title, lines, accent=None, fill=None):
        dr.rectangle([x, y, x + w, y + h], fill=fill or theme.surface,
                     outline=accent or theme.baseline, width=2 if accent else 1)
        dr.text((x + 10, y + 8), title, font=f_box, fill=theme.ink)
        yy = y + 26
        for ln, col in lines:
            dr.text((x + 10, yy), ln, font=f_small, fill=col)
            yy += 13

    def arrow(x0, y0, x1, y1, label=None):
        dr.line([x0, y0, x1, y1], fill=theme.baseline, width=2)
        dr.polygon([(x1, y1), (x1 - 7, y1 - 4), (x1 - 7, y1 + 4)], fill=theme.baseline)
        if label:
            dr.text((x0 + 4, y0 - 15), label, font=f_small, fill=theme.muted)

    # stage 1 - the raw domain
    y0 = 74
    stage(margin, y0, 176, 82, f"session {PIPE['session']}",
          [(f"domain {PIPE['domain']} of the stream", theme.muted),
           (f"{PIPE['total']:,} images, 10 classes", theme.secondary)])
    arrow(margin + 176 + 6, y0 + 41, margin + 176 + 46, y0 + 41, "80/20")

    # stage 2 - the split
    x2 = margin + 176 + 52
    stage(x2, y0, 150, 38, "train split", [], fill=theme.surface)
    dr.text((x2 + 96, y0 + 11), f"{PIPE['train']:,}", font=f_box, fill=theme.secondary)
    stage(x2, y0 + 44, 150, 38, "test split", [], fill=theme.surface)
    dr.text((x2 + 96, y0 + 55), f"{PIPE['test']:,}", font=f_box, fill=theme.secondary)
    arrow(x2 + 156, y0 + 41, x2 + 196, y0 + 41, f"\u00f7 {PIPE['clients']}")

    # stage 3 - this client's two slices, each labelled with where it goes
    x3 = x2 + 202
    sw = W - margin - x3
    for i, (name, count, col, dest) in enumerate([
        ("client 1 train slice", PIPE["train_slice"], theme.accent, "trained on this round, then gone"),
        ("client 1 test slice", PIPE["test_slice"], theme.orange, "handed to the evaluator, kept for good"),
    ]):
        y = y0 + i * 44
        dr.rectangle([x3, y, x3 + sw, y + 38], outline=col, width=2)
        dr.text((x3 + 10, y + 6), name, font=f_box, fill=theme.ink)
        cnt = f"{count:,}"
        dr.text((x3 + sw - 12 - dr.textlength(cnt, font=f_box), y + 6), cnt, font=f_box, fill=col)
        dr.text((x3 + 10, y + 22), dest, font=f_small, fill=col)

    # the two calls that produce those slices
    yc = H - 76
    dr.rectangle([margin, yc, W - margin, H - 16], outline=theme.gridline)
    dr.text((margin + 10, yc + 9), "client/data.py", font=f_small, fill=theme.muted)
    dr.text((margin + 10, yc + 26), 'x_train, y_train = load_domain(2, 1, 4, is_train=True)    \u2192 train on it',
            font=f_code, fill=theme.accent)
    dr.text((margin + 10, yc + 42), 'test_slice       = load_domain(2, 1, 4, is_train=False)   \u2192 ev.add_domain(test_slice)',
            font=f_code, fill=theme.orange)

    out = _out("03-domain-pipeline", theme, "png")
    im.save(out, optimize=True)
    return out


def asset_04_evaluator_lifecycle(theme: Theme) -> str:
    """How domains accumulate inside the evaluator, round by round.

    Three rounds, three steps each: register the round's domain (capturing the
    pre-training probe Forward Transfer is computed from), train on it, then
    score the model against every domain registered so far. The three panels
    are the evaluator's actual state -- its domain list, its off-history
    pre-train side channel, and its history matrix.
    """
    run = Run()
    rounds = run.rounds[:3]
    doms = [run.trained[r] for r in rounds]

    f_title, f_sub, f_box, f_small = _font(15, True), _font(12), _font(11, True), _font(9)
    f_code, f_code_s = _mono(10), _mono(9)

    W, H = 800, 436
    margin = 24
    code_y, state_y, emit_y = 62, 168, 352

    def render(k: int, step: int) -> Image.Image:
        """k = round index 0..2, step = 1 register / 2 train / 3 evaluate."""
        im = Image.new("RGB", (W, H), theme.surface)
        dr = ImageDraw.Draw(im)
        label = f"domain{doms[k]}"
        dr.text((margin, 14), "How a domain enters the evaluator", font=f_title, fill=theme.ink)
        caps = {
            1: f"\u2460  train() registers {label} and probes the incoming model \u2014 that probe is Forward Transfer",
            2: f"\u2461  train() fits the model on {label}'s train slice only \u2014 no replay of earlier domains",
            3: "\u2462  validate() scores that model on every domain registered so far",
        }
        dr.text((margin, 36), caps[step], font=f_sub,
                fill=theme.accent if step != 2 else theme.secondary)
        rl = f"round {k + 1}"
        dr.text((W - margin - dr.textlength(rl, font=f_box), 36), rl, font=f_box, fill=theme.muted)

        # ---- the three calls, as code
        dr.rectangle([margin, code_y, W - margin, code_y + 92], outline=theme.gridline)
        lines = [
            (1, f'ev.add_domain(test_slice, label="{label}", pre_evaluate_with=model)'),
            (2, "model = train(model, x_train, y_train)"),
            (3, "metrics = ev.evaluate(model)"),
        ]
        for i, (st, code) in enumerate(lines):
            y = code_y + 12 + i * 24
            active = st == step
            if active:
                dr.rectangle([margin + 6, y - 5, W - margin - 6, y + 17], fill=theme.gridline)
            dr.text((margin + 14, y), code, font=f_code,
                    fill=theme.ink if active else theme.muted)
            if st == 1:
                dr.text((margin + 14 + dr.textlength(code, font=f_code) + 10, y + 1),
                        "first time this domain appears, only", font=f_small, fill=theme.muted)

        # ---- evaluator state: three panels
        pw, gap = (W - 2 * margin - 2 * 14) // 3, 14
        panels = [
            ("domains[]", "registration order"),
            ("pretrain_value{}", "off-history \u00b7 feeds FWT"),
            ("history[][]", "one row per evaluate()"),
        ]
        for i, (name, sub) in enumerate(panels):
            x = margin + i * (pw + gap)
            dr.rectangle([x, state_y, x + pw, state_y + 168], outline=theme.gridline)
            dr.text((x + 10, state_y + 8), name, font=f_code, fill=theme.ink)
            dr.text((x + 10, state_y + 26), sub, font=f_small, fill=theme.muted)

        registered = k + 1 if step >= 1 else k
        # panel 1: domain chips
        x = margin
        for j in range(registered):
            yy = state_y + 46 + j * 26
            fresh = j == k and step == 1
            dr.rounded_rectangle([x + 10, yy, x + pw - 10, yy + 20], radius=5,
                                 fill=theme.ordinal[j],
                                 outline=theme.accent if fresh else None,
                                 width=2 if fresh else 0)
            txt = f"domain{doms[j]}"
            dr.text((x + 18, yy + 4), txt, font=f_small, fill="#ffffff")

        # panel 2: the pre-train probe, one entry per domain
        x = margin + pw + gap
        for j in range(registered):
            yy = state_y + 46 + j * 26
            fresh = j == k and step == 1
            dr.text((x + 10, yy + 4), f"domain{doms[j]}", font=f_small,
                    fill=theme.ink if fresh else theme.secondary)
            dr.text((x + 84, yy + 4), "probed", font=f_small,
                    fill=theme.accent if fresh else theme.muted)
        fwt = run.agg(rounds[k], "fwt_accuracy")
        if step == 3 and fwt is not None:
            dr.text((x + 10, state_y + 140), f"fwt_accuracy  {fwt:.3f}", font=f_small, fill=theme.orange)

        # panel 3: the history matrix
        x = margin + 2 * (pw + gap)
        cw, ch = 40, 20
        for j in range(registered):
            dr.text((x + 14 + j * (cw + 3), state_y + 44), f"d{doms[j]}", font=f_small, fill=theme.secondary)
        n_rows = k + 1 if step == 3 else k
        for r_i in range(k + 1):
            yy = state_y + 60 + r_i * (ch + 3)
            dr.text((x + 10 - 2, yy + 5), f"{r_i + 1}", font=f_small, fill=theme.muted)
            for j in range(registered):
                cx = x + 14 + j * (cw + 3)
                if r_i < n_rows and j <= r_i:
                    v = run.acc(rounds[r_i], doms[j])
                    if v is None:
                        continue
                    dr.rectangle([cx, yy, cx + cw, yy + ch], fill=theme.seq_color(v))
                    t = f"{v:.2f}".lstrip("0")
                    dr.text((cx + (cw - dr.textlength(t, font=f_small)) / 2, yy + 5), t,
                            font=f_small, fill=theme.cell_ink(v))
                else:
                    dr.rectangle([cx, yy, cx + cw, yy + ch], outline=theme.gridline)

        # ---- what the round emits
        dr.rectangle([margin, emit_y, W - margin, H - 18], outline=theme.gridline)
        dr.text((margin + 10, emit_y + 8), "emitted to client.log_metric / the dashboard",
                font=f_box, fill=theme.ink)
        if step == 3:
            keys = [f"domain{doms[j]}_accuracy" for j in range(registered)]
            keys += ["avg_accuracy", "plasticity_accuracy"]
            if fwt is not None:
                keys.append("fwt_accuracy")
            if k >= 1:
                keys += ["forgetting_accuracy", "bwt_accuracy"]
            dr.text((margin + 10, emit_y + 28), "  ".join(keys), font=f_code_s, fill=theme.orange)
        else:
            dr.text((margin + 10, emit_y + 28), "\u2014 nothing yet; emission happens on evaluate()",
                    font=f_code_s, fill=theme.muted)
        return im

    frames, durations = [], []
    for k in range(3):
        for step in (1, 2, 3):
            frames.append(render(k, step))
            durations.append(1500 if step == 3 else 1150)
    durations[-1] = 2200
    return _save_gif(frames, _out("04-evaluator-lifecycle", theme, "gif"), durations)


# ----------------------------------------------------------------- run metrics


class Run:
    """Per-round, fleet-mean view of one session's continual-learning metrics.

    Reads the compact export written by ``tools/export_run.py``: per-round means
    already reduced across clients, one line per round.
    """

    def __init__(self, path: str = RUN_JSON):
        if not os.path.exists(path):
            sys.exit(f"Run metrics not found at {path}\nExport them with tools/export_run.py")
        with open(path) as f:
            raw = json.load(f)

        self.rounds: List[int] = raw["rounds"]
        self.n_clients: int = raw["n_clients"]
        self.domain_sessions = raw["domain_sessions"]
        self.mean: Dict[int, Dict[str, float]] = {int(r): m for r, m in raw["mean"].items()}
        self.trained: Dict[int, int] = {int(r): d for r, d in raw["trained"].items()}

        # Columns are ordered by when the stream introduced each domain, not by
        # domain index: the round counter is deployment-global, so a session
        # usually starts mid-cycle. Visit order makes the "just trained" cells
        # an actual diagonal.
        self.visit_order: List[int] = []
        for r in self.rounds:
            d = self.trained.get(r)
            if d is not None and d not in self.visit_order:
                self.visit_order.append(d)

    def acc(self, rnd: int, domain: int) -> Optional[float]:
        return self.mean[rnd].get(f"domain{domain}_accuracy")

    def agg(self, rnd: int, name: str) -> Optional[float]:
        return self.mean[rnd].get(name)


def asset_05_accuracy_matrix(theme: Theme) -> str:
    """The canonical CL figure: rounds x domains, filling in one row per round."""
    run = Run()
    cols = run.visit_order
    n_col, n_row = len(cols), len(run.rounds)
    cw, ch, gap = 58, 26, 2
    left, top, margin, footer = 132, 100, 24, 58
    f_title, f_sub, f_hdr, f_cell, f_note = _font(15, True), _font(12), _font(10, True), _font(9), _font(10)

    # Wide enough for the grid *and* the header text: with few domains the
    # subtitle is wider than the grid and would otherwise be clipped.
    title = "Accuracy on every domain, every round"
    subtitle = f"mean over {run.n_clients} clients · each scores its own slice · deployment rounds {run.rounds[0]}-{run.rounds[-1]}"
    _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    header_w = max(_tw(_probe, title, f_title), _tw(_probe, subtitle, f_sub))
    width = max(left + n_col * (cw + gap) + 96, int(header_w)) + margin
    height = top + n_row * (ch + gap) + footer

    def render(upto: int) -> Image.Image:
        im = Image.new("RGB", (width, height), theme.surface)
        dr = ImageDraw.Draw(im)
        dr.text((margin, 14), title, font=f_title, fill=theme.ink)
        dr.text((margin, 36), subtitle, font=f_sub, fill=theme.secondary)
        dr.text((margin, 58), "domain, in the order the stream introduced it →", font=f_note, fill=theme.muted)
        for c, d in enumerate(cols):
            x = left + c * (cw + gap)
            lbl = f"d{d}"
            dr.text((x + (cw - _tw(dr, lbl, f_hdr)) / 2, top - 18), lbl, font=f_hdr, fill=theme.secondary)

        for r_i, rnd in enumerate(run.rounds):
            y = top + r_i * (ch + gap)
            shown = r_i < upto
            # Session-relative round, matching how the dashboard labels its x-axis.
            dr.text((margin, y + 7), f"round {r_i + 1}", font=f_cell, fill=theme.ink if shown else theme.gridline)
            if shown and run.trained.get(rnd) is not None:
                dr.text((margin + 62, y + 7), f"trains d{run.trained[rnd]}", font=f_cell, fill=theme.muted)
            for c, d in enumerate(cols):
                x = left + c * (cw + gap)
                val = run.acc(rnd, d) if shown else None
                if val is None:
                    dr.rectangle([x, y, x + cw, y + ch], outline=theme.gridline)
                    continue
                dr.rectangle([x, y, x + cw, y + ch], fill=theme.seq_color(val))
                txt = f"{val:.2f}".lstrip("0")
                dr.text((x + (cw - _tw(dr, txt, f_cell)) / 2, y + 8), txt, font=f_cell, fill=theme.cell_ink(val))
                if run.trained.get(rnd) == d:
                    dr.rectangle([x, y, x + cw, y + ch], outline=theme.orange, width=2)
            if shown:
                a = run.agg(rnd, "avg_accuracy")
                if a is not None:
                    dr.text((left + n_col * (cw + gap) + 14, y + 7), f"avg {a:.2f}", font=f_cell, fill=theme.secondary)

        dr.text((margin, height - 42), "orange outline = the domain trained that round", font=f_note, fill=theme.orange)
        dr.text((margin, height - 26), "bright down the diagonal = plasticity · fading to its left = forgetting", font=f_note, fill=theme.secondary)
        return im

    frames = [render(i + 1) for i in range(n_row)]
    frames += [render(n_row)] * 2
    durations = [420] * n_row + [900, 1600]
    return _save_gif(frames, _out("05-accuracy-matrix", theme, "gif"), durations)


def asset_06_drift_curves(theme: Theme) -> str:
    """Drift in action: the fleet average, the spread behind it, and what it costs.

    Two stacked panels, each with a single y-axis -- accuracies above, accuracy
    *deltas* below. Never one chart carrying two scales.

    Deliberately NOT one line per domain: with eight domains each spiking on
    the round it is retrained, that chart is unreadable spaghetti. Per-domain
    detail is asset 04's job; this one shows the level, the spread, and the
    second pass lifting both.
    """
    run = Run()
    rounds, cols = run.rounds, run.visit_order
    xs = list(range(1, len(rounds) + 1))
    n_dom = len(cols)
    # Where the stream wraps back to its first domain, i.e. the second pass.
    pass2 = next((i + 1 for i, r in enumerate(rounds)
                  if i > 0 and run.trained.get(r) == cols[0]), None)

    def series(upto: int):
        xx, lo, hi, avg = [], [], [], []
        for i, rnd in enumerate(rounds[:upto]):
            vals = [run.acc(rnd, d) for d in cols if run.acc(rnd, d) is not None]
            if not vals:
                continue
            xx.append(xs[i])
            lo.append(min(vals))
            hi.append(max(vals))
            a = run.agg(rnd, "avg_accuracy")
            avg.append(a if a is not None else sum(vals) / len(vals))
        return xx, lo, hi, avg

    def render(upto: int) -> Image.Image:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7.2, 5.2), dpi=100, sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.18},
        )
        fig.subplots_adjust(left=0.095, right=0.955, top=0.90, bottom=0.115)
        fig.patch.set_facecolor(theme.surface)
        for ax in (ax1, ax2):
            ax.set_facecolor(theme.surface)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(theme.baseline)
            ax.tick_params(colors=theme.muted, labelsize=8, length=3)
            ax.grid(True, axis="y", color=theme.gridline, linewidth=0.8)
            ax.set_axisbelow(True)
            if pass2 is not None and upto >= pass2:
                ax.axvline(pass2, color=theme.baseline, linewidth=1, linestyle=(0, (4, 3)), zorder=1)

        xx, lo, hi, avg = series(upto)
        band = theme.ordinal[3]
        if xx:
            ax1.fill_between(xx, lo, hi, color=band, alpha=0.22, linewidth=0, zorder=2)
            ax1.plot(xx, hi, color=band, linewidth=1.5, zorder=3)
            ax1.plot(xx, lo, color=band, linewidth=1.5, zorder=3)
            ax1.plot(xx, avg, color=theme.orange, linewidth=3, zorder=5, solid_capstyle="round")
            ax1.plot(xx[-1:], avg[-1:], "o", color=theme.orange, markersize=6,
                     markeredgecolor=theme.surface, markeredgewidth=2, zorder=6)
            ax1.annotate(f"{avg[-1]:.2f}", (xx[-1], avg[-1]), textcoords="offset points",
                         xytext=(9, -3), color=theme.orange, fontsize=9, fontweight="bold",
                         zorder=7, annotation_clip=False)

        if pass2 is not None and upto >= pass2:
            ax1.annotate("second pass begins", (pass2, 0.30), textcoords="offset points",
                         xytext=(6, 0), color=theme.secondary, fontsize=8, zorder=7)

        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("accuracy", color=theme.secondary, fontsize=9)
        ax1.set_title("One average hides a 0.5-wide spread across domains",
                      color=theme.ink, fontsize=13, fontweight="bold", loc="left", pad=10)
        ax1.legend(
            handles=[Line2D([], [], color=theme.orange, lw=3, label="average over domains seen"),
                     Patch(facecolor=band, alpha=0.32, edgecolor=band,
                           label=f"best-to-worst range across the {n_dom} domains")],
            loc="lower left", fontsize=8, frameon=False, labelcolor=theme.secondary,
            handlelength=1.6, borderpad=0,
        )

        for key, color, label in (("forgetting_accuracy", theme.orange, "forgetting"),
                                  ("bwt_accuracy", theme.accent, "backward transfer")):
            bx = [xs[i] for i, r in enumerate(rounds[:upto]) if run.agg(r, key) is not None]
            by = [run.agg(r, key) for r in rounds[:upto] if run.agg(r, key) is not None]
            if bx:
                ax2.plot(bx, by, color=color, linewidth=2, zorder=3, label=label, solid_capstyle="round")
                ax2.plot(bx[-1:], by[-1:], "o", color=color, markersize=5,
                         markeredgecolor=theme.surface, markeredgewidth=2, zorder=4)
        ax2.axhline(0, color=theme.baseline, linewidth=1, zorder=2)
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_ylabel("accuracy delta", color=theme.secondary, fontsize=9)
        ax2.set_xlabel("round of this session", color=theme.secondary, fontsize=9)
        ax2.set_xticks(xs)
        ax2.set_xlim(0.6, len(rounds) + 0.4)
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=theme.secondary,
                       ncol=2, handlelength=1.6, borderpad=0)

        fig.text(0.975, 0.018, f"mean over {run.n_clients} clients", ha="right",
                 color=theme.muted, fontsize=8)
        fig.canvas.draw()
        im = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                             bytes(fig.canvas.buffer_rgba())).convert("RGB")
        plt.close(fig)
        return im

    frames = [render(i + 1) for i in range(len(rounds))]
    frames += [frames[-1]]
    durations = [430] * len(rounds) + [1600]
    return _save_gif(frames, _out("06-drift-curves", theme, "gif"), durations)


# ------------------------------------------------------- dashboard screenshots

SHOTS = os.path.join(ASSETS, "screenshots")

# Hand-captured from the Scaleout Edge UI (light theme). Cropped here rather
# than by hand so the framing is reproducible and the raw captures stay
# untouched under assets/screenshots/. Boxes are (left, top, right, bottom)
# against the 1854x966-ish full-window captures; the left ~437px is the app
# sidebar, which carries no information for the README.
SHOT_CROPS = {
    "07-dashboard-metrics": ("06-dashboard-full.png", (437, 6, 1722, 896)),
    "07-dashboard-bwt": ("06-dashboard-bwt.png", (437, 224, 1718, 952)),
}


def asset_07_dashboard(theme: Theme) -> List[str]:
    """Crop the UI captures down to the panels the README actually points at.

    Single-variant: these are photographs of a light-theme product, so there is
    no honest way to synthesize a dark twin. The README uses a plain <img> for
    them and mats them on the light surface in both themes.
    """
    if theme.name != "light":
        return []
    out_paths = []
    for name, (src, box) in SHOT_CROPS.items():
        src_path = os.path.join(SHOTS, src)
        if not os.path.exists(src_path):
            sys.exit(f"Screenshot not found: {src_path}")
        im = Image.open(src_path).convert("RGB").crop(box)
        # Hairline border so the panel reads as a framed screenshot rather than
        # bleeding into the page on a white background.
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, im.width - 1, im.height - 1], outline=LIGHT.gridline)
        out = os.path.join(ASSETS, f"{name}.png")
        im.save(out, optimize=True)
        out_paths.append(out)
    return out_paths


ASSET_FNS = {
    "01": asset_01_domain_stream,
    "02": asset_02_same_object,
    "03": asset_03_domain_pipeline,
    "04": asset_04_evaluator_lifecycle,
    "05": asset_05_accuracy_matrix,
    "06": asset_06_drift_curves,
    "07": asset_07_dashboard,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="Asset number to build, e.g. 04")
    ap.add_argument("--theme", choices=sorted(THEMES), help="Build one theme only")
    args = ap.parse_args()

    if args.only and args.only not in ASSET_FNS:
        sys.exit(f"Unknown asset {args.only!r}; known: {', '.join(sorted(ASSET_FNS))}")
    todo = ASSET_FNS if args.only is None else {args.only: ASSET_FNS[args.only]}
    themes = list(THEMES.values()) if args.theme is None else [THEMES[args.theme]]

    total = 0
    for key, fn in todo.items():
        for theme in themes:
            result = fn(theme)
            for path in (result if isinstance(result, list) else [result]):
                size = os.path.getsize(path)
                total += size
                print(f"[{key}] {os.path.relpath(path, PROJECT):44s} {size // 1024:5d} KB")
    print(f"total {total / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
