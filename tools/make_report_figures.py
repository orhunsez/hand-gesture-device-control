"""Figures for the report, section 4.1 (system overview).

    .venv\\Scripts\\python.exe make_report_figures.py

Writes two PNG files to docs/figures/, both 6.5 inches wide so that they fill the text width of the report
without being scaled down (all text is 7.5 pt or larger at that size):
  fig_pipeline.png  the processing pipeline as a vertical flow, with the data shape and one measured number per
                    stage, and the status of every stage (implemented, pilot, planned)
  fig_stages.png    real data from the recorded clips: (a) raw landmarks of the same gesture at the smallest and
                    the largest hand size, (b) the same two hands after normalisation, (c) the speed noise of a
                    still hand before and after the 1-euro filter

The numbers in fig_pipeline.png are measured on the pilot clips (one hand, 41 clips, 1653 frames) and are
provisional; update them here when the data changes.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

import landmark_features as lf
import landmark_smoothing as ls
import view_clip

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 300

BLUE, BLUE_FILL = "#2F5D8C", "#DCE6F2"
ORANGE, ORANGE_FILL = "#B5651D", "#FBE6CF"
GRAY = "#7F7F7F"
STYLE = {
    "done": dict(fc=BLUE_FILL, ec=BLUE, ls="-"),
    "pilot": dict(fc=ORANGE_FILL, ec=ORANGE, ls="-"),
    "plan": dict(fc="white", ec=GRAY, ls=(0, (4, 3))),
}
TITLE_PT, NOTE_PT = 9.5, 8.3


def box(ax, cx, yc, w, h, title, notes, status):
    st = STYLE[status]
    ax.add_patch(FancyBboxPatch((cx - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=st["fc"], ec=st["ec"], ls=st["ls"], lw=1.2))
    n = len(notes)
    block = 0.20 + 0.155 * n
    top = yc + block / 2
    ax.text(cx, top - 0.10, title, ha="center", va="center", fontsize=TITLE_PT, fontweight="bold")
    for i, line in enumerate(notes):
        ax.text(cx, top - 0.26 - 0.155 * i, line, ha="center", va="center", fontsize=NOTE_PT, color="#222222")


def arrow(ax, p, q, dashed=False):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=GRAY if dashed else "#222222",
                                ls="--" if dashed else "-", shrinkA=0, shrinkB=0, mutation_scale=9))


def pipeline():
    fig = plt.figure(figsize=(6.5, 7.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 6.5); ax.set_ylim(-0.55, 6.8); ax.axis("off")
    cx, W = 2.4, 4.2
    ax.text(0.1, 6.65, "One pass per camera frame (51 ms). Time after MediaPipe: 0.20 ms in total.",
            fontsize=NOTE_PT, style="italic", ha="left", va="center")
    rows = [
        (6.15, 0.66, "Camera frame", ["640 x 480 RGB, 19.5 frames per second (51 ms)"], "done"),
        (5.28, 0.66, "MediaPipe landmarks", ["21 points x (x, y, z); hand found in 1653 of 1653 frames"], "done"),
        (4.41, 0.66, "Smoothing (1 euro filter)", ["jitter 84 % lower, delay 27 ms, 16 µs per frame"], "done"),
        (3.54, 0.66, "Normalisation", ["63 features; hand size 69 to 150 px becomes 1; 13 µs"], "done"),
    ]
    for yc, h, t, notes, st in rows:
        box(ax, cx, yc, W, h, t, notes, st)
    for (y1, h1, *_), (y2, h2, *_) in zip(rows[:-1], rows[1:]):
        arrow(ax, (cx, y1 - h1 / 2 - 0.01), (cx, y2 + h2 / 2 + 0.01))
    # calibration, a start-up step beside the normalisation
    box(ax, 5.65, 3.54, 1.5, 0.9, "Calibration", ["once at start-up:", "hand scale and", "thresholds"], "plan")
    arrow(ax, (4.9, 3.54), (4.5 + 0.01, 3.54), dashed=True)
    # fork
    fy, fh, fw = 2.38, 1.12, 2.05
    sx, dx = cx - fw / 2 - 0.05, cx + fw / 2 + 0.05
    box(ax, sx, fy, fw, fh, "Static branch", ["SVM on 63 features,", "hold time; first pilot", "on one hand; 169 µs"], "pilot")
    box(ax, dx, fy, fw, fh, "Dynamic branch", ["palm-centre speed,", "window of 10 to 20", "frames, window class"], "plan")
    arrow(ax, (cx - 0.3, 3.54 - 0.33 - 0.01), (sx, fy + fh / 2 + 0.01))
    arrow(ax, (cx + 0.3, 3.54 - 0.33 - 0.01), (dx, fy + fh / 2 + 0.01))
    # decision, command
    ty, th = 1.1, 0.82
    box(ax, cx, ty, W, th, "False-trigger prevention", ["confidence threshold, hold time,", "cooldown, no-gesture class"], "plan")
    arrow(ax, (sx, fy - fh / 2 - 0.01), (cx - 0.4, ty + th / 2 + 0.01))
    arrow(ax, (dx, fy - fh / 2 - 0.01), (cx + 0.4, ty + th / 2 + 0.01))
    cy, chh = 0.3, 0.66
    box(ax, cx, cy, W, chh, "Command mapping", ["postponed: target device not decided yet"], "plan")
    arrow(ax, (cx, ty - th / 2 - 0.01), (cx, cy + chh / 2 + 0.01))
    ax.legend(handles=[Patch(fc=BLUE_FILL, ec=BLUE, label="implemented and tested"),
                       Patch(fc=ORANGE_FILL, ec=ORANGE, label="first pilot (one hand)"),
                       Patch(fc="white", ec=GRAY, ls="--", label="planned")],
              loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False, fontsize=8.2,
              handlelength=1.6, columnspacing=1.3)
    fig.savefig(OUT / "fig_pipeline_page.png", dpi=DPI)
    plt.close(fig)


def skeleton(ax, pts, color, label=None, lw=1.2):
    for a, b in view_clip.CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=color, lw=lw, solid_capstyle="round")
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, zorder=3, label=label)


def stages():
    cands = []
    for f in ("u01/open_palm.json", "u99/open_palm.json"):
        for k in range(1, len(view_clip.read_file(f)["clips"]) + 1):
            lm, t, meta = view_clip.load_clip(f, k)
            lm = lm.astype(float)
            cands.append((float(np.nanmedian(lf.hand_size(lm, 640, 480))), f, k, lm))
    small = min(cands, key=lambda c: c[0])
    large = max(cands, key=lambda c: c[0])

    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 8.2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8})
    fig = plt.figure(figsize=(6.5, 5.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1], width_ratios=[1.1, 0.9], hspace=0.5, wspace=0.32)
    cs = ["#C0392B", BLUE]
    # (a) raw, in image pixels
    ax = fig.add_subplot(gs[0, 0])
    for (size, f, k, lm), c in zip((small, large), cs):
        px = lf.to_pixel_units(lm, 640, 480)[len(lm) // 2]
        skeleton(ax, px[:, :2], c, label=f"hand size {size:.0f} px")
    ax.legend(loc="upper left", fontsize=7.8, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlim(0, 640); ax.set_ylim(480, 0); ax.set_aspect("equal")
    ax.set_xticks([0, 200, 400, 600]); ax.set_yticks([0, 200, 400])
    ax.set_title("(a) Raw landmarks, image pixels"); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    # (b) normalised
    ax = fig.add_subplot(gs[0, 1])
    for (size, f, k, lm), c in zip((small, large), cs):
        q = lf.normalize(lm, 640, 480)[len(lm) // 2]
        skeleton(ax, q[:, :2], c)
    ax.scatter([0], [0], s=40, marker="+", color="black", zorder=4)
    ax.annotate("wrist = origin", xy=(0, 0), xytext=(-1.45, 0.62), fontsize=7.8, arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(0.9, -2.3); ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-2, -1, 0])
    ax.set_title("(b) After normalisation"); ax.set_xlabel("x / hand size"); ax.set_ylabel("y / hand size")
    ax.grid(alpha=0.25)
    # (c) speed noise of a still hand
    ax = fig.add_subplot(gs[1, :])
    lm, t, meta = view_clip.load_clip("u99/open_palm.json", 7)
    lm = lm.astype(float); t = t.astype(float)
    PALM = [0, 5, 9, 13, 17]

    def speed(l):
        px = lf.to_pixel_units(l, 640, 480)
        s = np.nanmedian(lf.hand_size(l, 640, 480))
        c = np.nanmean(px[:, PALM, :2], axis=1)
        return np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(t) / s
    raw = speed(lm); sm = speed(ls.smooth_clip(lm, t, 640, 480))
    ax.plot(t[1:], raw, color="#C0392B", lw=1.1, label=f"raw (95th percentile {np.percentile(raw, 95):.3f})")
    ax.plot(t[1:], sm, color=BLUE, lw=1.4, label=f"smoothed (95th percentile {np.percentile(sm, 95):.3f})")
    ax.set_title("(c) Speed of a still hand before and after the 1 euro filter")
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (hand sizes/s)")
    ax.set_ylim(0, 0.15); ax.set_xlim(0, 2.05); ax.grid(alpha=0.25); ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    fig.savefig(OUT / "fig_stages_page.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"(a)/(b): smallest open palm hand {small[0]:.0f} px ({small[1]} clip {small[2]}), "
          f"largest {large[0]:.0f} px ({large[1]} clip {large[2]})")
    print(f"(c): speed noise 95th percentile raw {np.percentile(raw, 95):.3f}, smoothed {np.percentile(sm, 95):.3f}")


if __name__ == "__main__":
    pipeline()
    stages()
    from PIL import Image
    for p in sorted(OUT.glob("fig_*.png")):
        with Image.open(p) as im:
            print(p.name, im.size, f"= {im.size[0] / DPI:.2f} x {im.size[1] / DPI:.2f} in at {DPI} dpi")
