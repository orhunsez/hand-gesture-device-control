"""Equation (3) and the swipe speed figure for section 4.5 -> docs/figures/eq_speed.png, fig_swipe_speed.png"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import landmark_features as lf, landmark_smoothing as ls, view_clip
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
PALM = [0, 5, 9, 13, 17]

fig = plt.figure(figsize=(6.5, 1.2))
fig.text(0.5, 0.72, r"$\mathbf{c}_t = \dfrac{1}{5}\sum_{i \in \{0,5,9,13,17\}} \mathbf{p}_{t,i}$", ha="center", va="center", fontsize=14)
fig.text(0.5, 0.25, r"$v_t = \dfrac{\Vert \mathbf{c}_t - \mathbf{c}_{t-1} \Vert}{s\,\Delta t}$", ha="center", va="center", fontsize=14)
fig.text(0.985, 0.5, "(3)", ha="right", va="center", fontsize=13)
fig.savefig(OUT / "eq_speed.png", dpi=300); plt.close(fig)

def speed(f, k):
    lm, t, meta = view_clip.load_clip(f, k)
    lm = ls.smooth_clip(lm.astype(float), t.astype(float), 640, 480); t = t.astype(float)
    px = lf.to_pixel_units(lm, 640, 480); s = np.nanmedian(lf.hand_size(lm, 640, 480))
    c = np.nanmean(px[:, PALM, :2], axis=1)
    return t[1:], np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(t) / s
fig, ax = plt.subplots(figsize=(6.5, 2.4))
for (f, k), col, lab in ((("u99/swipe_left.json", 1), "#2F5D8C", "swipe left"), (("u99/swipe_right.json", 1), "#B5651D", "swipe right")):
    t, v = speed(f, k); ax.plot(t, v, color=col, lw=1.4, label=lab)
ax.set_xlabel("time (s)", fontsize=8.5); ax.set_ylabel("palm-centre speed\n(hand sizes/s)", fontsize=8.5)
ax.tick_params(labelsize=8); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_swipe_speed.png", dpi=300); plt.close(fig)
print("ok")
