"""Checks for landmark_smoothing.py. Run:  .venv\\Scripts\\python.exe test_landmark_smoothing.py"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import numpy as np

import landmark_smoothing as ls

rng = np.random.default_rng(1)
FPS = 20.0
DT = 1.0 / FPS


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


def run(f, values, dt=DT):
    return np.array([f(v, i * dt) for i, v in enumerate(values)])


# 1. The first sample is returned unchanged.
f = ls.OneEuroFilter(1.0, 0.0)
check("first sample passes through", np.isclose(f(5.0, 0.0), 5.0))

# 2. A constant signal stays constant.
out = run(ls.OneEuroFilter(1.0, 0.0), np.full(50, 3.0))
check("constant signal stays constant", np.allclose(out, 3.0))

# 3. A step is followed without overshoot and converges.
step = np.concatenate([np.zeros(10), np.ones(60)])
out = run(ls.OneEuroFilter(2.0, 0.0), step)
check("step response never overshoots", out.max() <= 1.0 + 1e-12)
check("step response converges", abs(out[-1] - 1.0) < 1e-3)
check("step response lags behind the input", out[10] < 1.0)

# 4. Jitter: the smoothed noise around a still value is smaller, and a lower f_cmin smooths more.
noise = 1.0 + 0.05 * rng.standard_normal(400)
out_hi = run(ls.OneEuroFilter(5.0, 0.0), noise)
out_lo = run(ls.OneEuroFilter(0.5, 0.0), noise)
check("smoothing reduces jitter", out_lo[50:].std() < noise[50:].std())
check("lower f_cmin smooths more", out_lo[50:].std() < out_hi[50:].std())

# 5. beta: fast motion is followed more closely with a larger beta (smaller lag).
t = np.arange(120) * DT
fast = np.sin(2 * np.pi * 1.0 * t) * 100.0           # a 1 Hz movement, 100 units amplitude
err_b0 = np.abs(run(ls.OneEuroFilter(0.5, 0.0), fast) - fast)[20:].mean()
err_b1 = np.abs(run(ls.OneEuroFilter(0.5, 0.05), fast) - fast)[20:].mean()
check("larger beta follows fast motion more closely", err_b1 < err_b0)

# 6. Timestamps: duplicates and irregular steps do not break the filter.
f = ls.OneEuroFilter(1.0, 0.01)
a = f(1.0, 0.0)
b = f(2.0, 0.0)                                       # same timestamp
c = f(2.0, 0.07)                                      # irregular step
check("duplicate timestamp keeps the last output", np.isclose(b, a))
check("irregular timestamp step works", np.isfinite(c))

# 7. Arrays: every element is filtered independently, like separate scalar filters.
vals = rng.standard_normal((30, 3))
fa = ls.OneEuroFilter(1.0, 0.02)
arr = run(fa, vals)
cols = [run(ls.OneEuroFilter(1.0, 0.02), vals[:, j]) for j in range(3)]
check("array filter equals scalar filters", np.allclose(arr, np.stack(cols, axis=1)))

# 8. Landmarks: shape kept, NaN frame stays NaN and resets the filter.
W, H = 640, 480
clip = rng.uniform(0.3, 0.6, size=(12, 21, 3))
clip[:, :, 2] = 0.0
times = np.arange(12) * DT
clip_nan = clip.copy()
clip_nan[5] = np.nan
sm = ls.smooth_clip(clip_nan, times, W, H)
check("clip shape is kept", sm.shape == clip.shape)
check("a frame without a hand stays NaN", np.isnan(sm[5]).all())
check("other frames are finite", np.isfinite(sm[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]).all())
check("first frame after a gap is not dragged to the old position", np.allclose(sm[6], clip_nan[6]))

# 9. Streaming one frame at a time equals smoothing the whole clip.
sm_all = ls.smooth_clip(clip, times, W, H)
stream = ls.LandmarkSmoother()
sm_stream = np.stack([stream.update(clip[i], times[i], W, H) for i in range(12)])
check("streaming equals clip smoothing", np.allclose(sm_all, sm_stream))

# 10. Units: the output is in the stored units (a still hand comes back unchanged).
still = np.tile(clip[0], (30, 1, 1))
sm_still = ls.smooth_clip(still, np.arange(30) * DT, W, H)
check("a still hand keeps its stored coordinates", np.allclose(sm_still, still))

print("\nAll checks passed.")
