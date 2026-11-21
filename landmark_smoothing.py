"""Temporal smoothing of landmarks with the 1-euro filter (report section 4.4).

Casiez, Roussel and Vogel, "1 euro filter: a simple speed-based low-pass filter for noisy
input in interactive systems", CHI 2012.

Why this filter: a fixed low-pass filter has to trade jitter against lag. The 1-euro filter
lowers its cut-off frequency when the signal moves slowly (strong smoothing, little jitter)
and raises it when the signal moves fast (little lag):

    dx_hat = low-pass(dx, d_cutoff)               smoothed speed
    f_c    = f_cmin + beta * |dx_hat|             cut-off grows with speed
    alpha  = 1 / (1 + tau / dt),  tau = 1 / (2 pi f_c)
    x_hat  = alpha * x + (1 - alpha) * x_hat_prev

Parameters
    f_cmin   (Hz)   cut-off at rest. Lower = less jitter but more lag when moving slowly.
    beta            how fast the cut-off rises with speed. Higher = less lag in fast motion.
    d_cutoff (Hz)   cut-off of the speed estimate; the paper keeps it at 1 Hz.
Tuning order (from the paper): set beta = 0 and tune f_cmin against jitter on a still hand,
then raise beta until the lag in fast movements is acceptable.

The filter is causal (it only uses past frames), so the same code runs on a recorded clip
and on the live camera stream. It must be applied to the RAW landmarks, before
normalisation (landmark_features.py). It works in pixel units, because beta depends on the
unit of the signal; the result is converted back to the stored units.

A frame without a hand (NaN) resets the filter, so a hand that comes back later is not
dragged towards the position it had before.

The default parameters are PROVISIONAL: they were chosen on the pilot clips of one person
(see evaluate_smoothing.py) and must be tuned again on the training users.
"""
import numpy as np

import landmark_features as lf

# Provisional values, see the note above. Chosen by evaluate_smoothing.py on the pilot
# clips of one person: at most 30 ms added delay and at most 3 percent flattening of a
# swipe, then the largest drop in still-hand speed noise. The result sits at the edge of
# both limits; f_cmin = 1.0 with beta = 0.02 is almost as good with a little more margin.
F_CMIN = 0.5
BETA = 0.02
D_CUTOFF = 1.0


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """1-euro filter for a value or an array of values (all elements filtered independently)."""

    def __init__(self, f_cmin=F_CMIN, beta=BETA, d_cutoff=D_CUTOFF):
        self.f_cmin, self.beta, self.d_cutoff = f_cmin, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t_prev = None
        self.x_hat = None
        self.dx_hat = None

    def __call__(self, x, t):
        """Filter one sample x taken at time t (seconds). Returns the smoothed sample."""
        x = np.asarray(x, dtype=np.float64)
        if self.t_prev is None:
            self.t_prev, self.x_hat, self.dx_hat = t, x.copy(), np.zeros_like(x)
            return x.copy()
        dt = t - self.t_prev
        if dt <= 0:                       # duplicate or backwards timestamp: keep the last output
            return self.x_hat.copy()
        dx = (x - self.x_hat) / dt
        self.dx_hat = self.dx_hat + _alpha(self.d_cutoff, dt) * (dx - self.dx_hat)
        cutoff = self.f_cmin + self.beta * np.abs(self.dx_hat)
        self.x_hat = self.x_hat + _alpha(cutoff, dt) * (x - self.x_hat)
        self.t_prev = t
        return self.x_hat.copy()


class LandmarkSmoother:
    """Smooths a stream of (21, 3) landmark frames, one frame at a time (live use).

    update(landmarks, t, width, height) takes the raw landmarks in the stored units
    (x, y normalised by the image size) and returns the smoothed landmarks in the same units.
    """

    def __init__(self, f_cmin=F_CMIN, beta=BETA, d_cutoff=D_CUTOFF):
        self.filter = OneEuroFilter(f_cmin, beta, d_cutoff)

    def reset(self):
        self.filter.reset()

    def update(self, landmarks, t, width=None, height=None):
        px = lf.to_pixel_units(landmarks, width, height)
        if np.isnan(px).any():
            self.filter.reset()
            return np.full(px.shape, np.nan)
        out = self.filter(px, t)
        w, h = (width, height) if width and height else lf.DEFAULT_SIZE
        return out / np.array([w, h, w], dtype=np.float64)


def smooth_clip(landmarks, times, width=None, height=None,
                f_cmin=F_CMIN, beta=BETA, d_cutoff=D_CUTOFF):
    """Smooth a recorded clip, shape (T, 21, 3), with timestamps (T,) in seconds.

    Same result as feeding the frames one by one through LandmarkSmoother.
    """
    smoother = LandmarkSmoother(f_cmin, beta, d_cutoff)
    lm = np.asarray(landmarks, dtype=np.float64)
    return np.stack([smoother.update(lm[i], times[i], width, height) for i in range(len(lm))])
