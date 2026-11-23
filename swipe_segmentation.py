"""Swipe segmentation (report section 4.5): find the out-and-back cycles in the stream of frames.

    analyzer = live_pipeline.FrameAnalyzer(classifier, profile)
    segmenter = SwipeSegmenter(profile)                   # profile: calibration.UserProfile or None
    r = analyzer.process(landmarks, t, width, height)     # one camera frame (live)
    swipe = segmenter.update(r)                           # a Swipe, or None
    if swipe:
        gate.on_swipe(*swipe.event())                     # trigger_gate.py decides about the command

    swipes = segmenter.segment_clip(analyzer.process_clip(lm, times, width, height))   # a recorded clip

Method. A swipe is an out-and-back cycle of the open palm (TEAM_BRIEF A.12): the hand goes to one side
and comes back, and the direction of the first stroke gives the label. The motion cue is the palm
centre (mean of landmarks 0, 5, 9, 13, 17) and its speed in hand sizes per second, both computed by
live_pipeline.FrameAnalyzer from past frames only. The segmenter is causal: it sees one frame at a time
and never looks ahead, so it runs on the live camera stream.

    IDLE    a ring buffer keeps the last PRE_FRAMES frames. When the speed rises above tau_on, a cycle
            starts (onset). The start position is the palm centre just before the movement: the latest
            buffered frame whose speed was below tau_off (or the frame before the onset).
    ACTIVE  every frame is added to the window, and the palm-centre path is measured relative to the
            start position, in hand sizes. The cycle ENDS in the first frame in which all three hold:
              - out:  the hand has gone at least swipe_min_amplitude hand sizes horizontally,
              - back: it is again within RETURN_TOL hand sizes of the start horizontally (and within
                      RETURN_FRACTION of the largest excursion, see below),
              - slow: the speed is below tau_off.
            That frame is the swipe event.

Why "back at the start" and not "still for a short time" (the first idea in A.8): at the turning point
of a swipe the palm stops for a moment. On the pilot clips (u99) the speed stayed below tau_off for 0 to
2 frames there in swipe_left, but for 6 to 8 frames (0.3 to 0.4 s) in 3 of the 5 valid swipe_right
clips. A rule "end when slow for 0.2 s" cuts those cycles at the turning point, and the return stroke
then looks like a swipe in the opposite direction. (docs/results/segmentation_pilot_2026-11-24.txt.)

Label (temporary rule, the rule baseline of 4.7): the sign of the first stroke in x, i.e. the sign of
the horizontal displacement in the first frame in which it reaches swipe_min_amplitude. The stored
picture is not mirrored, so x increasing first is swipe_left (validate_dataset.SWIPE_FIRST_STROKE).
An overshoot on the way back cannot change the label. Confidence: CONFIDENCE (1.0) for every cycle that
passes the checks below; the dynamic classifier (4.7) will replace label and confidence, using the
window stored in the Swipe.

A cycle is REJECTED (no swipe, back to IDLE) when
  - the speed stays below tau_off for STILL_REJECT_S without the end condition: a one-way movement
    (reaching for the keyboard) or a movement smaller than swipe_min_amplitude,
  - it ends less than MIN_DURATION_S after the onset (a jerk), or it is still going after
    MAX_DURATION_S (waving, talking with the hands),
  - the path is mostly vertical: vertical extent > VERTICAL_MAX x horizontal extent,
  - the hand is lost for more than MAX_LOST_FRAMES frames in a row.
After a rejection a new cycle can only start once the speed has stayed below tau_off for REARM_S;
otherwise waving that goes on past MAX_DURATION_S would start a new cycle at once and could end as a
false swipe. (After an accepted swipe no wait is needed; the cooldown is the trigger gate's job.)

RETURN_FRACTION: with the default minimum amplitude (0.5) the turning point is always further than
RETURN_TOL (0.4) from the start, but calibration may lower the minimum amplitude to 0.25. Requiring the
hand to be back within half of its largest excursion keeps a pause at the turning point from counting
as the return in that case.

Boundary with 4.9 (A.8): this module only answers "is this a swipe candidate?". No cooldown, no
confidence threshold and no command logic here; they are in trigger_gate.py.

The thresholds tau_on, tau_off and swipe_min_amplitude come from the calibration profile
(calibration.UserProfile), or from calibration.DEFAULTS without one. The constants below are this
module's own.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

import calibration as cal
import validate_dataset as vd

# PROVISIONAL: pilot data of one person (u99, 6 swipe_left and 5 valid swipe_right clips, 20 frames per s).
PRE_FRAMES = 5           # frames kept before the onset (0.25 s); the frame before the onset is always among them
RETURN_TOL = 0.4         # hand sizes; the pilot cycles ended 0.02 to 0.35 hand sizes from the start
RETURN_FRACTION = 0.5    # "back" also means within half of the largest excursion (pilot: 0.02 to 0.33 of it)
STILL_REJECT_S = 0.6     # slow for this long without return = no swipe; longest pause at a pilot turning point 0.40 s
MIN_DURATION_S = 0.25    # onset to end; the pilot cycles took 0.61 to 1.30 s (a 0.1 s jerk measures about 0.25 s)
MAX_DURATION_S = 2.0     # onset to end; about 1.5 x the slowest pilot cycle
VERTICAL_MAX = 1.0       # vertical / horizontal extent of the path; pilot swipes 0.17 to 0.86
MAX_LOST_FRAMES = 2      # frames without a hand tolerated inside a cycle (the gate also tolerates 2)
REARM_S = 0.25           # slow for this long after a rejection before a new onset counts (5 frames)
CONFIDENCE = 1.0         # every accepted cycle, until the dynamic classifier (4.7) gives a real value

LABEL = {sign: g for g, sign in vd.SWIPE_FIRST_STROKE.items()}   # +1 -> swipe_left, -1 -> swipe_right
EPS = 1e-6


@dataclass
class Swipe:
    """One out-and-back cycle. The window runs from the start-position frame to the end frame."""
    t_start: float           # movement onset: first frame with palm speed above tau_on
    t_end: float             # the cycle is complete: the time of the swipe event
    label: str               # swipe_left or swipe_right (temporary rule: sign of the first stroke)
    conf: float
    amplitude: float         # first stroke: largest horizontal excursion in its direction, hand sizes
    times: np.ndarray        # (N,) frame times of the window
    landmarks: np.ndarray    # (N, 21, 3) smoothed landmarks in the stored units; NaN where the hand was lost
    path: np.ndarray         # (N, 2) palm centre minus the start position, in hand sizes (x, y as in the picture)

    def event(self):
        """(t, label, confidence): the arguments of TriggerGate.on_swipe and an item of run(swipes=...)."""
        return self.t_end, self.label, self.conf


class SwipeSegmenter:
    def __init__(self, profile=None):
        p = profile or cal.UserProfile()
        self.tau_on = p.tau_on
        self.tau_off = p.tau_off
        self.min_amplitude = p.swipe_min_amplitude
        self.reset()

    def reset(self):
        self.active = False
        self.buffer = deque(maxlen=PRE_FRAMES)    # IDLE: recent frames with a hand
        self.armed = True
        self.calm_since = None                    # start of the current slow stretch while not armed
        self.rejects = []                         # one dict per rejected cycle (diagnostics only)

    # ------------------------------------------------------------------ helpers
    def _slow(self, speed):
        return speed is not None and not np.isnan(speed) and speed < self.tau_off

    @staticmethod
    def _frame(f):
        return {"t": float(f["t"]), "speed": float(f["speed"]), "centre": np.asarray(f["centre"], dtype=np.float64),
                "size": float(f["size"]), "smoothed": np.asarray(f["smoothed"], dtype=np.float64)}

    def _start(self, cur):
        before = list(self.buffer)[:-1]
        ref = next((f for f in reversed(before) if self._slow(f["speed"])), before[-1] if before else cur)
        k = next(i for i, f in enumerate(self.buffer) if f is ref)
        self.active = True
        self.t_onset = cur["t"]
        self.c0 = ref["centre"]
        self.times, self.lms, self.path = [], [], []
        for f in list(self.buffer)[k:]:
            self._add(f)
        self.sign, self.peak, self.amp = 0, 0.0, 0.0
        self.missing = ["first stroke below minimum amplitude"]   # end conditions not met in the latest frame
        self.slow_since = None
        self.lost = 0
        self.buffer.clear()

    def _add(self, f):
        d = (f["centre"] - self.c0) / f["size"]
        self.times.append(f["t"])
        self.lms.append(f["smoothed"])
        self.path.append(d)
        return d

    def _reject(self, t, reason, slow):
        self.rejects.append({"t_start": self.t_onset, "t": t, "reason": reason, "peak": self.peak})
        self.active = False
        self.armed = False
        self.calm_since = self.slow_since if slow else None   # a hand that is already still counts at once
        self.buffer.clear()

    # ------------------------------------------------------------------ input
    def update(self, frame):
        """One result of FrameAnalyzer.process(). Returns a Swipe when a cycle ends in this frame, else None."""
        t = float(frame["t"])
        present = bool(frame["present"])
        if self.active:
            return self._update_active(frame, t, present)
        if not present:
            self.buffer.clear()
            self.calm_since = None
            return None
        f = self._frame(frame)
        self.buffer.append(f)
        if not self.armed:
            if self._slow(f["speed"]):
                self.calm_since = t if self.calm_since is None else self.calm_since
                if t - self.calm_since >= REARM_S - EPS:
                    self.armed, self.calm_since = True, None
            else:
                self.calm_since = None
            return None
        if f["speed"] > self.tau_on:
            self._start(f)
        return None

    def _update_active(self, frame, t, present):
        if not present:
            self.lost += 1
            if self.lost > MAX_LOST_FRAMES:
                self._reject(t, "hand lost", slow=False)
            else:
                self.times.append(t)
                self.lms.append(np.full((21, 3), np.nan))
                self.path.append(np.full(2, np.nan))
            return None
        self.lost = 0
        f = self._frame(frame)
        dx = self._add(f)[0]
        if self.sign == 0 and abs(dx) >= self.min_amplitude - EPS:
            self.sign = int(np.sign(dx))
        self.peak = max(self.peak, abs(dx))
        if self.sign:
            self.amp = max(self.amp, self.sign * dx)
        slow = self._slow(f["speed"])
        if slow:
            self.slow_since = t if self.slow_since is None else self.slow_since
        else:
            self.slow_since = None

        duration = t - self.t_onset
        back = abs(dx) <= RETURN_TOL and abs(dx) <= RETURN_FRACTION * self.peak
        self.missing = [name for name, ok in (("first stroke below minimum amplitude", self.sign),
                                              ("not back at the start", back),
                                              ("speed not below tau_off", slow)) if not ok]
        if self.sign and back and slow:
            path = np.array(self.path)
            ext_x, ext_y = np.nanmax(path[:, 0]) - np.nanmin(path[:, 0]), np.nanmax(path[:, 1]) - np.nanmin(path[:, 1])
            if duration < MIN_DURATION_S - EPS:
                self._reject(t, f"too short ({duration:.2f} s)", slow)
                return None
            if ext_y > VERTICAL_MAX * ext_x:
                self._reject(t, f"mostly vertical (vertical {ext_y:.2f}, horizontal {ext_x:.2f} hand sizes)", slow)
                return None
            swipe = Swipe(t_start=self.t_onset, t_end=t, label=LABEL[self.sign], conf=CONFIDENCE,
                          amplitude=float(self.amp), times=np.array(self.times), landmarks=np.array(self.lms),
                          path=path)
            self.active = False
            self.buffer.clear()
            self.buffer.append(f)                 # the resting frame can be the start of the next swipe
            return swipe
        if duration > MAX_DURATION_S + EPS:
            self._reject(t, f"too long (over {MAX_DURATION_S:.1f} s)", slow)
        elif self.slow_since is not None and t - self.slow_since >= STILL_REJECT_S - EPS:
            reason = "no return (one-way movement)" if self.sign else "below minimum amplitude"
            self._reject(t, reason, slow)
        return None

    # ------------------------------------------------------------------ recorded clips
    def segment_clip(self, result):
        """All swipes of a recorded clip: `result` is FrameAnalyzer.process_clip() output. Starts from a fresh
        state (reset) and gives exactly what update() gives frame by frame. A cycle still running at the end
        of the clip is listed in self.rejects as "unfinished at the end", with the end conditions that were
        not met in the last frame."""
        self.reset()
        out = []
        keys = ("t", "present", "speed", "centre", "size", "smoothed")
        for i in range(len(result["t"])):
            s = self.update({k: result[k][i] for k in keys})
            if s is not None:
                out.append(s)
        if self.active:
            self.rejects.append({"t_start": self.t_onset, "t": float(result["t"][-1]),
                                 "reason": f"unfinished at the end ({', '.join(self.missing)})", "peak": self.peak})
        return out
