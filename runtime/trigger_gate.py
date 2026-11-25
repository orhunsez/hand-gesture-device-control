"""False-trigger prevention (report section 4.9, experiment E4): which recognised gestures become commands.

The gate sits after the classifiers and before the command mapping (4.10). It receives one result per
camera frame from the static branch (label, confidence, is the hand calm, is it moving fast) and the
swipes found by the dynamic branch, and it emits events:

    press   a static gesture starts: the command is applied once, at once
    repeat  the gesture is still held: the command is applied again (every 0.2 s)
    release the held gesture has ended
    swipe   a swipe gives its command (one per out-and-back cycle)

Decided behaviour (TEAM_BRIEF D.4 items 10 to 13):
  - No activation gesture: a static gesture starts its command as soon as it is recognised, like a
    phone's volume key. "Recognised" = N_ON consecutive frames (about 0.15 s at 20 frames per second)
    with the same label, enough confidence and a calm hand.
  - Only the open palm waits about half a second (HOLD_S), because it is also the swipe hand shape.
  - Holding repeats: while the gesture is held, the command repeats every REPEAT_S = 0.2 s, the same
    for every gesture; the first repeat also comes 0.2 s after the press. The gesture has to be
    released before the same gesture can press again.

Safeguards, each of which can be switched off for the E4 ablation (GateConfig):
  use_confidence    a frame counts only if the classifier confidence reaches conf_on (to start) or
                    conf_hold (to keep holding); no_gesture and frames without a hand never count
  use_confirmation  N_ON consecutive frames before a press, and the open-palm wait (off: 1 frame)
  use_stillness     the hand must be calm to start a gesture (palm speed and pose change below the
                    stability thresholds), and fast movement (palm speed above tau_on, the swipe onset)
                    ends a hold: the speed gate of A.12 rule 1
  use_hysteresis    holding is easier than starting: confidence conf_hold < conf_on, a moving (but not
                    fast) hand is allowed, and up to DROP_FRAMES frames of another label or of a lost
                    hand are tolerated, so a flickering hold does not turn into two presses
  use_cooldown      after a swipe, no press and no other swipe for SWIPE_COOLDOWN_S (A.12 rule 3)

What is a frame result: label (a gesture name, "no_gesture" or None for no hand), confidence in 0..1,
calm and moving (bool). live_pipeline.FrameAnalyzer computes them; the thresholds come from the user's
calibration profile when there is one (GateConfig.from_profile).
The dynamic branch (segmentation 4.5 and classifier 4.7) is not written yet; on_swipe() is its entry.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import dataset_config as cfg

STATIC = {g for g, kind in cfg.GESTURES.items() if kind == "static"}
DYNAMIC = {g for g, kind in cfg.GESTURES.items() if kind == "dynamic"}
EPS = 1e-6

N_ON = 3                    # frames, about 0.15 s at 20 frames per second
HOLD_S = {"open_palm": 0.5}  # extra wait before these gestures press (A.12 rule 2, D.4 item 12)
REPEAT_S = 0.2               # D.4 item 13
DROP_FRAMES = 2
SWIPE_COOLDOWN_S = 1.0
CONF_ON = 0.7
CONF_HOLD = 0.5


class Event(NamedTuple):
    t: float
    kind: str        # press, repeat, release, swipe
    gesture: str


@dataclass
class GateConfig:
    conf_on: float = CONF_ON
    conf_hold: float = CONF_HOLD
    conf_swipe: float = CONF_ON
    n_on: int = N_ON
    hold_s: dict = field(default_factory=lambda: dict(HOLD_S))
    repeat_delay_s: float = REPEAT_S
    repeat_interval_s: float = REPEAT_S
    drop_frames: int = DROP_FRAMES
    swipe_cooldown_s: float = SWIPE_COOLDOWN_S
    use_confidence: bool = True
    use_confirmation: bool = True
    use_stillness: bool = True
    use_hysteresis: bool = True
    use_cooldown: bool = True

    @classmethod
    def from_profile(cls, profile, model_name, **overrides):
        """Confidence threshold of the static classifier `model_name` ("rules" or "svm") from a
        calibration.UserProfile. (The swipe threshold belongs to the dynamic classifier and stays.)"""
        on = profile.confidence_threshold(model_name)
        return cls(conf_on=on, conf_hold=min(CONF_HOLD, on), **overrides)

    def without(self, *safeguards):
        """A copy with the named safeguards switched off, e.g. without("cooldown", "hysteresis")."""
        return replace(self, **{f"use_{name}": False for name in safeguards})


SAFEGUARDS = ("confidence", "confirmation", "stillness", "hysteresis", "cooldown")


class TriggerGate:
    def __init__(self, config=None):
        self.cfg = config or GateConfig()
        self.reset()

    def reset(self):
        self.held = None              # gesture being held, or None
        self.next_repeat = None
        self.misses = 0
        self.cand, self.cand_count, self.cand_start = None, 0, None
        self.cooldown_until = float("-inf")
        self.blocked_swipes = 0

    # ------------------------------------------------------------------ helpers
    def _counts(self, t, label, conf, calm, moving):
        """Does this frame count towards STARTING `label`?"""
        c = self.cfg
        if label not in STATIC:
            return False
        if c.use_confidence and conf < c.conf_on:
            return False
        if c.use_stillness and (not calm or moving):
            return False
        if c.use_cooldown and t < self.cooldown_until:
            return False
        return True

    def _keeps(self, t, label, conf, calm, moving):
        """Does this frame keep the held gesture going?"""
        c = self.cfg
        if label != self.held:
            return False
        if not c.use_hysteresis:
            return self._counts(t, label, conf, calm, moving)
        if c.use_confidence and conf < c.conf_hold:
            return False
        if c.use_stillness and moving:
            return False
        return True

    def _clear_candidate(self):
        self.cand, self.cand_count, self.cand_start = None, 0, None

    # ------------------------------------------------------------------ input
    def update(self, t, label, conf=1.0, calm=True, moving=False):
        """One frame of the static branch. Returns the list of events of this frame."""
        c = self.cfg
        events = []
        if self.held is not None:
            if self._keeps(t, label, conf, calm, moving):
                self.misses = 0
                if t >= self.next_repeat - EPS:
                    events.append(Event(t, "repeat", self.held))
                    self.next_repeat += c.repeat_interval_s
                    if self.next_repeat <= t:            # a long gap between frames: no burst of repeats
                        self.next_repeat = t + c.repeat_interval_s
                self._clear_candidate()
                return events
            self.misses += 1
            if self.misses > (c.drop_frames if c.use_hysteresis else 0):
                events.append(Event(t, "release", self.held))
                self.held, self.misses = None, 0

        # count frames towards a (new) gesture; also during tolerated misses, so that a direct change
        # from one gesture to another presses the new one as soon as the old one is released
        if label != self.held and self._counts(t, label, conf, calm, moving):
            if label == self.cand:
                self.cand_count += 1
            else:
                self.cand, self.cand_count, self.cand_start = label, 1, t
        else:
            self._clear_candidate()

        if self.held is None and self.cand is not None:
            need = c.n_on if c.use_confirmation else 1
            wait = c.hold_s.get(self.cand, 0.0) if c.use_confirmation else 0.0
            if self.cand_count >= need and t - self.cand_start >= wait - EPS:
                events.append(Event(t, "press", self.cand))
                self.held, self.misses = self.cand, 0
                self.next_repeat = t + c.repeat_delay_s
                self._clear_candidate()
        return events

    def on_swipe(self, t, label, conf=1.0):
        """A swipe found by the dynamic branch at time t. Returns the list of events."""
        c = self.cfg
        if label not in DYNAMIC or (c.use_confidence and conf < c.conf_swipe):
            return []
        if c.use_cooldown and t < self.cooldown_until:
            self.blocked_swipes += 1
            return []
        events = []
        if self.held is not None:                   # a swipe ends any hold
            events.append(Event(t, "release", self.held))
            self.held, self.misses = None, 0
        events.append(Event(t, "swipe", label))
        if c.use_cooldown:
            self.cooldown_until = t + c.swipe_cooldown_s
        self._clear_candidate()
        return events

    # ------------------------------------------------------------------ recorded streams
    def run(self, times, labels, conf, calm, moving, swipes=()):
        """Feed a whole stream (arrays, one value per frame) and return all events.

        swipes: (t, label, confidence) tuples; each is delivered before the first frame at or after t.
        """
        self.reset()
        pending = sorted(swipes, key=lambda s: s[0])
        k, events = 0, []
        for i, t in enumerate(times):
            while k < len(pending) and pending[k][0] <= t + EPS:
                events += self.on_swipe(*pending[k])
                k += 1
            events += self.update(t, labels[i], conf[i], calm[i], moving[i])
        while k < len(pending):
            events += self.on_swipe(*pending[k])
            k += 1
        return events
