"""User calibration (report section 4.8, experiment E3).

At start-up the user shows every static gesture for about 2 seconds and makes one swipe in each
direction. The program says which gesture to show, so a label never comes from the user's choice.
calibrate() turns these recordings into a UserProfile: the thresholds that later stages read.
Without a profile every stage uses the population defaults (DEFAULTS).

What is measured and what it sets (every value is kept within LIMITS around its default, so a bad
calibration cannot move the system far from the uncalibrated one):
  still hand: all frames of the held gestures, except deliberate movement (palm faster than the
      highest allowed v_max)
      palm speed and pose change, 99th percentile x STILL_MARGIN -> v_max, pose_max
      (stability check of the static branch, used by the trigger gate, 4.9; live_pipeline.py)
      These two are only RAISED (STILL_LIMITS): the defaults are already about twice the largest
      value on still clips, so a calmer user gains nothing from a stricter check, while a stricter
      check can break that user's real holds (on the pilot, one 2-second hold per gesture gave
      0.15 to 0.26 for the same person). A shakier user gets a looser check, up to twice the default.
      Only the steadiest second would be biased low: by construction it is the calmest part of the
      hold, and on the pilot clips it gave thresholds that some frames of the same person's still
      clips exceed.
  held gestures: the steadiest second (STEADY_S) of each
      finger measures -> thresholds of the rule-based classifier (RuleClassifier.fit on these frames,
      mixed BLEND : (1 - BLEND) with the population thresholds, then limited to RULE_LIMIT around them)
      confidence of a classifier on the frames it gets right -> its confidence threshold in 4.9;
      only lowered (to the CONF_PERCENTILE-th percentile), never below CONF_FLOOR
  swipes
      peak palm speed -> suggested tau_on = ON_FRACTION x the slower swipe (and at least
      2 x v_max, so a still hand and a swipe onset never overlap), tau_off = OFF_RATIO x tau_on
      first-stroke size -> suggested minimum swipe amplitude (half of the smaller swipe)
      (for the swipe segmentation, 4.5, which is not written yet; the gate's speed gate uses tau_on)
  hand size in pixels -> only a warning when it is outside the range validate_dataset.py accepts

Checks: every held gesture is classified with the UNCALIBRATED rules first. If most of its steady frames
look like a DIFFERENT gesture, the step fails and has to be repeated (a wrong pose would teach the
calibration the wrong thing; compare the u01 open_palm clips that were thumb-up poses). If most frames
are "no gesture", the step is kept with a warning: that is exactly the case calibration should fix.
A swipe whose first stroke goes the wrong way, or whose peak speed is below the lowest allowed tau_on,
fails as well. A gesture that was not recorded keeps the defaults ("missing").

Not done on purpose (TEAM_BRIEF D.4 item 14): calibration does not set the scale s, because
normalisation already divides every frame by its own hand size; and there is no separate neutral-pose
step, because the noise of a still hand is measured on the held gestures.

The live, guided calibration with the camera is not written yet (postponed with the live system).
Recorded clips work the same way:

    .venv\\Scripts\\python.exe calibration.py --user u99                    # clip 1 of every gesture
    .venv\\Scripts\\python.exe calibration.py --user u99 --clip 2 --save profiles/u99.json
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import argparse
import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import dataset_config as cfg
import dataset_loader as dl
import landmark_features as lf
import landmark_smoothing as ls
import rule_classifier as rc
import train_static as ts
import validate_dataset as vd

STATIC_STEPS = [g for g, kind in cfg.GESTURES.items() if kind == "static"]
SWIPE_STEPS = [g for g, kind in cfg.GESTURES.items() if kind == "dynamic"]

# Population defaults. PROVISIONAL: pilot clips of one person (u99).
#   v_max, pose_max: the stability thresholds of dataset_loader.py (about twice the largest value on
#                    still clips).
#   tau_on, tau_off: palm speed in hand sizes per second. Still clips of u99 stayed under 0.18, the
#                    peaks of the 12 swipes were 1.8 to 9.0.
#   conf_threshold : rules 0.7 (E1 pilot: at 0.8 only 84 % of the right answers pass, at 0.9 26 %);
#                    the SVM gets the same value until there are data to choose it on.
DEFAULTS = {
    "v_max": dl.V_MAX,
    "pose_max": dl.POSE_MAX,
    "tau_on": 1.0,
    "tau_off": 0.5,
    "swipe_min_amplitude": vd.SWIPE_MIN_AMPLITUDE,
    "conf_threshold": {"rules": 0.7, "svm": 0.7},
}
LIMITS = (0.5, 2.0)       # a calibrated value stays between 0.5 and 2 times its default
STILL_LIMITS = (1.0, 2.0) # v_max and pose_max are only raised (see the module text)
RULE_LIMIT = 0.3          # a calibrated finger threshold stays within +-30 % of the population threshold
BLEND = 0.5               # weight of the user's own finger thresholds
STILL_MARGIN = 2.0
STEADY_S = 1.0            # length of the steadiest stretch taken from every held gesture
MIN_MATCH = 0.5           # share of steady frames that must not look like another gesture
CONF_PERCENTILE = 10
CONF_FLOOR = 0.5
ON_FRACTION = 0.3
OFF_RATIO = 0.5
SWIPE_AMPLITUDE_FRACTION = 0.5


@dataclass
class UserProfile:
    """The numbers calibration produces. Every stage falls back to DEFAULTS without a profile."""
    user: str = ""
    v_max: float = DEFAULTS["v_max"]
    pose_max: float = DEFAULTS["pose_max"]
    tau_on: float = DEFAULTS["tau_on"]
    tau_off: float = DEFAULTS["tau_off"]
    swipe_min_amplitude: float = DEFAULTS["swipe_min_amplitude"]
    rule_thr: list = field(default_factory=lambda: rc.DEFAULT_THR.tolist())
    rule_scale: list = field(default_factory=lambda: rc.DEFAULT_SCALE.tolist())
    conf_threshold: dict = field(default_factory=lambda: dict(DEFAULTS["conf_threshold"]))
    hand_size_px: float = None
    calibrated: list = field(default_factory=list)      # names of the values calibration changed
    created: str = ""

    def rules(self):
        """A RuleClassifier with this profile's finger thresholds."""
        r = rc.RuleClassifier()
        r.thr, r.scale = np.array(self.rule_thr, dtype=np.float64), np.array(self.rule_scale, dtype=np.float64)
        return r

    def confidence_threshold(self, model_name):
        return float(self.conf_threshold.get(model_name, DEFAULTS["conf_threshold"].get(model_name, 0.7)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class StepResult:
    step: str
    status: str           # ok, warn, fail, missing
    message: str
    frames: int = 0


@dataclass
class Calibration:
    profile: UserProfile
    steps: list           # StepResult per step, in the order of STATIC_STEPS + SWIPE_STEPS
    Q: np.ndarray         # steady frames of the accepted static steps, normalised (N, 21, 3)
    y: np.ndarray         # their gesture names

    def failed(self):
        return [s.step for s in self.steps if s.status == "fail"]


def _limit(value, default, limits=LIMITS):
    return float(np.clip(value, default * limits[0], default * limits[1]))


def steadiest(speed, pose, times, seconds=STEADY_S):
    """Indices of the stretch of `seconds` with the lowest median palm speed and a hand in every frame.

    Returns an empty array if no such stretch exists.
    """
    n = len(speed)
    if n < 2:
        return np.array([], dtype=int)
    fps = 1.0 / np.median(np.diff(times))
    w = max(3, min(n, int(round(seconds * fps))))
    best, best_i = np.inf, None
    for i in range(n - w + 1):
        v, p = speed[i:i + w], pose[i:i + w]
        if np.isnan(v).any() or np.isnan(p).any():
            continue
        m = np.median(v)
        if m < best:
            best, best_i = m, i
    return np.array([], dtype=int) if best_i is None else np.arange(best_i, best_i + w)


def _majority(labels):
    names, counts = np.unique(labels, return_counts=True)
    k = counts.argmax()
    return names[k], counts[k] / len(labels)


def calibrate(segments, user="", rules=None, models=None, blend=BLEND):
    """Build a UserProfile from calibration recordings.

    segments : {gesture: (landmarks (T, 21, 3) raw stored values, times (T,), width, height)} for the
               static gestures and the swipes; a gesture that is not in it keeps the defaults.
    rules    : the population RuleClassifier (uncalibrated); default = default thresholds. It checks
               every held gesture and is the starting point of the finger thresholds.
    models   : {name: fitted classifier} whose confidence threshold is calibrated, e.g. {"svm": svm}.
               "rules" is always included (the calibrated rules).
    Returns a Calibration (profile, step results, the accepted steady frames).
    """
    rules = rules or rc.RuleClassifier()
    prof = UserProfile(user=user, rule_thr=rules.thr.tolist(), rule_scale=rules.scale.tolist(),
                       created=datetime.date.today().isoformat())
    steps, speeds, poses, sizes = [], [], [], []
    Qs, ys = [], []

    # ------------------------------------------------------------ held gestures
    for g in STATIC_STEPS:
        if g not in segments:
            steps.append(StepResult(g, "missing", "not recorded: defaults kept"))
            continue
        lm, t, w, h = segments[g]
        sm = ls.smooth_clip(lm, t, w, h)
        present = ~np.isnan(sm[:, 0, 0])
        if present.sum() < 3:
            steps.append(StepResult(g, "fail", "hand not found: repeat this step"))
            continue
        speed, pose = dl.motion(sm, t, w, h)
        idx = steadiest(speed, pose, t)
        if len(idx) == 0:
            steps.append(StepResult(g, "fail", "no steady stretch with the hand in every frame: repeat this step"))
            continue
        Q = lf.normalize(sm[idx], w, h)
        labels, _ = rules.predict_with_confidence(Q)
        top, share = _majority(labels)
        others = labels[(labels != g) & (labels != "no_gesture")]
        if len(others) > (1 - MIN_MATCH) * len(labels):
            wrong, wshare = _majority(others)
            steps.append(StepResult(g, "fail", f"looks like {wrong} ({wshare * len(others) / len(labels) * 100:.0f}% "
                                               f"of the steady frames): repeat this step", len(idx)))
            continue
        status, msg = "ok", f"recognised in {np.mean(labels == g) * 100:.0f}% of the steady frames"
        if top == "no_gesture":
            status = "warn"
            msg = (f"uncalibrated rules see no gesture in {share * 100:.0f}% of the steady frames; kept, "
                   f"calibration adapts the finger thresholds")
        steps.append(StepResult(g, status, msg, len(idx)))
        hold = ~np.isnan(speed) & ~np.isnan(pose) & (speed < DEFAULTS["v_max"] * LIMITS[1])
        speeds.append(speed[hold])
        poses.append(pose[hold])
        sizes.append(lf.hand_size(sm[idx], w, h))
        Qs.append(Q)
        ys.append(np.full(len(idx), g, dtype=object))

    Qcal = np.concatenate(Qs) if Qs else np.zeros((0, 21, 3))
    ycal = np.concatenate(ys) if ys else np.array([], dtype=object)

    if speeds:
        v, p = np.concatenate(speeds), np.concatenate(poses)
        prof.v_max = _limit(STILL_MARGIN * np.percentile(v, 99), DEFAULTS["v_max"], STILL_LIMITS)
        prof.pose_max = _limit(STILL_MARGIN * np.percentile(p, 99), DEFAULTS["pose_max"], STILL_LIMITS)
        prof.hand_size_px = float(np.nanmedian(np.concatenate(sizes)))
        prof.calibrated += ["v_max", "pose_max"]

    # ------------------------------------------------------------ finger thresholds of the rules
    if len(ycal):
        own = rc.RuleClassifier()
        own.thr, own.scale = rules.thr.copy(), rules.scale.copy()   # fingers without data keep the population value
        own.fit(Qcal, ycal)
        thr = (1 - blend) * rules.thr + blend * own.thr
        prof.rule_thr = np.clip(thr, rules.thr * (1 - RULE_LIMIT), rules.thr * (1 + RULE_LIMIT)).tolist()
        prof.rule_scale = ((1 - blend) * rules.scale + blend * own.scale).tolist()
        if not np.allclose(prof.rule_thr, rules.thr):
            prof.calibrated.append("rule_thr")

    # ------------------------------------------------------------ confidence thresholds
    all_models = {"rules": prof.rules(), **(models or {})}
    for name, model in all_models.items():
        if not len(ycal):
            break
        pred, conf = model.predict_with_confidence(Qcal)
        right = conf[pred == ycal]
        default = DEFAULTS["conf_threshold"].get(name, 0.7)
        if len(right) < 5:
            continue
        value = float(np.clip(min(default, np.percentile(right, CONF_PERCENTILE)), CONF_FLOOR, default))
        prof.conf_threshold[name] = value
        if value < default:
            prof.calibrated.append(f"conf_threshold[{name}]")

    # ------------------------------------------------------------ swipes
    peaks, amps = [], []
    for g in SWIPE_STEPS:
        if g not in segments:
            steps.append(StepResult(g, "missing", "not recorded: defaults kept"))
            continue
        lm, t, w, h = segments[g]
        sm = ls.smooth_clip(lm, t, w, h)
        if (~np.isnan(sm[:, 0, 0])).sum() < 5:
            steps.append(StepResult(g, "fail", "hand not found: repeat this step"))
            continue
        speed, _ = dl.motion(sm, t, w, h)
        peak = float(np.nanmax(speed))
        x, _, _ = vd.palm_x_and_y(sm, w, h)
        amp, sign = vd.first_stroke(x)
        if sign != vd.SWIPE_FIRST_STROKE[g]:
            steps.append(StepResult(g, "fail", "the first stroke goes the wrong way: repeat this step", len(t)))
            continue
        if peak < DEFAULTS["tau_on"] * LIMITS[0]:
            steps.append(StepResult(g, "fail", f"too slow (peak {peak:.2f} hand sizes per second): repeat faster",
                                    len(t)))
            continue
        status, msg = "ok", f"peak speed {peak:.2f} hand sizes per second, first stroke {abs(amp):.2f} hand sizes"
        if abs(amp) < vd.SWIPE_MIN_AMPLITUDE:
            status, msg = "warn", msg + " (small: make the movement larger)"
        steps.append(StepResult(g, status, msg, len(t)))
        peaks.append(peak)
        amps.append(abs(amp))
    if peaks:
        on = max(ON_FRACTION * min(peaks), 2 * prof.v_max)
        prof.tau_on = _limit(on, DEFAULTS["tau_on"])
        prof.tau_off = _limit(OFF_RATIO * prof.tau_on, DEFAULTS["tau_off"])
        prof.swipe_min_amplitude = _limit(SWIPE_AMPLITUDE_FRACTION * min(amps), DEFAULTS["swipe_min_amplitude"])
        prof.calibrated += ["tau_on", "tau_off", "swipe_min_amplitude"]

    # ------------------------------------------------------------ distance
    if prof.hand_size_px is not None and not (vd.HAND_SIZE_PX[0] <= prof.hand_size_px <= vd.HAND_SIZE_PX[1]):
        steps.append(StepResult("hand size", "warn",
                                f"{prof.hand_size_px:.0f} px is outside {vd.HAND_SIZE_PX[0]:.0f} to "
                                f"{vd.HAND_SIZE_PX[1]:.0f} px: move {'closer' if prof.hand_size_px < vd.HAND_SIZE_PX[0] else 'back'}"))
    return Calibration(prof, steps, Qcal, ycal)


def segments_from_clips(clips, user, clip=1):
    """{gesture: segment} from load_clips() output: clip number `clip` of every gesture of `user`."""
    out = {}
    for c in clips:
        if c["user"] == user and c["clip"] == clip and c["gesture"] in STATIC_STEPS + SWIPE_STEPS:
            out[c["gesture"]] = (c["landmarks"], c["times"], c["width"], c["height"])
    return out


def adapted_svm(pop_Q, pop_y, pop_groups, cal_Q, cal_y, copies=1, **svm_args):
    """The learned model adapted to one user (E3 variant): LandmarkSVM trained on the population frames
    plus `copies` copies of the user's calibration frames.

    Clip groups (they decide the held-out parts of the confidence calibration in LandmarkSVM): one group
    per calibration gesture. A gesture that the population does not have at all is cut into 3 groups of
    consecutive frames instead; otherwise some held-out parts would lack it completely. Frames of one
    recording are alike, so the confidences of such a gesture come out somewhat too high.
    """
    cal_y = np.asarray(cal_y, dtype=object)
    known = set(np.asarray(pop_y).tolist())
    cal_groups = np.empty(len(cal_y), dtype=object)
    for g in set(cal_y.tolist()):
        idx = np.flatnonzero(cal_y == g)
        parts = 1 if g in known else 3
        for k, chunk in enumerate(np.array_split(idx, parts)):
            cal_groups[chunk] = f"calibration#{g}#{k}"
    Q = np.concatenate([pop_Q] + [cal_Q] * copies)
    y = np.concatenate([pop_y] + [cal_y] * copies)
    groups = np.concatenate([np.asarray(pop_groups).astype(str).astype(object)] + [cal_groups] * copies)
    return ts.LandmarkSVM(**svm_args).fit(Q, y, groups=groups)


def describe(cal):
    """Readable summary: the step results and every profile value next to its default."""
    p = cal.profile
    lines = [f"Calibration of {p.user or '(no user id)'}:"]
    for s in cal.steps:
        lines.append(f"   {s.step:<12} {s.status:<8} {s.message}" + (f"  [{s.frames} frames]" if s.frames else ""))
    lines.append("   value                      calibrated    default")
    for name in ("v_max", "pose_max", "tau_on", "tau_off", "swipe_min_amplitude"):
        mark = "" if name in p.calibrated else "   (not calibrated)"
        lines.append(f"   {name:<24} {getattr(p, name):>12.3f} {DEFAULTS[name]:>10.3f}{mark}")
    for name, value in p.conf_threshold.items():
        lines.append(f"   {'confidence ' + name:<24} {value:>12.3f} {DEFAULTS['conf_threshold'].get(name, 0.7):>10.3f}")
    for k, f in enumerate(rc.FINGERS):
        lines.append(f"   {'rule threshold ' + f:<24} {p.rule_thr[k]:>12.3f} {rc.DEFAULT_THR[k]:>10.3f}")
    if p.hand_size_px is not None:
        lines.append(f"   hand size: {p.hand_size_px:.0f} px (only used for the distance warning)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Calibration profile from recorded clips (report section 4.8).")
    ap.add_argument("--user", required=True)
    ap.add_argument("--clip", type=int, default=1, help="clip number of every gesture used as calibration (default 1)")
    ap.add_argument("--save", help="write the profile as JSON to this path")
    args = ap.parse_args()
    clips = dl.load_clips(users=[args.user])
    if not clips:
        raise SystemExit(f"no clips for user {args.user}")
    cal = calibrate(segments_from_clips(clips, args.user, args.clip), user=args.user)
    print(describe(cal))
    if cal.failed():
        print(f"\nFailed steps (would be repeated in the live calibration): {', '.join(cal.failed())}")
    if args.save:
        cal.profile.save(args.save)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
