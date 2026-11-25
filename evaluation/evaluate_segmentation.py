"""Swipe segmentation (report section 4.5) on the recorded clips.

    .venv\\Scripts\\python.exe evaluate_segmentation.py                       # minimum amplitude from DEFAULTS
    .venv\\Scripts\\python.exe evaluate_segmentation.py --min-amplitude 0.3   # what a lower minimum would change

Every clip that has no validator ERROR (as in dataset_loader.py) is played frame by frame through the
static-branch analysis (live_pipeline.FrameAnalyzer: smoothing, palm-centre speed; the classifier output
is not used here) and the swipe segmenter (swipe_segmentation.py). Thresholds: calibration.DEFAULTS
(no calibration); --min-amplitude replaces only swipe_min_amplitude, for this evaluation.

Expected: every swipe clip gives exactly one swipe with its own label; a clip of a static gesture gives
none. Per clip it prints the number of swipes, label, onset (first frame above tau_on), end (the swipe
event), first-stroke amplitude and the delay from onset to event; a clip without a swipe gets the reason
(the cycle that was started and why it was dropped, or the peak speed when no cycle started).

Pilot finding (A.8): for every swipe clip it also prints the longest stretch below tau_off between the
first and the last frame above tau_on (the pause at the turning point), and whether the earlier end rule
"slow for 0.2 s" (evaluate_triggers.stand_in_swipe) would end the cycle while the hand is still moving.
"""
import sys, pathlib  # the code lives in folders (see README): make them importable
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1] / d) for d in ('data_tools', 'processing', 'recognition', 'runtime', 'evaluation', 'tests', 'tools')]
import argparse
import dataclasses

import numpy as np

import calibration as cal
import dataset_loader as dl
import evaluate_triggers as et
import live_pipeline as lp
import rule_classifier as rc
import swipe_segmentation as ss
import trigger_gate as tg


def longest_slow_run(r, profile):
    """Longest run of frames below tau_off between the first and last frame above tau_on: (frames, seconds)."""
    fast = np.flatnonzero(r["speed"] > profile.tau_on)
    if len(fast) < 2:
        return 0, 0.0
    best, run, best_s = 0, 0, 0.0
    t = r["t"]
    for i in range(fast[0], fast[-1] + 1):
        run = run + 1 if r["speed"][i] < profile.tau_off else 0
        if run > best:
            best, best_s = run, t[i] - t[i - run + 1] + (t[i] - t[i - 1])
    return best, best_s


def old_rule_splits(r, profile):
    """Does the end rule 'slow for 0.2 s' fire before the last frame above tau_on (the cycle is cut)?"""
    _, t_event = et.stand_in_swipe(r, profile)
    if t_event is None:
        return False
    return bool(np.any((r["t"] > t_event) & (r["speed"] > profile.tau_on)))


def describe_miss(seg, r):
    if seg.rejects:
        return "; ".join(f"cycle {x['t_start']:.2f}-{x['t']:.2f} s dropped: {x['reason']} "
                         f"(largest excursion {x['peak']:.2f} hand sizes)" for x in seg.rejects)
    return f"no cycle started (peak speed {np.nanmax(r['speed']):.2f}, tau_on {seg.tau_on:.2f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-amplitude", type=float,
                    help=f"minimum first stroke in hand sizes (default {cal.DEFAULTS['swipe_min_amplitude']}, "
                         f"calibration.DEFAULTS)")
    args = ap.parse_args()
    profile = cal.UserProfile()
    if args.min_amplitude is not None:
        profile = dataclasses.replace(profile, swipe_min_amplitude=args.min_amplitude)

    clips = dl.load_clips()
    analyzer = lp.FrameAnalyzer(rc.RuleClassifier(), profile)
    print(f"Thresholds: tau_on {profile.tau_on:.2f}, tau_off {profile.tau_off:.2f} hand sizes per second, "
          f"minimum amplitude {profile.swipe_min_amplitude:.2f} hand sizes"
          f"{' (--min-amplitude)' if args.min_amplitude is not None else ' (calibration.DEFAULTS)'}")
    print(f"Segmenter: return within {ss.RETURN_TOL} hand sizes (and {ss.RETURN_FRACTION} of the excursion), "
          f"reject when slow {ss.STILL_REJECT_S} s without return, duration {ss.MIN_DURATION_S} to "
          f"{ss.MAX_DURATION_S} s, vertical/horizontal <= {ss.VERTICAL_MAX}, at most {ss.MAX_LOST_FRAMES} lost frames")
    print(f"Clips: {len(clips)} without a validator ERROR "
          f"({sum(c['gesture'] in tg.DYNAMIC for c in clips)} swipe, {sum(c['gesture'] in tg.STATIC for c in clips)} static)")

    print(f"\n{'clip':<26} {'swipes':>6}  {'label':<12} {'onset s':>7} {'end s':>6} {'ampl':>5} {'delay s':>7}   note")
    ok, missed, wrong, delays, false_static, static_peaks = [], [], [], [], [], {}
    dips = []
    for c in clips:
        r = analyzer.process_clip(c["landmarks"], c["times"], c["width"], c["height"])
        seg = ss.SwipeSegmenter(profile)
        swipes = seg.segment_clip(r)
        name = f"{c['user']} {c['gesture']} #{c['clip']}"
        is_swipe = c["gesture"] in tg.DYNAMIC
        if not swipes:
            note = describe_miss(seg, r)
            if is_swipe:
                reason = seg.rejects[-1]["reason"] if seg.rejects else "no cycle started"
                if "below minimum amplitude" in reason:
                    reason = "below minimum amplitude"
                    note = "missed: below minimum amplitude; " + note
                missed.append((name, reason))
            print(f"{name:<26} {0:>6}  {'-':<12} {'-':>7} {'-':>6} {'-':>5} {'-':>7}   {note}")
        for k, s in enumerate(swipes):
            right = s.label == c["gesture"]
            note = "" if not is_swipe else ("" if right else "WRONG LABEL")
            if not is_swipe:
                note = "FALSE SWIPE in a static clip"
            print(f"{name if k == 0 else '':<26} {len(swipes) if k == 0 else '':>6}  {s.label:<12} "
                  f"{s.t_start:>7.2f} {s.t_end:>6.2f} {s.amplitude:>5.2f} {s.t_end - s.t_start:>7.2f}   {note}")
        if is_swipe:
            if len(swipes) == 1 and swipes[0].label == c["gesture"]:
                ok.append(name)
                delays.append(swipes[0].t_end - swipes[0].t_start)
            elif swipes:
                wrong.append(name)
            dips.append((name, *longest_slow_run(r, profile), old_rule_splits(r, profile)))
        else:
            static_peaks.setdefault((c["user"], c["gesture"]), []).append(float(np.nanmax(r["speed"])))
            if swipes:
                false_static.append(name)

    n_swipe = sum(c["gesture"] in tg.DYNAMIC for c in clips)
    n_static = len(clips) - n_swipe
    print(f"\nSwipe clips: {len(ok)} of {n_swipe} give exactly one swipe with their own label; "
          f"{len(missed)} missed, {len(wrong)} wrong label or more than one swipe")
    for name, reason in missed:
        print(f"   missed: {name}: {reason}")
    for name in wrong:
        print(f"   wrong: {name}")
    if delays:
        print(f"   delay from movement onset to the swipe event: median {np.median(delays):.2f} s "
              f"(range {min(delays):.2f} to {max(delays):.2f} s)")
    if any(reason == "below minimum amplitude" for _, reason in missed):
        print("   (the minimum amplitude is a calibrated value: per-user calibration (4.8) can lower it, down to "
              f"{cal.DEFAULTS['swipe_min_amplitude'] * cal.LIMITS[0]:.2f})")
    print(f"Static clips: {len(false_static)} of {n_static} give a swipe" +
          (f": {', '.join(false_static)}" if false_static else ""))
    print("   peak palm speed per user and gesture (hand sizes per second):")
    for (u, g), peaks in sorted(static_peaks.items()):
        print(f"   {u} {g:<11} {len(peaks)} clips, peaks {', '.join(f'{p:.2f}' for p in peaks)}")

    print(f"\nPause at the turning point (longest run below tau_off between the first and last frame above tau_on), "
          f"and the earlier end rule 'slow for {et.SETTLE_S} s':")
    for name, frames, seconds, split in dips:
        print(f"   {name:<24} {frames:>2} frames ({seconds:.2f} s)   'slow for {et.SETTLE_S} s' "
              f"{'CUTS the cycle at the turning point' if split else 'keeps the cycle whole'}")


if __name__ == "__main__":
    main()
