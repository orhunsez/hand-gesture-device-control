"""Measure what the 1-euro filter does to recorded clips (experiment E2) and tune it.

    .venv\\Scripts\\python.exe evaluate_smoothing.py

Metrics (all on recorded landmarks, the same clips with and without the filter):
  jitter       RMS of the landmark noise on still-hand clips, in hand sizes. The noise is
               the difference between each landmark and its 5-frame moving average.
  speed noise  95th percentile of the palm-centre speed on still-hand clips, in hand sizes
               per second. This is the number the segmentation threshold (4.5) has to stay
               above, so lowering it makes motion detection easier.
  delay        how much later the movement appears in the smoothed clip, in milliseconds,
               measured on swipe clips as the shift of the time centre of the palm-centre
               speed. This is the lag the filter adds.
  amplitude    peak-to-peak palm-centre x of the smoothed clip divided by the raw one on
               swipe clips; below 1 means the filter flattens the movement.

Tuning order from the 1-euro paper: beta = 0 and f_cmin first (jitter), then beta (lag).
The choice below is PROVISIONAL: pilot clips of one person only (user u99). Repeat it on
the training users before using the numbers in the report.
"""
import numpy as np

import landmark_features as lf
import landmark_smoothing as ls
import view_clip

PALM = [0, 5, 9, 13, 17]
STILL = [("u99/fist.json"), ("u99/open_palm.json"), ("u99/thumb_up.json")]
SWIPE = [("u99/swipe_left.json"), ("u99/swipe_right.json")]
W, H = lf.DEFAULT_SIZE


def clips(files):
    for f in files:
        for k in range(1, len(view_clip.read_file(f)["clips"]) + 1):
            lm, t, meta = view_clip.load_clip(f, k)
            yield f, k, lm.astype(np.float64), t.astype(np.float64), meta


def smoothed(lm, t, meta, f_cmin, beta):
    w, h = meta.get("image_width"), meta.get("image_height")
    return ls.smooth_clip(lm, t, w, h, f_cmin=f_cmin, beta=beta)


def jitter(lm, meta, win=5):
    w, h = meta.get("image_width"), meta.get("image_height")
    px = lf.to_pixel_units(lm, w, h)
    s = np.nanmedian(lf.hand_size(lm, w, h))
    kern = np.ones(win) / win
    res = []
    for li in range(21):
        for c in (0, 1):
            x = px[:, li, c]
            ma = np.convolve(x, kern, mode="valid")
            res.append(x[win // 2:win // 2 + len(ma)] - ma)
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2)) / s)


def palm_speed(lm, t, meta):
    w, h = meta.get("image_width"), meta.get("image_height")
    px = lf.to_pixel_units(lm, w, h)
    s = np.nanmedian(lf.hand_size(lm, w, h))
    c = np.nanmean(px[:, PALM, :2], axis=1)
    v = np.linalg.norm(np.diff(c, axis=0), axis=1) / np.diff(t) / s
    return v, (t[1:] + t[:-1]) / 2


def palm_x(lm, meta):
    w, h = meta.get("image_width"), meta.get("image_height")
    px = lf.to_pixel_units(lm, w, h)
    return np.nanmean(px[:, PALM, 0], axis=1) / np.nanmedian(lf.hand_size(lm, w, h))


def time_centre(v, tm):
    return float(np.sum(tm * v) / np.sum(v))


def evaluate(f_cmin, beta, raw):
    jr, sr, dl, am = [], [], [], []
    for (f, k, lm, t, meta), (rj, rsp) in zip(raw["still"], raw["still_stats"]):
        sm = smoothed(lm, t, meta, f_cmin, beta)
        jr.append(jitter(sm, meta) / rj)
        sr.append(palm_speed(sm, t, meta)[0])
    for (f, k, lm, t, meta), (rc, rx) in zip(raw["swipe"], raw["swipe_stats"]):
        sm = smoothed(lm, t, meta, f_cmin, beta)
        v, tm = palm_speed(sm, t, meta)
        dl.append((time_centre(v, tm) - rc) * 1000.0)
        am.append(np.ptp(palm_x(sm, meta)) / rx)
    speed_ratio = np.percentile(np.concatenate(sr), 95) / raw["speed_p95"]
    return (1 - np.median(jr)) * 100, (1 - speed_ratio) * 100, float(np.mean(dl)), float(np.mean(am))


def main():
    still = list(clips(STILL))
    swipe = list(clips(SWIPE))
    raw = {"still": still, "swipe": swipe}
    raw["still_stats"] = [(jitter(lm, meta), None) for f, k, lm, t, meta in still]
    speeds = [palm_speed(lm, t, meta)[0] for f, k, lm, t, meta in still]
    raw["speed_p95"] = float(np.percentile(np.concatenate(speeds), 95))
    raw["swipe_stats"] = []
    for f, k, lm, t, meta in swipe:
        v, tm = palm_speed(lm, t, meta)
        raw["swipe_stats"].append((time_centre(v, tm), np.ptp(palm_x(lm, meta))))

    print(f"Still clips: {len(still)} (fist, open palm, thumb up); swipe clips: {len(swipe)} (left, right); user u99")
    print(f"Without the filter: median jitter {np.median([s[0] for s in raw['still_stats']]):.4f} hand sizes, "
          f"speed noise (95th percentile) {raw['speed_p95']:.3f} hand sizes/s\n")
    print("f_cmin   beta | jitter -%  speed noise -%  delay ms  amplitude")
    rows = []
    for f_cmin in (0.5, 1.0, 2.0, 3.0, 5.0):
        for beta in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3):
            j, s, d, a = evaluate(f_cmin, beta, raw)
            rows.append((f_cmin, beta, j, s, d, a))
            print(f"{f_cmin:>5.1f} {beta:>7.3f} | {j:>9.1f} {s:>14.1f} {d:>9.1f} {a:>10.3f}")

    # Provisional choice: the largest drop in speed noise with at most 30 ms delay
    # (about half a frame at 20 frames per second) and at most 3 percent flattening.
    ok = [r for r in rows if r[4] <= 30.0 and r[5] >= 0.97]
    print("\nChoice rule: delay <= 30 ms and amplitude >= 0.97, then the largest drop in speed noise.")
    if ok:
        best = max(ok, key=lambda r: r[3])
        print(f"Provisional choice: f_cmin = {best[0]}, beta = {best[1]}  ->  jitter -{best[2]:.1f} %, "
              f"speed noise -{best[3]:.1f} %, delay {best[4]:.1f} ms, amplitude {best[5]:.3f}")
    else:
        print("No setting met the rule.")


if __name__ == "__main__":
    main()
