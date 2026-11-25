# 4.1 System Overview (draft text with measured numbers)

All numbers are measured on the pilot clips of one hand (41 clips, 1,653 frames, plain background, bright light,
near distance) and are provisional. Figures: `docs/figures/fig_pipeline.png` and `docs/figures/fig_stages.png`
(made by `tools/make_report_figures.py`). Timings are for the development machine and cover only the stages after MediaPipe.

## Text

The system turns a continuous video stream into a recognised gesture through a chain of stages, and every stage
replaces its input with a smaller and more meaningful representation (Figure 1, Table 1). The decisive design choice
is made at the second stage. Instead of classifying pixels, the system reduces each camera frame to the 21 hand
landmarks that MediaPipe Hands estimates. Methods that threshold skin colour or follow contours depend on the
intensity of the image, and therefore on lighting, background and skin tone, which are exactly the factors this
project has to evaluate. Landmark coordinates describe the geometry of the hand. Everything after this stage works
on 63 numbers per frame (21 points with x, y and z) instead of the 921,600 values of a 640 x 480 colour image.

The first four stages are shared by every gesture and run once per camera frame. The webcam delivers 19.5 frames per
second, so one frame lasts 51 ms. MediaPipe Hands finds the hand with a palm detector and then estimates the landmarks
with a second network; both are pretrained and are not part of our contribution. In the pilot data it found a hand in
all 1,653 recorded frames. The landmarks jitter even when the hand is still, so they are smoothed with a 1 euro filter,
which lowers its cut-off frequency when the hand moves slowly and raises it when the hand moves fast. On the pilot
clips it reduced the landmark jitter by 84 percent and the speed noise of a still hand by 68 percent (from 0.19 to 0.06
hand sizes per second), at the price of 27 ms of added delay, about half a frame (Figure 2c). The smoothing is applied
to the raw landmarks before the normalisation, because the hand size is estimated from two noisy landmarks and
normalising first would copy its fluctuation into all 63 numbers. The normalisation then moves the wrist to the origin
and divides all coordinates by the hand size. In the recorded clips the hand size lies between 69 and 150 pixels, a
factor of 2.2, so the raw coordinates of one and the same gesture differ strongly between clips (Figure 2a); after the
normalisation the hands lie on top of each other (Figure 2b). The orientation of the hand is deliberately kept,
because thumb up and thumb down differ only by it.

After the normalisation the chain splits into two branches, because the two kinds of gestures are defined by different
information. A static gesture (fist, open palm, thumb up, thumb down, point) is defined by the hand shape in a single
frame. The static branch classifies every frame and accepts a gesture only when the same class is held while the hand
is still. A dynamic gesture (swipe left, swipe right) is defined by a movement over time. In the pilot clips such a
movement, an outward stroke followed by the return of the hand, lasted 10 to 20 frames (0.5 to 1 s). The dynamic branch
therefore detects the movement from the speed of the palm centre and classifies the whole window. The open palm is both
a static gesture and the hand shape of the swipes, so the two branches are told apart by movement: the static branch is
silenced while the hand moves fast and for about one second after a swipe. The outputs of both branches pass through
the false-trigger prevention, which decides whether a recognised candidate becomes a command (confidence threshold,
hold time, cooldown and a no-gesture class). Calibration is not a stage of the per-frame chain; it is a short
measurement at start-up that sets the hand scale and the thresholds of the later stages. The mapping to device commands
is postponed until the target device is decided.

The stages after MediaPipe are cheap. Smoothing takes 16 microseconds per frame, normalisation 13 microseconds and
the static classifier 169 microseconds, 0.20 ms in total, which is 0.4 percent of the 51 ms available per frame.
The time of MediaPipe itself and the end-to-end latency have not been measured yet (experiment E9). Figure 1 shows which
stages are implemented and tested (camera to normalisation), which have a first pilot on one hand (static branch) and
which are still planned (dynamic branch, calibration, false-trigger prevention, command mapping).

## Table 1 (representation at each stage)

| Stage | Representation per frame | Size |
|---|---|---|
| Camera frame | RGB image | 640 x 480 x 3 = 921,600 values |
| MediaPipe landmarks | 21 points, x, y, z (image units) | 63 numbers |
| Smoothing | 21 points, less jitter | 63 numbers |
| Normalisation | 21 points, wrist at the origin, hand size 1 | 63 numbers |
| Static branch | gesture class | 1 label |
| Dynamic branch | window of the movement | 10 to 20 frames x 63 numbers |

## Captions

Figure 1. Processing pipeline. Each box shows the data and one measured number of its stage. Blue: implemented and
tested; orange: first pilot on one hand; dashed: planned. Measured on 41 pilot clips of one hand (provisional).

Figure 2. Real data from the pilot clips. (a) Raw landmarks of an open palm at the smallest (69 px) and at the largest
(150 px) hand size in the recordings. (b) The same two hands after the normalisation: wrist at the origin, hand size 1.
(c) Palm-centre speed of a still open palm before and after the 1 euro filter; the 95th percentile falls from 0.095
to 0.018 hand sizes per second.
