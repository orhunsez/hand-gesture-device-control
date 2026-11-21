"""Landmark normalisation (report section 4.3).

The same functions must be used when the training data is built and when the live
system runs, otherwise training and real use would not match.

Input  : MediaPipe landmarks, shape (21, 3) for one frame or (T, 21, 3) for a clip,
         columns x, y, z, exactly as stored in the JSON files (raw, not mirrored).
         Frames without a hand are NaN (the loader turns the JSON "null" into NaN).
Output : the same shape. For every frame
             q_i = (p_i - p_0) / s
         where p_0 is the wrist and s = || p_9 - p_0 || is the hand size (wrist to the
         base of the middle finger, measured in the image plane).
         After this the wrist is the origin and the hand size is 1, so the numbers no
         longer depend on where the hand is in the picture (translation) or how far it
         is from the camera (scale).

What it does NOT remove: rotation of the hand in the image plane, and the viewing angle.
Those stay in the data on purpose (thumb up and thumb down differ by orientation).

Aspect ratio: MediaPipe divides x by the image width and y by the image height, so a
step of 0.1 in x and 0.1 in y are different lengths on a 4:3 camera. Passing the image
size converts x, y (and z, which MediaPipe gives in about the same scale as x) to
pixel-like units first, so that distances are measured equally in both directions.
Clips recorded before the image size was stored are assumed to be 640 x 480, which is
what the development camera gives.
"""
import numpy as np

WRIST = 0
MIDDLE_MCP = 9            # base of the middle finger
DEFAULT_SIZE = (640, 480)  # (width, height) assumed for clips without a stored size
EPS = 1e-6


def to_pixel_units(landmarks, width=None, height=None):
    """Multiply x and z by the image width and y by the image height."""
    lm = np.asarray(landmarks, dtype=np.float64)
    width, height = (width, height) if width and height else DEFAULT_SIZE
    return lm * np.array([width, height, width], dtype=np.float64)


def hand_size(landmarks, width=None, height=None):
    """Wrist to middle-finger-base distance in the image plane, per frame.

    Returns a float for one frame (21, 3) or an array (T,) for a clip.
    NaN where the frame has no hand or the hand is degenerate (size almost zero).
    """
    lm = to_pixel_units(landmarks, width, height)
    s = np.linalg.norm(lm[..., MIDDLE_MCP, :2] - lm[..., WRIST, :2], axis=-1)
    return np.where(s < EPS, np.nan, s)


def normalize(landmarks, width=None, height=None):
    """Return landmarks relative to the wrist and divided by the hand size."""
    lm = to_pixel_units(landmarks, width, height)
    s = hand_size(landmarks, width, height)
    q = lm - lm[..., WRIST:WRIST + 1, :]
    return q / np.asarray(s)[..., None, None]


def flatten(normalized, drop_wrist=False, keep_z=True):
    """Turn normalised landmarks into feature vectors, one row per frame.

    drop_wrist: the wrist is always (0, 0, 0) after normalisation and carries no
                information, so it can be left out (20 points instead of 21).
    keep_z    : False leaves out the depth column (z is the least reliable).
    Shapes: (21, 3) -> (n,)   (T, 21, 3) -> (T, n)   with n = points * columns.
    """
    q = np.asarray(normalized)
    if drop_wrist:
        q = q[..., 1:, :]
    if not keep_z:
        q = q[..., :2]
    return q.reshape(*q.shape[:-2], -1)
