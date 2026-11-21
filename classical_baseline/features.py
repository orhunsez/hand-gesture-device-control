"""Step 3: measure things about the hand (fingers, centre, fingertip)."""

import math
import cv2
import config


def get_center(contour):
    """Middle point (x, y) of the hand."""
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)


def get_top_point(contour):
    """Highest point of the hand. When one finger is up, this is the fingertip."""
    top_index = contour[:, :, 1].argmin()
    x, y = contour[top_index][0]
    return (int(x), int(y))


def get_solidity(contour):
    """How 'filled' the hand is: hand area divided by hull area. Fist = close to 1."""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return cv2.contourArea(contour) / hull_area


def _angle_at_gap(start, end, far):
    """Angle (degrees) of the 'V' shape formed at the bottom of a gap."""
    a = math.dist(start, end)
    b = math.dist(start, far)
    c = math.dist(end, far)
    if b == 0 or c == 0:
        return 180
    cos_value = (b * b + c * c - a * a) / (2 * b * c)
    cos_value = max(-1, min(1, cos_value))
    return math.degrees(math.acos(cos_value))


def count_gaps(contour):
    """Count the V-shaped gaps between fingers (convexity defects)."""
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return 0

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        # OpenCV sometimes refuses twisted outlines. Treat that as "no gaps".
        return 0
    if defects is None:
        return 0

    # OpenCV 4 gives the shape (N, 1, 4), OpenCV 5 gives (N, 4).
    # reshape turns both into a simple list of rows with 4 numbers.
    defects = defects.reshape(-1, 4)

    gaps = 0
    for start_i, end_i, far_i, depth in defects:
        depth = depth / 256.0  # OpenCV stores depth multiplied by 256
        start = tuple(contour[start_i][0])
        end = tuple(contour[end_i][0])
        far = tuple(contour[far_i][0])

        deep_enough = depth > config.MIN_GAP_DEPTH
        sharp_enough = _angle_at_gap(start, end, far) < config.MAX_GAP_ANGLE
        if deep_enough and sharp_enough:
            gaps += 1
    return gaps


def count_fingers(contour):
    """Guess how many fingers are up (0 to 5)."""
    gaps = count_gaps(contour)
    if gaps == 0:
        # No gaps: either a fist (0 fingers) or a single pointing finger (1 finger).
        if get_solidity(contour) > config.FIST_SOLIDITY:
            return 0
        return 1
    return min(gaps + 1, 5)
