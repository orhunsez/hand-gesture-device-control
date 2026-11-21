"""Step 2: find the hand in the picture."""

import cv2
import numpy as np
import config


def find_skin(frame):
    """Return a black-and-white picture: white = skin, black = everything else."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array(config.SKIN_LOWER, dtype=np.uint8)
    upper = np.array(config.SKIN_UPPER, dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def clean_mask(mask):
    """Remove white specks (open) and fill black holes (close)."""
    size = config.MORPH_KERNEL_SIZE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_hand(mask):
    """Return the outline of the biggest white blob, or None if there is no hand."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < config.MIN_HAND_AREA:
        return None
    return biggest
