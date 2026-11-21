"""Step 1: make the camera picture nicer before we look at it."""

import cv2
import config


def fit_to_size(frame):
    """Make every camera picture exactly FRAME_WIDTH x FRAME_HEIGHT.

    Cameras give different sizes (640x480, 1280x720, ...). We cut off the
    extra sides so the shape matches, then shrink it. Nothing gets squashed.
    """
    height, width = frame.shape[:2]
    wanted_ratio = config.FRAME_WIDTH / config.FRAME_HEIGHT

    if width / height > wanted_ratio:
        # Too wide: cut equal strips from the left and right.
        new_width = int(height * wanted_ratio)
        left = (width - new_width) // 2
        frame = frame[:, left:left + new_width]
    else:
        # Too tall: cut equal strips from the top and bottom.
        new_height = int(width / wanted_ratio)
        top = (height - new_height) // 2
        frame = frame[top:top + new_height, :]

    return cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))


def mirror(frame):
    """Flip left-right, so moving your hand right moves the cursor right."""
    return cv2.flip(frame, 1)


def blur(frame):
    """Smooth out tiny specks of noise (spatial filtering)."""
    return cv2.GaussianBlur(frame, (config.BLUR_SIZE, config.BLUR_SIZE), 0)


def prepare(frame):
    """Do all preprocessing steps in order."""
    frame = fit_to_size(frame)
    frame = mirror(frame)
    smooth_frame = blur(frame)
    return frame, smooth_frame
