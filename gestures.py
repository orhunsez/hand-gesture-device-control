"""Step 4: turn a finger count into a gesture name."""

FIST = "FIST"
POINT = "POINT"
PEACE = "PEACE"
OPEN_HAND = "OPEN_HAND"
UNKNOWN = "UNKNOWN"
NO_HAND = "NO_HAND"

# TODO (team): add a THREE_FINGERS gesture for volume control.
FINGERS_TO_GESTURE = {
    0: FIST,
    1: POINT,
    2: PEACE,
    5: OPEN_HAND,
}


def name_gesture(finger_count):
    """Return the gesture name for a number of fingers."""
    if finger_count is None:
        return NO_HAND
    if finger_count == 4:
        return OPEN_HAND  # a hand with 4 fingers seen is almost always open
    return FINGERS_TO_GESTURE.get(finger_count, UNKNOWN)


class GestureStabilizer:
    """Only believe a gesture after seeing it several frames in a row.

    The finger count jumps around a little from frame to frame. Without this,
    one wrong frame could cause a click you did not want.
    """

    def __init__(self, frames_needed):
        self.frames_needed = frames_needed
        self.candidate = NO_HAND   # the gesture we are currently seeing
        self.times_seen = 0        # how many frames in a row we have seen it
        self.stable = NO_HAND      # the gesture we trust

    def update(self, gesture):
        """Give it the newest gesture, get back the gesture we trust."""
        if gesture == self.candidate:
            self.times_seen += 1
        else:
            self.candidate = gesture
            self.times_seen = 1

        # A missing hand is trusted right away, so an old gesture
        # (like FIST) can never "come back" and click when the hand returns.
        if gesture == NO_HAND or self.times_seen >= self.frames_needed:
            self.stable = gesture
        return self.stable
