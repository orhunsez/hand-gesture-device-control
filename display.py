"""Draw helpful things on the screen so we can see what the computer sees."""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW = (0, 255, 255)
GREY = (200, 200, 200)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def draw_hand(frame, contour, center, fingertip):
    """Outline the hand, mark its centre and fingertip."""
    cv2.drawContours(frame, [contour], -1, GREEN, 2)
    cv2.circle(frame, center, 8, BLUE, -1)      # blue = centre
    cv2.circle(frame, fingertip, 10, RED, -1)   # red = fingertip


def draw_text(frame, gesture, seen_gesture, finger_count, paused):
    """Write what we see on the top-left of the picture.

    gesture      = the steady gesture the mouse follows
    seen_gesture = what this single frame looks like (can flicker)
    """
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), FONT, 0.8, YELLOW, 2)
    cv2.putText(frame, f"This frame: {seen_gesture}  Fingers: {finger_count}",
                (10, 60), FONT, 0.6, GREY, 2)
    if paused:
        cv2.putText(frame, "PAUSED (press p)", (10, 90), FONT, 0.8, RED, 2)
