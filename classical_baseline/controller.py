"""Step 5: do real mouse actions."""

import os
import sys
import time

import numpy as np

import config
import gestures

try:
    import pyautogui
except Exception as error:  # on Linux this fails when there is no X11 screen
    sys.exit(
        f"Could not start pyautogui ({error}).\n"
        "On Linux, log in with an 'Xorg' / 'X11' session instead of Wayland "
        "and run this from a normal desktop terminal."
    )

if sys.platform.startswith("linux") and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    print("Warning: you are on Wayland. The mouse may not move. "
          "Log out and pick an 'Xorg' / 'X11' session on the login screen.")

pyautogui.PAUSE = 0  # no built-in delay, we want a fast cursor
# Safety: slam the mouse into a screen corner to stop the program.
pyautogui.FAILSAFE = True

# One step of the scroll wheel. Windows counts 120 units per step,
# macOS and Linux count 1 per step.
WHEEL_STEP = 120 if sys.platform == "win32" else 1


class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.cursor_x, self.cursor_y = pyautogui.position()
        self.last_click_time = 0
        self.last_gesture = gestures.NO_HAND
        self.last_scroll_y = None

    # ---------- small actions ----------

    def move_cursor(self, point):
        """Move the cursor to where the fingertip is, smoothly."""
        x, y = point
        margin = config.EDGE_MARGIN
        # Stay 1 pixel away from the screen edges: touching an exact corner
        # would trigger the pyautogui emergency stop by accident.
        right = self.screen_w - 2
        bottom = self.screen_h - 2
        target_x = np.interp(x, [margin, config.FRAME_WIDTH - margin], [1, right])
        target_y = np.interp(y, [margin, config.FRAME_HEIGHT - margin], [1, bottom])

        self.cursor_x += (target_x - self.cursor_x) / config.SMOOTHING
        self.cursor_y += (target_y - self.cursor_y) / config.SMOOTHING
        pyautogui.moveTo(self.cursor_x, self.cursor_y)

    def click(self):
        """Click, but not too often."""
        now = time.time()
        if now - self.last_click_time > config.CLICK_COOLDOWN:
            pyautogui.click()
            self.last_click_time = now

    def scroll(self, hand_y):
        """Scroll one wheel step for every SCROLL_STEP pixels the hand moves."""
        if self.last_scroll_y is None:
            self.last_scroll_y = hand_y  # first frame: just remember where the hand is
            return

        moved = self.last_scroll_y - hand_y  # hand goes up -> page goes up
        steps = int(moved / config.SCROLL_STEP)
        if steps != 0:
            pyautogui.scroll(steps * WHEEL_STEP)
            self.last_scroll_y -= steps * config.SCROLL_STEP

    # ---------- the main decision ----------

    def act(self, gesture, fingertip, center):
        """Look at the gesture and do the matching mouse action."""
        if gesture == gestures.POINT:
            self.move_cursor(fingertip)

        elif gesture == gestures.FIST:
            # Click only at the moment the hand BECOMES a fist, not while it stays one.
            if self.last_gesture != gestures.FIST:
                self.click()

        elif gesture == gestures.PEACE:
            self.scroll(center[1])

        # OPEN_HAND, UNKNOWN, NO_HAND: do nothing (this is the "pause" pose)

        if gesture != gestures.PEACE:
            self.last_scroll_y = None  # forget scroll position when we stop scrolling
        self.last_gesture = gesture
