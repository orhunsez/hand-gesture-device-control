"""Gesture Mouse - control your PC mouse with your hand.

Keys:  q = quit     p = pause / resume the mouse control
Emergency stop: push your real mouse into a screen corner.
"""

import sys

import cv2

import config
import preprocess
import segmentation
import features
import gestures
import display
from controller import MouseController, pyautogui

WINDOW_NAME = "Gesture Mouse"
MASK_WINDOW_NAME = "Skin mask"
MAX_FAILED_READS = 30  # about one second of bad frames before we give up


# ---------- camera ----------

def camera_systems():
    """Which camera systems to try, best one first, for this operating system."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def read_frame(camera):
    """Get one picture from the camera. Returns None if the camera stopped working."""
    for _ in range(MAX_FAILED_READS):
        ok, frame = camera.read()
        if ok:
            return frame
    return None


def open_camera():
    """Turn on a webcam that really gives pictures. Returns None if none works."""
    other_ids = [i for i in range(3) if i != config.CAMERA_ID]
    for camera_id in [config.CAMERA_ID] + other_ids:
        for system in camera_systems():
            camera = cv2.VideoCapture(camera_id, system)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                if read_frame(camera) is not None:
                    if camera_id != config.CAMERA_ID:
                        print(f"Camera {config.CAMERA_ID} did not work, using camera {camera_id}.")
                    return camera
            camera.release()
    return None


# ---------- windows ----------

def can_show_windows():
    """False if OpenCV was installed without window support (opencv-python-headless)."""
    try:
        cv2.namedWindow(WINDOW_NAME)
        return True
    except cv2.error:
        return False


def window_is_open():
    """True while the main window is on the screen."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False  # some systems throw an error once the window is gone


# ---------- the main loop ----------

def run(camera):
    mouse = MouseController()
    stabilizer = gestures.GestureStabilizer(config.STABLE_FRAMES)
    paused = False
    # Not every system can tell us when the window's X was clicked.
    # We find out after the first frame is shown.
    can_check_window = None

    while True:
        frame = read_frame(camera)
        if frame is None:
            print("The camera stopped giving pictures. Close other apps that use it "
                  "(Camera, Teams, Zoom, browser) and try again.")
            return

        # 1. Preprocess
        frame, smooth_frame = preprocess.prepare(frame)

        # 2. Find the hand
        mask = segmentation.find_skin(smooth_frame)
        mask = segmentation.clean_mask(mask)
        hand = segmentation.find_hand(mask)

        # 3. Measure the hand
        finger_count = None
        center = None
        fingertip = None
        if hand is not None:
            finger_count = features.count_fingers(hand)
            center = features.get_center(hand)
            fingertip = features.get_top_point(hand)
            display.draw_hand(frame, hand, center, fingertip)

        # 4. Name the gesture, and only trust it once it stays the same for a moment
        seen_gesture = gestures.name_gesture(finger_count)
        gesture = stabilizer.update(seen_gesture)

        # 5. Move the mouse
        if paused:
            mouse.act(gestures.NO_HAND, None, None)
        else:
            mouse.act(gesture, fingertip, center)

        # 6. Show everything
        display.draw_text(frame, gesture, seen_gesture, finger_count, paused)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.imshow(MASK_WINDOW_NAME, mask)

        # 7. Check the keyboard and the window
        key = cv2.waitKey(1) & 0xFF
        if can_check_window is None:
            can_check_window = window_is_open()
        if key == ord("q"):
            return
        if can_check_window and not window_is_open():
            return
        if key == ord("p"):
            paused = not paused


def main():
    if not can_show_windows():
        print("OpenCV cannot open windows on this computer. Fix it with:\n"
              "  pip uninstall -y opencv-python-headless opencv-python\n"
              "  pip install opencv-python")
        return

    camera = open_camera()
    if camera is None:
        print("Could not open any camera. Check that it is plugged in, not used by "
              "another app, and that this terminal is allowed to use the camera "
              "(macOS: System Settings > Privacy & Security > Camera).")
        cv2.destroyAllWindows()
        return

    try:
        run(camera)
    except pyautogui.FailSafeException:
        print("Emergency stop: the mouse was pushed into a screen corner.")
    finally:
        # Always turn the camera off and close the windows, even after an error.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
