"""All the numbers you can tweak live here, so nobody hunts through the code."""

# --- Camera ---
CAMERA_ID = 0          # if the wrong camera opens, try 1 or 2
FRAME_WIDTH = 640      # every camera picture is cut and resized to this size,
FRAME_HEIGHT = 480     # so all the numbers below work the same on every laptop

# --- Preprocessing ---
BLUR_SIZE = 7  # must be an odd number. Bigger = smoother picture.

# --- Skin colour range in YCrCb (lower, upper) ---
# If your hand is not found, change these numbers!
SKIN_LOWER = (0, 133, 77)
SKIN_UPPER = (255, 173, 127)

# --- Cleaning the black-and-white mask ---
MORPH_KERNEL_SIZE = 5

# --- Hand rules ---
MIN_HAND_AREA = 5000      # ignore blobs smaller than this (pixels)
MIN_GAP_DEPTH = 20        # how deep a gap between fingers must be (pixels)
MAX_GAP_ANGLE = 90        # gaps sharper than this (degrees) count as "between fingers"
FIST_SOLIDITY = 0.85      # a very "filled" hand shape is a fist

# --- Gesture stability ---
STABLE_FRAMES = 5         # a gesture must be seen this many frames in a row before we act on it

# --- Mouse ---
EDGE_MARGIN = 100         # unused border of the camera image (lets you reach screen corners)
SMOOTHING = 5             # bigger = slower but steadier cursor
CLICK_COOLDOWN = 0.6      # seconds to wait between clicks
SCROLL_STEP = 15          # how many pixels the hand must move for one scroll-wheel step
