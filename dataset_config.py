"""Shared label lists for the dataset. Recorder, validator and loader all read this file.

If the team changes the gesture set, change it here and nowhere else.
"""

# Gesture name -> type.
#   static : the hand shape in a single frame decides the gesture; the user holds the pose
#            still.
#   dynamic: the motion over time decides the gesture.
#   both   : the negative class. One label, but the loader uses it for both classifiers:
#            stable frames become negatives for the static classifier, moving segments
#            become negatives for the dynamic classifier.
GESTURES = {
    "fist": "static",
    "open_palm": "static",
    "thumb_up": "static",
    "thumb_down": "static",
    "point": "static",          # index finger extended; its command is not decided yet
    "swipe_left": "dynamic",
    "swipe_right": "dynamic",
    "no_gesture": "both",
}

# no_gesture clips should contain what a user really does without meaning a command:
#   still poses   : half-closed hand, random pose, hand resting, hand leaving the picture
#   movements     : waving, reaching for the keyboard, adjusting hair, horizontal wiping,
#                   moving the open palm very slowly, stopping the palm after a swipe

# Swipes are performed with the OPEN PALM, the same hand shape as the static gesture
# open_palm. The two are told apart by motion only (brief A.12, option 1):
#   1. speed gate: the static branch is suppressed while the wrist speed is above the
#      segmentation threshold,
#   2. minimum hold time before the static open_palm counts (about half a second; the
#      other static gestures start at once, brief D.4 item 12),
#   3. cooldown after a swipe, so the palm stopping at the end of a swipe does not
#      trigger the static open_palm command.
# Record static open_palm clips with the hand still, swipe clips with the hand moving.
# A swipe is an OUT-AND-BACK movement (decided after the pilot recording: users bring the
# hand back by themselves, and the return stroke would otherwise look like a swipe in the
# opposite direction). The gesture is the whole cycle; its direction is the direction of
# the FIRST stroke:
#   swipe_left : the hand goes to the USER's left, then comes back.
#   swipe_right: the hand goes to the USER's right, then comes back.
# The saved landmarks are not mirrored. For a webcam facing the user, the user's left
# is image-right, so swipe_left means x increases first, then decreases (confirmed on
# the pilot clips: 6 of 6). The motion cue is the PALM CENTRE (mean of landmarks
# 0, 5, 9, 13, 17), not the wrist: the wrist moved only about half as far.
# Recording a swipe: wait about half a second after the countdown ends, make the
# out-and-back movement once at a steady speed, then keep the hand still.

# No command mapping for now (report section 4.10). It will be added here once the
# target device and the commands are decided.

# Clip length in seconds. In the pilot (palm centre, one person) the active out-and-back
# movement took 10 to 20 frames (0.5 to 1 s at about 20 frames per second) after a still
# lead-in of 8 to 17 frames, and fitted into 2 s in 11 of 12 clips. Swipes get 3 s to leave
# room for slower users and a longer lead-in.
DEFAULT_DURATION = 2.0
DURATIONS = {
    "swipe_left": 3.0,
    "swipe_right": 3.0,
    "no_gesture": 3.0,   # movements such as waving need room too
}

BACKGROUNDS = ["plain", "cluttered"]
LIGHTINGS = ["bright", "dim"]
DISTANCES = ["near", "far"]
