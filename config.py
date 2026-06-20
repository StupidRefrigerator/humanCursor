"""Placeholder constants for CrossFit for Data Entry (v1)."""

# --- Webcam (M1) ---
CAMERA_INDEX = 0
WINDOW_NAME = "CrossFit for Data Entry"
QUIT_KEY = "q"
SAVE_KEY = "s"
NEAR_KEY = "n"
FAR_KEY = "f"

# --- Neon green headband HSV bounds (OpenCV: H 0-179, S/V 0-255) ---
# Starting guess for a bright neon green band; tune live with trackbars in main.py.
HSV_LOWER = (20, 125, 72)
HSV_UPPER = (49, 248, 255)
MIN_CONTOUR_AREA = 500
MORPH_KERNEL = 3

# --- Signal smoothing (M5) ---
EMA_ALPHA = 0.1

# --- Axis mapping (M6) ---
TRACKING_DOT_COLOR = (203, 20, 255)  # bright neon pink (BGR)
TRACKING_DOT_RADIUS = 10
SENSITIVITY_X = 1.0
SENSITIVITY_Y = 1.0
FORWARD_IS_UP = True  # larger blob (closer) moves cursor up
CALIBRATION_COUNTDOWN_SEC = 5

# --- Squat detection (M7) ---
SQUAT_VELOCITY_THRESHOLD = 300  # pixels per second (downward = positive)
SQUAT_COOLDOWN_MS = 500
SQUAT_AREA_TOLERANCE = 0.5  # max fractional area change during squat
SQUAT_HISTORY_FRAMES = 10
SQUAT_FLASH_MS = 600
SQUAT_MOTION_VELOCITY_THRESHOLD = 40  # low vy gate: Y freeze while above, resume after calm
SQUAT_Y_UNFREEZE_FRAMES = 1 # consecutive sub-threshold frames before Y tracking resumes

# --- Cursor control (M8) ---
CURSOR_LIBRARY = "pynput"  # "pynput" or "pyautogui"
CHEAT_CLICK_KEY = " "  # Spacebar manual click fallback

# --- Paths (M4+) ---
CALIBRATION_PATH = "calibration.json"
