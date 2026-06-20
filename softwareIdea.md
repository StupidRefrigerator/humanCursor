# CrossFit for Data Entry — Software Idea

## 1. Project Overview & Hackathon Context

**The Event:** A three-day "Terrible Idea" hackathon where the goal is to build something technically functional but completely absurd.

**The Idea:** A hardware-free human interface system designed to "defeat office syndrome" by eliminating the mouse and keyboard entirely. Users control the cursor and execute inputs using full-body physical movements tracked via a standard webcam.

## 2. Showcase Environment & Constraints

**The Setting:** An open-air, high-traffic "marketplace/science fair" style exhibition hall.

**The Challenges:** The room will be loud, crowded, and visually chaotic, with attendees constantly walking through the background behind the user.

**The Strategy:** The product is presented as an interactive arcade game booth (e.g. controlling a simple browser game like Aim Lab or Flappy Bird). The engineering must be crowd-proof — reliable cursor tracking despite people moving through the background — to ensure consistent public demos.

## 3. Scope

**In scope for v1 (build this now):**
- Cursor movement controlled by the physical position of a single neon-colored headband, tracked via webcam.
- Left click triggered by a deep physical squat.
- A single local Python script running entirely on the demo laptop — no second device, no networking.

**Explicitly deferred (not built in v1, see Future Work):**
- Right click via shouting/screaming.
- Any phone-as-microphone or networking component.

## 4. Technical Stack (v1)

- **Python 3** — core logic.
- **OpenCV** (`opencv-python`) — webcam capture, color thresholding, contour detection.
- **PyAutoGUI** (or `pynput` as a lighter-weight alternative) — OS-level cursor movement and click injection.
- No networking libraries are needed for v1.

## 5. Interaction Mapping (v1)

- **Horizontal Cursor (X-axis):** the tracked blob's horizontal position in the camera frame, scaled to side-to-side physical stepping.
- **Vertical/Depth Cursor (Y-axis):** the tracked blob's pixel area (used as a proxy for distance from the camera), scaled to forward/backward physical stepping.
- **Left Click:** a deep squat — registered as a rapid vertical drop in the blob's frame position.

## 6. Key Challenges & Edge Cases (v1)

- **Background visual noise:** standard facial or pose tracking would fail or lag due to people moving through the background. Tracking must work without relying on face/pose models.
- **Z-axis (depth) sensitivity:** at roughly one meter from the laptop, forward/backward stepping produces only small pixel-level changes, which are hard to track accurately with a single 2D camera.
- **Squat vs. backward-movement conflict:** when the user squats, the tracked blob also moves down and changes shape/size in the frame — the system must distinguish a deliberate squat (click) from ordinary depth-axis movement to avoid misfired clicks.

## 7. Engineering Approach / Mitigations (v1)

- **Color isolation:** apply HSV color thresholding (`cv2.inRange`) to isolate the headband's color, then run contour detection (`cv2.findContours`) and select the largest contour as the tracked blob. This avoids face/pose models entirely and is inherently robust to a crowded background, since nothing else in frame matches the marker color.
- **On-site calibration:** build a small calibration utility (e.g. an HSV tuner using `cv2.createTrackbar`, or a click-to-sample-color tool) to run once at the venue before each demo session — venue lighting will differ from the dev environment, so calibration should be a required setup step, not a one-time configuration.
- **Z-axis via blob area:** use `cv2.contourArea` of the tracked contour as a stand-in for distance. Smooth the raw area readings (e.g. with an exponential moving average or simple low-pass filter) before mapping to cursor movement, to reduce jitter from natural body sway.
- **Squat vs. depth disambiguation:** start with a velocity-based debounce — if the blob's vertical position drops faster than a configurable threshold rate, classify it as a squat/click event and briefly freeze depth-axis cursor updates (roughly 200–300ms) before resuming normal tracking. Only add an area-stability check (area must stay within tolerance during the drop) on top of this if the velocity-only approach produces false positives in testing.
- **Sensitivity scalars:** apply configurable multiplier constants to both axes so that a few inches of physical movement at ~1m from the camera produces full-range cursor travel.
- **PyAutoGUI gotchas:** disable the built-in corner fail-safe (`pyautogui.FAILSAFE = False`) so cursor drift to a screen corner doesn't crash the script mid-demo. If input lag becomes noticeable, fall back to `pynput` for cursor/click control.

## 8. Future Work (Not in v1 Scope)

- **Right click via "scream":** triggered by a sudden volume spike.
- **Phone-as-microphone bridge:** a basic local webpage running on a phone held near the user's mouth, streaming audio/volume data to the laptop over a local WebSocket (`Flask-SocketIO`), to get a cleaner signal than the laptop's built-in mic in a loud room.
- **Noise-robust audio detection:** if a raw volume threshold proves unreliable even with the close-proximity phone mic, consider band-pass filtering tuned to the transient signature of a shout rather than raw amplitude.
- **Possible simpler alternative:** a second vision-based gesture (e.g. both arms raised) instead of audio for right-click, if the audio/networking path proves too fragile on venue Wi-Fi.

## 9. Explicit Non-Goals for v1

- No networking, no second device, no audio processing.
- No face or pose-estimation models (e.g. MediaPipe) — single color-blob tracking only.
