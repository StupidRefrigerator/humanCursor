# CrossFit for Data Entry — v1 Implementation Plan

This plan translates [softwareIdea.md](./softwareIdea.md) into a buildable, testable sequence for a 2-day hackathon. It assumes a single developer laptop (macOS primary dev target; Windows/Linux as secondary), one neon headband marker, and a venue demo on a separate machine that may have different lighting.

**v1 deliverable:** A local Python app that tracks a neon headband via webcam, maps blob position/area to cursor X/Y, and fires a left click on a deep squat — robust enough for a crowded exhibition hall.

---

## 1. Build Order (Milestones)

Each milestone should be runnable as a small script or mode flag, with a **visual or behavioral pass criterion** before moving on. Order is deliberate: vision pipeline first, cursor injection last (requires OS permissions and is harder to debug in parallel with vision).

| # | Milestone | What you build | Done when (verify visually) |
|---|-----------|----------------|----------------------------|
| **M0** | Project scaffold | `requirements.txt`, virtualenv, empty module layout, `config.py` with placeholder constants | `python -c "import cv2, numpy"` succeeds; folder structure exists |
| **M1** | Webcam capture loop | Open camera, display live frame, handle quit hotkey, log FPS | Window shows live video at ≥15 FPS; clean exit on `q` |
| **M2** | Raw color isolation | HSV threshold on hardcoded/default bounds; show binary mask side-by-side | Headband appears as a clean white blob on mask; background mostly black |
| **M3** | Blob detection | `findContours` → largest contour → centroid, bounding box, area | Green overlay on headband only; centroid crosshair tracks movement smoothly |
| **M4** | Calibration tool (standalone) | HSV trackbar UI **or** click-to-sample; save/load `calibration.json` | Re-tune at dev desk in <30s; saved values reload and still isolate headband |
| **M5** | Signal smoothing | EMA/low-pass on centroid X, Y, and contour area | Overlay jitter visibly reduced; numeric readout stable while standing still |
| **M6** | Axis mapping (no cursor yet) | Map blob X → logical cursor X; blob area → logical cursor Y; draw target dot on preview | Stepping left/right moves dot across preview width; stepping forward/back moves dot vertically in preview |
| **M7** | Squat detection (preview only) | Velocity-based vertical drop classifier; optional area-stability gate | Deliberate squat flashes "CLICK" on preview; slow lean-back does **not** flash click |
| **M8** | Cursor injection | `PyAutoGUI` or `pynput` wrapper; map logical coords → screen pixels; left click on squat event | Cursor follows headband in a browser window or blank desktop; squat triggers click at cursor position |
| **M9** | Full integration + demo modes | Single entrypoint; tracking-only / cursor-active modes; corner fail-safe off; kill switch | 5-minute continuous demo without crash; can pause cursor while keeping preview |
| **M10** | Venue hardening | On-site calibration ritual, sensitivity presets, "lost blob" recovery, README for booth operator | Operator doc fits on one screen; recovery from occlusion <2s |

### Suggested day split (2-day hackathon)

| Day | Target milestones | Buffer |
|-----|-------------------|--------|
| **Day 1 AM** | M0–M3 | Get blob tracking solid before any OS input |
| **Day 1 PM** | M4–M6 | Calibration + mapping; tune at ~1m distance |
| **Day 2 AM** | M7–M8 | Squat disambiguation + cursor; highest risk area |
| **Day 2 PM** | M9–M10 | Integration, crowd/lighting tests, booth polish |

---

## 2. Exact Libraries and Versions

### Recommended pin set (hackathon-stable)

Use **Python 3.12** (avoid 3.14 for now — OpenCV/NumPy ABI issues reported on bleeding-edge Python). Create a fresh venv; install in this order:

```
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install "numpy==2.0.2"
pip install "opencv-python==4.10.0.84"
pip install "pyautogui==0.9.54"
pip install "pynput==1.7.7"
```

| Package | Version | Role |
|---------|---------|------|
| `numpy` | `2.0.2` | Array ops; OpenCV dependency |
| `opencv-python` | `4.10.0.84` | Webcam, HSV, contours, calibration GUI (`createTrackbar`) |
| `pyautogui` | `0.9.54` | Primary cursor move + click (idea doc default) |
| `pynput` | `1.7.7` | Fallback if PyAutoGUI latency too high |

**Not needed for v1:** MediaPipe, Flask, SocketIO, audio libs, `opencv-python-headless` (headless lacks GUI trackbars).

### `requirements.txt` (proposed)

```
numpy==2.0.2
opencv-python==4.10.0.84
pyautogui==0.9.54
pynput==1.7.7
```

### Known installation gotchas

| Issue | Platforms | Mitigation |
|-------|-----------|------------|
| **NumPy 2.x / OpenCV ABI mismatch** | All | Pin versions above; never `pip install opencv-python` without pinning numpy first. If `import cv2` fails, wipe venv and reinstall in order. |
| **PyAutoGUI + NumPy 2 warning** | All | Known open issue (#882); usually non-fatal. PyAutoGUI is not on OpenCV's hot path — if warnings annoy or errors appear, use `pynput` for injection only. |
| **Wrong OpenCV package** | All | Do **not** install `opencv` (deprecated metapackage) alongside `opencv-python` — causes import conflicts. |
| **macOS camera permission** | macOS | System Settings → Privacy & Security → Camera → enable Terminal/IDE. First run may prompt; plan a permission check in M1. |
| **macOS accessibility permission** | macOS | System Settings → Privacy & Security → Accessibility → enable Terminal/IDE for cursor control (M8). **Must be done before venue** — booth laptop needs same step. |
| **Linux display / X11** | Linux | `opencv-python` GUI windows need `$DISPLAY`. Wayland may need `QT_QPA_PLATFORM=xcb` or XWayland. |
| **Linux `python3-tk` for PyAutoGUI** | Debian/Ubuntu | `sudo apt install python3-tk scrot` — PyAutoGUI screenshot deps. |
| **Webcam index** | All | Default `VideoCapture(0)` may be wrong on multi-camera laptops. M1 should enumerate indices 0–3 and persist `camera_index` in config. |
| **PyAutoGUI fail-safe** | All | Corner fail-safe kills the script — **must** set `FAILSAFE = False` before demo (idea doc). Document prominently; add keyboard kill switch (`q` or `Esc`). |
| **High-DPI / Retina scaling** | macOS | Screen coordinates may not match preview mapping. M8 should map via `pyautogui.size()` (or pynput equivalent) and test on the actual demo display. |

### Open question: PyAutoGUI vs pynput as primary

**Recommendation:** Implement a thin `CursorController` interface with **pynput as default** and PyAutoGUI as optional swap-in. Rationale: pynput tends to have lower latency and fewer screenshot dependencies; idea doc already lists it as fallback. **Confirm with operator before M8** which library the booth laptop will use.

---

## 3. File / Module Structure

### Proposed layout

```
humanCursor/
├── softwareIdea.md
├── softwarePlan.md
├── requirements.txt
├── README.md                    # booth operator quick-start (M10)
├── config.py                    # defaults: sensitivity, thresholds, paths
├── main.py                      # entrypoint: --mode demo|calibrate|debug
├── calibration.json             # generated at venue; gitignore
│
├── vision/
│   ├── __init__.py
│   ├── capture.py               # VideoCapture wrapper, FPS, camera index
│   ├── tracker.py               # HSV mask, contours, BlobState dataclass
│   └── calibrate.py             # standalone HSV tuner UI; save/load JSON
│
├── mapping/
│   ├── __init__.py
│   ├── smooth.py                # EMA / low-pass filters
│   ├── axes.py                  # blob → normalized → screen coords
│   └── squat.py                 # squat classifier + click cooldown
│
├── control/
│   ├── __init__.py
│   └── cursor.py                # CursorController protocol; pynput/pyautogui impl
│
└── tools/
    ├── test_background.py       # replay recorded video through tracker (M10 testing)
    └── record_session.py        # optional: save raw webcam clips for offline replay
```

### Why this structure for a 2-day hackathon

| Choice | Justification |
|--------|---------------|
| **Split `vision/`, `mapping/`, `control/`** | The three subsystems fail independently. During M1–M6 you never touch OS permissions. Squat logic (M7) stays isolated from HSV tuning (M4). |
| **`main.py` + `--mode`** | One entrypoint for the booth operator; `calibrate` mode is the venue ritual from the idea doc. Avoids "which script do I run?" confusion. |
| **`calibrate.py` runnable standalone** | `python -m vision.calibrate` works without cursor permissions — critical when debugging on a machine without accessibility access. |
| **`config.py` + `calibration.json`** | Code constants vs per-venue/per-session values. JSON holds HSV bounds and sensitivity scalars; avoids recompiling between demo sessions. |
| **No classes-heavy framework** | ~6–8 small modules, mostly functions + one `BlobState` dataclass. Fast to write, easy to gut if something doesn't work. |
| **`tools/` optional** | Background-crowd testing (see §5) without bloating the demo binary. Add only if time allows on Day 2. |

### Open question: monolith fallback

If Day 1 runs long, collapse `mapping/` and `control/` into `main.py` temporarily — but **keep `vision/tracker.py` and `vision/calibrate.py` separate**. Color isolation is the core risk; it must remain testable alone.

---

## 4. Approach per Major Component

### 4.1 Blob tracking (`vision/tracker.py`)

**Method (function-level):**

1. `load_calibration() → HsvBounds` — read lower/upper HSV from `calibration.json`.
2. `preprocess_frame(frame) → hsv` — `cv2.cvtColor(BGR2HSV)`.
3. `build_mask(hsv, bounds) → mask` — `cv2.inRange`; optional `cv2.erode`/`cv2.dilate` (small kernel) to denoise.
4. `find_blob(mask) → BlobState | None` — `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`; pick largest contour by area if area > `min_contour_area`.
5. `BlobState` fields: `centroid_x`, `centroid_y`, `area`, `bbox`, `found: bool`, `timestamp`.

**Done looks like:**

- With headband on, `found=True` at ≥20 FPS; centroid stable within ~5 px while standing still (before smoothing).
- Person walking through background in neon-free clothing does **not** produce a competing contour above `min_contour_area`.
- Brief occlusion (hand across band) recovers within 1–2 frames when band reappears.
- Preview draws contour + centroid + numeric area.

**Open question: headband color**

Idea doc says "neon-colored" but does not specify hue. **Recommendation:** Standardize on **high-saturation neon green or pink** (not orange — common booth signage). Bring **two physical headbands** (green + pink) in case venue decor clashes. Calibration per band.

---

### 4.2 Calibration tool (`vision/calibrate.py`)

**Method:**

1. Open live webcam; show original + mask side-by-side.
2. **Primary approach (idea doc):** `cv2.createTrackbar` for H low/high, S low/high, V low/high (6 trackbars).
3. **Secondary helper:** click on headband in original frame → sample 5×5 pixel HSV median → auto-set bounds ± tolerance (tunable sliders still available).
4. `save_calibration(bounds, camera_index, timestamp) → calibration.json`.
5. `load_calibration()` on app start; refuse `demo` mode if file missing (force calibrate first).

**JSON schema (proposed):**

```json
{
  "hsv_lower": [H, S, V],
  "hsv_upper": [H, S, V],
  "camera_index": 0,
  "min_contour_area": 500,
  "morph_kernel": 3
}
```

**Done looks like:**

- Operator completes calibration in under 60 seconds at the venue.
- Mask shows clean blob under fluorescent hall lighting, near window, and with phone flashlight on band (worst-case fill light test).
- Saved file reloads correctly across app restarts.

---

### 4.3 Depth / area-based Z-axis → cursor Y (`mapping/axes.py` + `mapping/smooth.py`)

**Method:**

1. **Calibrate range (interactive, part of M6):** User stands at "near" and "far" comfortable positions (~1m baseline); app records `area_near`, `area_far` (or min/max seen over 3s). Store in `calibration.json` or a separate `range.json`.
2. `smooth_area(raw_area) → filtered_area` — EMA with `alpha ≈ 0.2–0.35` (tune in M5).
3. `normalize_area(filtered_area, area_near, area_far) → y_norm ∈ [0, 1]` — linear clamp + invert if needed (larger area = closer = map to screen top or bottom per convention).
4. `map_to_screen(x_norm, y_norm, screen_w, screen_h) → (sx, sy)` — apply `sensitivity_x`, `sensitivity_y` scalars; optional dead-zone in center.
5. Similarly `centroid_x → x_norm` using frame width or calibrated left/right bounds from user stepping.

**Done looks like:**

- At ~1m from camera, **3–4 inches** of forward/back movement moves cursor from ~25% to ~75% of screen height (tunable via scalars).
- Side-to-side stepping spans full screen width with similar physical effort.
- Standing still: cursor drift < ~30 px over 10 seconds.
- Preview mode shows raw area, filtered area, and mapped target position as numbers (essential for venue tuning).

**Open question: invert Y axis?**

Forward step = cursor up vs down is a UX choice. **Recommendation:** Default **forward (larger area) = cursor up** (matches "lean toward screen to reach top"). Add `invert_y: bool` in config; confirm during M6 with a simple browser game.

---

### 4.4 Squat vs depth disambiguation (`mapping/squat.py`)

**Method (phased per idea doc):**

**Phase A — velocity-only (build this first):**

1. Track `centroid_y` history (last 5–10 frames, timestamps).
2. `vertical_velocity = Δy / Δt` (pixels per second; positive = moving down in frame).
3. If `vertical_velocity > squat_velocity_threshold` (e.g. 400–800 px/s, tune empirically) → emit `SquatEvent`.
4. On event: `cursor.click()`; enter **cooldown** 300–500 ms; **freeze Y-axis cursor updates** for 200–300 ms (idea doc); continue X tracking if desired.
5. Ignore squat detection if `found=False` or `area < min_contour_area`.

**Phase B — area-stability gate (only if Phase A false-positives):**

6. During drop, require `|Δarea| / area < area_tolerance_pct` (squat changes Y more than area; lean-back changes area more than Y).
7. Or: require drop duration < 400 ms (squat is fast; sitting/back-step is slower).

**Done looks like:**

- 10 deliberate squats → 10 clicks, zero misses, in tracking preview mode.
- 10 slow backward leans → 0 clicks.
- 10 fast backward steps → ≤1 false click (if more, enable Phase B).
- After squat, cursor Y does not jump wildly during freeze window.

**Open question: click position during squat**

When squat fires, does click land at pre-squat cursor position or mid-drop position? **Recommendation:** Click at **last known good position** (1 frame before velocity trigger) to avoid Y-drop moving cursor before click. Confirm feels right in M8 with Aim Lab or a button-click test page.

---

### 4.5 Cursor control (`control/cursor.py`)

**Method:**

1. Abstract interface: `move_to(x, y)`, `click()`, `enabled: bool`.
2. `PyAutoGUIController`: `pyautogui.FAILSAFE = False`; `moveTo` with `duration=0` or minimal.
3. `PynputController`: `mouse.position = (x, y)`; `mouse.click(Button.left)`.
4. `main.py` gates injection behind `--enable-cursor` flag or toggle key (`c`) so development doesn't fight the mouse.

**Done looks like:**

- Cursor keeps up with slow walking pace without visible stutter.
- Squat click registers in OS (test with native button or https://www.google.com/search?q=mouse+tester — use a simple HTML target locally).
- Kill switch instantly stops injection; preview keeps running.

---

## 5. Testing Strategy

### Per-milestone validation

| Milestone | Test | Pass criterion |
|-----------|------|----------------|
| M1 | Open/close camera 10× | No hang; correct camera |
| M2 | Wave neon band; have someone walk behind | Mask isolates band only |
| M3 | Spin 360°, occlude band briefly | `found` toggles predictably; largest contour is band |
| M4 | Change desk lamp angle; recalibrate | Mask usable after <60s tune |
| M5 | Stand still 30s | Filtered values within tight band |
| M6 | Step to extremes | Normalized coords hit 0 and 1 |
| M7 | 10 squats + 10 leans | Confusion matrix recorded (see §4.4) |
| M8 | Move + click in browser game | Playable, no fail-safe crash |
| M9 | 5-min soak test | No memory leak, no drift crash |
| M10 | Full booth script rehearsal | Operator completes setup solo |

### Simulating venue problems **before** the physical venue

#### A. Background crowd (visual noise)

| Technique | How |
|-----------|-----|
| **Live staged crowd** | Record 2–3 min of webcam footage with 2–3 people walking behind the wearer in street clothes (no neon). Replay via `tools/test_background.py` through `tracker.py`. |
| **Picture-in-picture chaos** | Composite a downloaded crowd-walking video (e.g. stock footage) into the background of a headband test recording using OpenCV — no ML needed, just blend for stress test. |
| **Clothing collision test** | Someone in solid bright red/green shirt walks behind user — verify **HSV bounds exclude** them. Tune saturation floor. |
| **Multi-blob contest** | Two neon objects in frame (band + water bottle). Verify **largest contour** policy; if bottle wins, raise `min_contour_area` or add "connected to top-of-frame" heuristic. **Open question** if this happens at venue — bring only one neon item in frame policy. |

#### B. Lighting variance

| Technique | How |
|-----------|-----|
| **Desk lamp sweep** | 360° lamp rotation around user; run calibrate between extremes; confirm one saved profile works or document "recalibrate when lights change". |
| **Exposure lock** | Test `cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ...)` (camera-dependent). **Open question:** If auto-exposure washes out neon, lock exposure after calibration. |
| **Window backlight** | Stand with window behind vs in front; note failure modes for README. |
| **Venue fluorescent simulation** | Use overhead cool-white LED/fluorescent if available; avoid only warm desk lamp testing. |

#### C. Squat vs lean-back (logic testing)

- Record a **labeled clip**: voice-over or clap marking "squat" vs "lean back"; run offline classifier on clip; count TP/FP/TN/FN.
- Parameter sweep script (optional): grid search `squat_velocity_threshold` and cooldown on recorded clip.

#### D. Cursor / game integration

- Local HTML file with large buttons and click counter — faster than external game for M8.
- Final demo: Aim Lab or Flappy Bird in browser fullscreen on **the same display** as venue.

#### E. Booth rehearsal checklist (M10)

1. Plug in external webcam if laptop camera angle is bad (**open question:** bring USB webcam?).
2. Run `python main.py --mode calibrate`.
3. Run `python main.py --mode demo --enable-cursor`.
4. 30-second play test; adjust `sensitivity_x/y` in config or live keys (`+`/`-`).
5. Confirm kill switch and FAILSAFE disabled.

---

## 6. Risk Flags and Mitigations

| Risk | Source (idea doc) | Likelihood | Impact | Plan mitigation |
|------|-------------------|------------|--------|-----------------|
| **Z-axis insensitivity at 1m** | §6 — small pixel area change | **High** | Cursor Y feels dead | Area-based mapping with aggressive sensitivity scalar; calibrate near/far range on-site; EMA smoothing; consider wider FOV (step back to 1.2m and zoom digitally) |
| **Squat triggers on lean-back** | §6 — Y drop + area change overlap | **High** | False clicks ruin game demo | Phase A velocity + cooldown + Y-freeze; Phase B area gate; click at pre-squat position; tune on labeled video |
| **Squat misses real clicks** | Deep squat may not drop centroid as fast as expected if knees bend out of frame | **Medium** | User frustration | Lower velocity threshold during venue tune; consider detecting **area spike + Y drop** combo; test with actual squat form early (M7) |
| **Background person in neon** | Crowded hall | **Low–Med** | Wrong blob | High-S headband; largest contour; `min_contour_area`; brief "lost tracking" hold-last-position <500ms |
| **Auto-exposure kills HSV** | Venue lighting shifts | **High** | Mask empty | On-site calibration mandatory; test exposure lock; operator recalibrate between sessions |
| **Camera permission / accessibility** | macOS gate | **Med** | Demo won't start | Document in README; complete M8 on booth laptop **day before** venue |
| **PyAutoGUI latency** | §7 | **Med** | Sluggish cursor | pynput fallback behind interface; `duration=0` moves |
| **PyAutoGUI fail-safe crash** | §7 | **Med** if ignored | Script exit mid-demo | `FAILSAFE=False` + README warning + kill switch |
| **Single 2D camera limits** | §6 | **Inherent** | No true depth | Accept proxy; tune expectations; arcade game tolerates imprecision |
| **Headband occluded by hair/hood** | Not in doc | **Med** | `found=False` drift | Hold last position short window; visual "TRACK LOST" on preview; operator asks user to expose band |
| **Wi-Fi / networking** | Deferred | N/A v1 | — | Explicitly out of scope |
| **NumPy/OpenCV install failure** | Tooling | **Med** on fresh laptop | Blocks all work | Pinned `requirements.txt`; venv; test M0 on booth machine first |

### Schedule risks

- **If M7 slips past Day 2 AM:** Ship with a **keyboard fallback click** (`Space` = click) for demo backup only — document as cheat, not v1 scope. **Open question:** Is a hidden operator key acceptable for the hackathon jury?
- **If Z-axis unusable:** Degrade to **X-only cursor + squat click** and map Y to fixed screen center — still demoable but weaker; decide at end of Day 1 during M6.

---

## 7. Open Questions and Recommendations (for review)

Please confirm or override before implementation begins:

1. **Primary cursor library:** Recommend `pynput` default; idea doc says PyAutoGUI first. Which is preferred for the demo laptop?
2. **Neon color standard:** Green vs pink headband — which will be purchased/branded?
3. **Y-axis convention:** Forward step → cursor up (recommended) or down?
4. **Click position on squat:** Pre-squat cursor position (recommended) or live position?
5. **Exposure lock:** Attempt camera exposure lock in M1, or rely purely on HSV recalibration?
6. **External USB webcam:** Bring one for better FOV/tripod mount, or laptop camera only?
7. **Demo game:** Aim Lab, Flappy Bird, or custom local HTML target — affects acceptable latency and click precision.
8. **Hidden operator fallback:** Is a secret keyboard key for click acceptable if squat detection isn't reliable in time?
9. **Y-only degradation:** If Z-axis tuning fails on Day 1, is X-only + squat acceptable for judging, or is full 2D movement required?
10. **Multi-neon collision:** Strict "only headband in frame" booth rule vs implementing secondary blob heuristics?

---

## 8. Implementation Sequence Summary (quick reference)

```
M0 scaffold
  → M1 webcam
    → M2 HSV mask
      → M3 contours / BlobState
        → M4 calibration tool + JSON
          → M5 smoothing
            → M6 axis mapping (preview)
              → M7 squat detect (preview)
                → M8 cursor injection
                  → M9 integrated demo
                    → M10 venue hardening + crowd/lighting replay tests
```

**Do not start cursor control (M8) until M4 calibration and M7 squat preview pass on recorded crowd footage.**

---

*Plan derived from [softwareIdea.md](./softwareIdea.md). No code implemented yet — awaiting review of §7 open questions.*
