"""M2–M6: HSV blob tracking, calibration, smoothing, and axis mapping preview."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class Blob:
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float


@dataclass
class TrackbarHsv:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    @classmethod
    def from_bounds(
        cls,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> TrackbarHsv:
        return cls(
            h_min=lower[0],
            h_max=upper[0],
            s_min=lower[1],
            s_max=upper[1],
            v_min=lower[2],
            v_max=upper[2],
        )

    def lower(self) -> tuple[int, int, int]:
        return (self.h_min, self.s_min, self.v_min)

    def upper(self) -> tuple[int, int, int]:
        return (self.h_max, self.s_max, self.v_max)


@dataclass
class CalibrationData:
    hsv: TrackbarHsv
    area_near: float | None = None  # max smoothed area (closer / forward)
    area_far: float | None = None   # min smoothed area (farther / back)


@dataclass
class SmoothedBlob:
    centroid: tuple[float, float]
    area: float


@dataclass
class PendingAreaCalibration:
    kind: str  # "near" or "far"
    deadline: float
    last_announced_second: int | None = None


class EmaFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, raw: float) -> float:
        if self.value is None:
            self.value = raw
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class BlobSmoother:
    def __init__(self, alpha: float) -> None:
        self._cx = EmaFilter(alpha)
        self._cy = EmaFilter(alpha)
        self._area = EmaFilter(alpha)

    def update(self, blob: Blob) -> SmoothedBlob:
        cx = self._cx.update(float(blob.centroid[0]))
        cy = self._cy.update(float(blob.centroid[1]))
        area = self._area.update(blob.area)
        return SmoothedBlob(centroid=(cx, cy), area=area)

    def reset(self) -> None:
        self._cx.reset()
        self._cy.reset()
        self._area.reset()


def open_camera(preferred_index: int) -> cv2.VideoCapture | None:
    """Try preferred camera index, then fall back through 0–3."""
    candidates = [preferred_index] + [i for i in range(4) if i != preferred_index]
    for index in candidates:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"Opened camera index {index}")
            return cap
        cap.release()
    return None


def load_calibration(path: str | Path = config.CALIBRATION_PATH) -> CalibrationData:
    """Load HSV and near/far area baselines from calibration.json, or use config defaults."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        print(f"No {calibration_path.name} found — using config.py HSV defaults.")
        return CalibrationData(hsv=TrackbarHsv.from_bounds(config.HSV_LOWER, config.HSV_UPPER))

    with calibration_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if all(key in data for key in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")):
        hsv = TrackbarHsv(
            h_min=int(data["h_min"]),
            h_max=int(data["h_max"]),
            s_min=int(data["s_min"]),
            s_max=int(data["s_max"]),
            v_min=int(data["v_min"]),
            v_max=int(data["v_max"]),
        )
    elif "hsv_lower" in data and "hsv_upper" in data:
        hsv = TrackbarHsv.from_bounds(
            tuple(data["hsv_lower"]),
            tuple(data["hsv_upper"]),
        )
    else:
        print(
            f"{calibration_path.name} is missing HSV fields — using config.py defaults.",
            file=sys.stderr,
        )
        hsv = TrackbarHsv.from_bounds(config.HSV_LOWER, config.HSV_UPPER)

    area_near = data.get("area_near")
    area_far = data.get("area_far")
    if area_near is not None:
        area_near = float(area_near)
    if area_far is not None:
        area_far = float(area_far)

    print(f"Loaded calibration from {calibration_path.name}")
    if area_near is not None:
        print(f"  area_near (max/closer): {area_near:.0f} px")
    if area_far is not None:
        print(f"  area_far (min/back): {area_far:.0f} px")

    return CalibrationData(hsv=hsv, area_near=area_near, area_far=area_far)


def persist_calibration(
    window: str,
    area_near: float | None,
    area_far: float | None,
    path: str | Path = config.CALIBRATION_PATH,
) -> dict:
    trackbars = TrackbarHsv(
        h_min=cv2.getTrackbarPos("H min", window),
        h_max=cv2.getTrackbarPos("H max", window),
        s_min=cv2.getTrackbarPos("S min", window),
        s_max=cv2.getTrackbarPos("S max", window),
        v_min=cv2.getTrackbarPos("V min", window),
        v_max=cv2.getTrackbarPos("V max", window),
    )
    lower, upper = get_hsv_bounds(window)
    payload: dict = {
        "h_min": trackbars.h_min,
        "h_max": trackbars.h_max,
        "s_min": trackbars.s_min,
        "s_max": trackbars.s_max,
        "v_min": trackbars.v_min,
        "v_max": trackbars.v_max,
        "hsv_lower": list(lower),
        "hsv_upper": list(upper),
        "camera_index": config.CAMERA_INDEX,
        "min_contour_area": config.MIN_CONTOUR_AREA,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if area_near is not None:
        payload["area_near"] = area_near
    if area_far is not None:
        payload["area_far"] = area_far

    calibration_path = Path(path)
    with calibration_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def save_hsv_calibration(
    window: str,
    area_near: float | None,
    area_far: float | None,
) -> None:
    payload = persist_calibration(window, area_near, area_far)
    print(f"Saved calibration to {config.CALIBRATION_PATH}")
    print(f"  HSV_LOWER = {tuple(payload['hsv_lower'])}")
    print(f"  HSV_UPPER = {tuple(payload['hsv_upper'])}")


def save_near_baseline(
    window: str,
    smoothed_area: float,
    area_far: float | None,
) -> float:
    persist_calibration(window, smoothed_area, area_far)
    print(
        f"Saved area_near (max/closer) = {smoothed_area:.0f} px "
        f"to {config.CALIBRATION_PATH}"
    )
    return smoothed_area


def save_far_baseline(
    window: str,
    smoothed_area: float,
    area_near: float | None,
) -> float:
    persist_calibration(window, area_near, smoothed_area)
    print(
        f"Saved area_far (min/back) = {smoothed_area:.0f} px "
        f"to {config.CALIBRATION_PATH}"
    )
    return smoothed_area


def load_trackbar_hsv(path: str | Path = config.CALIBRATION_PATH) -> TrackbarHsv:
    return load_calibration(path).hsv


def save_calibration(window: str, path: str | Path = config.CALIBRATION_PATH) -> None:
    """Backward-compatible HSV save; preserves any existing near/far values."""
    existing = load_calibration(path)
    save_hsv_calibration(window, existing.area_near, existing.area_far)


def get_hsv_bounds(window: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    h_lo = cv2.getTrackbarPos("H min", window)
    h_hi = cv2.getTrackbarPos("H max", window)
    s_lo = cv2.getTrackbarPos("S min", window)
    s_hi = cv2.getTrackbarPos("S max", window)
    v_lo = cv2.getTrackbarPos("V min", window)
    v_hi = cv2.getTrackbarPos("V max", window)
    lower = (min(h_lo, h_hi), min(s_lo, s_hi), min(v_lo, v_hi))
    upper = (max(h_lo, h_hi), max(s_lo, s_hi), max(v_lo, v_hi))
    return lower, upper


def print_hsv_bounds(window: str) -> None:
    lower, upper = get_hsv_bounds(window)
    print(
        "HSV bounds updated — "
        f"HSV_LOWER = {lower}  HSV_UPPER = {upper}"
    )


def setup_trackbars(window: str, initial: TrackbarHsv) -> None:
    cv2.createTrackbar("H min", window, initial.h_min, 179, lambda _: print_hsv_bounds(window))
    cv2.createTrackbar("H max", window, initial.h_max, 179, lambda _: print_hsv_bounds(window))
    cv2.createTrackbar("S min", window, initial.s_min, 255, lambda _: print_hsv_bounds(window))
    cv2.createTrackbar("S max", window, initial.s_max, 255, lambda _: print_hsv_bounds(window))
    cv2.createTrackbar("V min", window, initial.v_min, 255, lambda _: print_hsv_bounds(window))
    cv2.createTrackbar("V max", window, initial.v_max, 255, lambda _: print_hsv_bounds(window))


def build_mask(hsv: np.ndarray, window: str) -> np.ndarray:
    lower, upper = get_hsv_bounds(window)
    return cv2.inRange(hsv, np.array(lower), np.array(upper))


def find_largest_blob(mask: np.ndarray) -> Blob | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < config.MIN_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return Blob(centroid=(cx, cy), bbox=(x, y, w, h), area=area)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def map_x_norm(centroid_x: float, frame_width: int) -> float:
    if frame_width <= 1:
        return 0.5
    return clamp01(centroid_x / (frame_width - 1))


def map_y_norm(
    area: float,
    area_near: float | None,
    area_far: float | None,
) -> float | None:
    if area_near is None or area_far is None:
        return None
    span = area_near - area_far
    if abs(span) < 1.0:
        return 0.5
    if config.FORWARD_IS_UP:
        y_norm = (area_near - area) / span
    else:
        y_norm = (area - area_far) / span
    return clamp01(y_norm)


def norm_to_preview(
    x_norm: float,
    y_norm: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    x = int(round(x_norm * (frame_width - 1)))
    y = int(round(y_norm * (frame_height - 1)))
    return x, y


def draw_tracking_dot(
    frame: np.ndarray,
    x_norm: float,
    y_norm: float,
) -> None:
    height, width = frame.shape[:2]
    dot_x, dot_y = norm_to_preview(x_norm, y_norm, width, height)
    cv2.circle(
        frame,
        (dot_x, dot_y),
        config.TRACKING_DOT_RADIUS,
        config.TRACKING_DOT_COLOR,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (dot_x, dot_y),
        config.TRACKING_DOT_RADIUS + 2,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def draw_key_legend(frame: np.ndarray) -> None:
    legend = (
        f"{config.QUIT_KEY}=quit  {config.SAVE_KEY}=save HSV  "
        f"{config.NEAR_KEY}=near ({config.CALIBRATION_COUNTDOWN_SEC}s)  "
        f"{config.FAR_KEY}=far ({config.CALIBRATION_COUNTDOWN_SEC}s)"
    )
    height = frame.shape[0]
    cv2.putText(
        frame,
        legend,
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_calibration_countdown(
    frame: np.ndarray,
    kind: str,
    seconds_left: float,
) -> None:
    label = "NEAR" if kind == "near" else "FAR"
    seconds_display = max(0, math.ceil(seconds_left))

    height, width = frame.shape[:2]
    banner_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    title = f"GET READY FOR {label} CALIBRATION"
    if seconds_display > 0:
        countdown = f"IN {seconds_display}..."
    else:
        countdown = "SNAP!"

    title_scale = 0.9
    title_thickness = 2
    title_size, _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_DUPLEX, title_scale, title_thickness
    )
    title_x = max(10, (width - title_size[0]) // 2)
    cv2.putText(
        frame,
        title,
        (title_x, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        title_scale,
        (0, 255, 255),
        title_thickness,
        cv2.LINE_AA,
    )

    countdown_scale = 1.8 if seconds_display <= 3 else 1.2
    countdown_thickness = 3 if seconds_display <= 3 else 2
    countdown_size, _ = cv2.getTextSize(
        countdown, cv2.FONT_HERSHEY_DUPLEX, countdown_scale, countdown_thickness
    )
    countdown_x = max(10, (width - countdown_size[0]) // 2)
    countdown_color = (0, 0, 255) if seconds_display <= 3 else (0, 165, 255)
    cv2.putText(
        frame,
        countdown,
        (countdown_x, 95),
        cv2.FONT_HERSHEY_DUPLEX,
        countdown_scale,
        countdown_color,
        countdown_thickness,
        cv2.LINE_AA,
    )


def start_area_calibration(kind: str) -> PendingAreaCalibration:
    label = "NEAR (max/closer)" if kind == "near" else "FAR (min/back)"
    print(
        f"{label} calibration started — "
        f"get into position within {config.CALIBRATION_COUNTDOWN_SEC} seconds."
    )
    return PendingAreaCalibration(
        kind=kind,
        deadline=time.perf_counter() + config.CALIBRATION_COUNTDOWN_SEC,
    )


def complete_area_calibration(
    pending: PendingAreaCalibration,
    smoothed: SmoothedBlob | None,
    window: str,
    area_near: float | None,
    area_far: float | None,
) -> tuple[float | None, float | None]:
    label = "near" if pending.kind == "near" else "far"
    if smoothed is None:
        print(
            f"{label.upper()} calibration failed — no tracked blob at snap time.",
            file=sys.stderr,
        )
        return area_near, area_far

    if pending.kind == "near":
        area_near = save_near_baseline(window, smoothed.area, area_far)
    else:
        area_far = save_far_baseline(window, smoothed.area, area_near)
    print(f"{label.upper()} calibration complete — snapped area {smoothed.area:.0f} px")
    return area_near, area_far


def draw_mapping_overlay(
    frame: np.ndarray,
    x_norm: float | None,
    y_norm: float | None,
    area_near: float | None,
    area_far: float | None,
    y_offset: int = 104,
) -> None:
    if x_norm is not None and y_norm is not None:
        mapping_line = f"mapped   x={x_norm:.3f}  y={y_norm:.3f}"
        color = (203, 20, 255)
    else:
        mapping_line = "mapped   press 'n' (near) and 'f' (far) to calibrate depth"
        color = (0, 165, 255)

    cv2.putText(
        frame,
        mapping_line,
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )

    near_text = f"{area_near:.0f}" if area_near is not None else "—"
    far_text = f"{area_far:.0f}" if area_far is not None else "—"
    cv2.putText(
        frame,
        f"baselines  near(max)={near_text}  far(min)={far_text}",
        (10, y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def draw_crosshair(
    frame: np.ndarray,
    center: tuple[float, float],
    color: tuple[int, int, int],
    size: int,
    thickness: int = 2,
) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)


def draw_blob_overlay(
    frame: np.ndarray,
    blob: Blob | None,
    smoothed: SmoothedBlob | None,
    x_norm: float | None = None,
    y_norm: float | None = None,
    area_near: float | None = None,
    area_far: float | None = None,
) -> None:
    if blob is None:
        cv2.putText(
            frame,
            "NO BLOB",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        draw_mapping_overlay(frame, None, None, area_near, area_far, y_offset=86)
        return

    x, y, w, h = blob.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    draw_crosshair(frame, blob.centroid, (0, 255, 255), size=10, thickness=1)

    if smoothed is not None:
        draw_crosshair(frame, smoothed.centroid, (255, 255, 0), size=14, thickness=2)

    raw_cx, raw_cy = blob.centroid
    lines = [
        f"raw      cx={raw_cx:4d}  cy={raw_cy:4d}  area={blob.area:6.0f}",
    ]
    if smoothed is not None:
        sm_cx, sm_cy = smoothed.centroid
        lines.append(
            f"smooth   cx={sm_cx:6.1f}  cy={sm_cy:6.1f}  area={smoothed.area:6.1f}"
        )

    y_offset = 60
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y_offset += 22

    draw_mapping_overlay(frame, x_norm, y_norm, area_near, area_far, y_offset=y_offset)

    if x_norm is not None and y_norm is not None:
        draw_tracking_dot(frame, x_norm, y_norm)


def main() -> int:
    cap = open_camera(config.CAMERA_INDEX)
    if cap is None:
        print(
            "Could not open a webcam (tried indices 0–3).\n"
            "On macOS: System Settings → Privacy & Security → Camera → "
            "enable your terminal or IDE, then retry.",
            file=sys.stderr,
        )
        return 1

    calibration = load_calibration()
    area_near = calibration.area_near
    area_far = calibration.area_far
    smoother = BlobSmoother(config.EMA_ALPHA)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    setup_trackbars(config.WINDOW_NAME, calibration.hsv)
    print_hsv_bounds(config.WINDOW_NAME)
    print(
        f"Keys: {config.QUIT_KEY}=quit  {config.SAVE_KEY}=save HSV  "
        f"{config.NEAR_KEY}=near ({config.CALIBRATION_COUNTDOWN_SEC}s delay)  "
        f"{config.FAR_KEY}=far ({config.CALIBRATION_COUNTDOWN_SEC}s delay)"
    )

    frame_count = 0
    fps_interval_start = time.perf_counter()
    displayed_fps = 0.0
    last_smoothed_area: float | None = None
    pending_calibration: PendingAreaCalibration | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame from webcam.", file=sys.stderr)
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, config.WINDOW_NAME)
            blob = find_largest_blob(mask)

            smoothed: SmoothedBlob | None = None
            x_norm: float | None = None
            y_norm: float | None = None
            if blob is not None:
                smoothed = smoother.update(blob)
                frame_height, frame_width = frame.shape[:2]
                x_norm = map_x_norm(smoothed.centroid[0], frame_width)
                y_norm = map_y_norm(smoothed.area, area_near, area_far)
            else:
                smoother.reset()

            annotated = frame.copy()
            draw_blob_overlay(
                annotated,
                blob,
                smoothed,
                x_norm=x_norm,
                y_norm=y_norm,
                area_near=area_near,
                area_far=area_far,
            )

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_interval_start
            if elapsed >= 1.0:
                displayed_fps = frame_count / elapsed
                if blob is not None and smoothed is not None:
                    if y_norm is not None:
                        area_msg = (
                            f"area raw={blob.area:.0f} smooth={smoothed.area:.0f} "
                            f"| map x={x_norm:.2f} y={y_norm:.2f}"
                        )
                    else:
                        area_msg = (
                            f"area raw={blob.area:.0f} smooth={smoothed.area:.0f} "
                            f"| map x={x_norm:.2f} y=needs n/f"
                        )
                    last_smoothed_area = smoothed.area
                else:
                    area_msg = "area — (not found)"
                print(f"FPS: {displayed_fps:.1f} | {area_msg}")
                frame_count = 0
                fps_interval_start = now

            fps_label = f"FPS: {displayed_fps:.1f}"
            cv2.putText(
                annotated,
                fps_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            draw_key_legend(annotated)

            if pending_calibration is not None:
                remaining = pending_calibration.deadline - time.perf_counter()
                if remaining <= 0:
                    area_near, area_far = complete_area_calibration(
                        pending_calibration,
                        smoothed,
                        config.WINDOW_NAME,
                        area_near,
                        area_far,
                    )
                    pending_calibration = None
                else:
                    draw_calibration_countdown(
                        annotated,
                        pending_calibration.kind,
                        remaining,
                    )
                    seconds_display = max(0, math.ceil(remaining))
                    if pending_calibration.last_announced_second != seconds_display:
                        pending_calibration.last_announced_second = seconds_display
                        label = pending_calibration.kind.upper()
                        print(f"{label} calibration: {seconds_display}...")

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                mask_bgr,
                "mask",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_bgr,
                "yellow=smooth  cyan=raw",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if annotated.shape != mask_bgr.shape:
                mask_bgr = cv2.resize(mask_bgr, (annotated.shape[1], annotated.shape[0]))

            combined = np.hstack([annotated, mask_bgr])
            cv2.imshow(config.WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.SAVE_KEY):
                save_hsv_calibration(config.WINDOW_NAME, area_near, area_far)
            elif key == ord(config.NEAR_KEY):
                pending_calibration = start_area_calibration("near")
            elif key == ord(config.FAR_KEY):
                pending_calibration = start_area_calibration("far")
            elif key == ord(config.QUIT_KEY):
                print("Quit requested.")
                if last_smoothed_area is not None:
                    print(f"Last smoothed blob area: {last_smoothed_area:.0f} px")
                if area_near is not None:
                    print(f"area_near (max/closer): {area_near:.0f} px")
                if area_far is not None:
                    print(f"area_far (min/back): {area_far:.0f} px")
                lower, upper = get_hsv_bounds(config.WINDOW_NAME)
                print(f"Final HSV_LOWER = {lower}")
                print(f"Final HSV_UPPER = {upper}")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
