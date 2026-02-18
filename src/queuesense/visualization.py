"""Draw zones, track labels, and the two queue numbers on a frame."""

from __future__ import annotations

import cv2
import numpy as np

QUEUE_COLOR = (60, 170, 60)
SERVICE_COLOR = (0, 140, 255)
STATE_COLORS = {
    "OUTSIDE": (180, 180, 180),
    "JOIN_CANDIDATE": (180, 180, 0),
    "WAITING": (255, 170, 0),
    "SERVICE_CANDIDATE": (0, 200, 255),
    "IN_SERVICE": (40, 40, 220),
    "COMPLETED": (40, 160, 40),
    "ABANDONED": (0, 120, 255),
    "LOST": (140, 140, 255),
}


def _contour(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 3:
        return None
    return np.round(np.array(points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)


def _text_metrics(width: int) -> tuple[float, int, int]:
    """Return font scale, thickness, and the pixel step between HUD baselines."""
    scale = max(0.45, min(0.7, width / 1600))
    thickness = 1 if width < 1100 else 2
    (_, text_height), baseline = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return scale, thickness, text_height + baseline + 10


def hud_baselines(width: int, height: int, lines: int = 3) -> list[int]:
    """Baselines for the status lines, spaced so one line does not cover the next."""
    scale, thickness, step = _text_metrics(width)
    (_, text_height), _baseline = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    first = text_height + 8
    positions = [first + index * step for index in range(lines)]
    limit = max(first, height - 8)
    return [min(position, limit) for position in positions]


def _label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, _step = _text_metrics(frame.shape[1])
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(height + 4, min(y, frame.shape[0] - baseline - 2))
    x = max(0, min(x, max(0, frame.shape[1] - width - 8)))
    cv2.rectangle(frame, (x, y - height - 4), (x + width + 6, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_zones(
    frame: np.ndarray,
    queue_polygon: list[tuple[float, float]],
    service_polygon: list[tuple[float, float]],
) -> np.ndarray:
    """Return a copy with the two zones filled, outlined, and named."""
    canvas = frame.copy()
    overlay = canvas.copy()
    queue = _contour(queue_polygon)
    service = _contour(service_polygon)
    if queue is not None:
        cv2.fillPoly(overlay, [queue], QUEUE_COLOR)
    if service is not None:
        cv2.fillPoly(overlay, [service], SERVICE_COLOR)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
    if queue is not None:
        cv2.polylines(canvas, [queue], True, QUEUE_COLOR, 2, cv2.LINE_AA)
        anchor = tuple(int(value) for value in queue[0, 0])
        _label(canvas, "Queue", (anchor[0], max(22, anchor[1])), QUEUE_COLOR)
    if service is not None:
        cv2.polylines(canvas, [service], True, SERVICE_COLOR, 2, cv2.LINE_AA)
        anchor = tuple(int(value) for value in service[0, 0])
        _label(canvas, "Service", (anchor[0], max(22, anchor[1])), SERVICE_COLOR)
    return canvas


def prediction_overlay(status: str, estimate_sec: float | None) -> str:
    if status == "ok" and estimate_sec is not None:
        return f"ESTIMATE new-arrival wait {estimate_sec:.1f}s"
    if status == "insufficient_data":
        return "ESTIMATE insufficient data"
    if status == "one_server_violation":
        return "ESTIMATE suppressed: more than one person in service"
    return "ESTIMATE unavailable"


def draw_annotated_frame(
    frame: np.ndarray,
    queue_polygon: list[tuple[float, float]],
    service_polygon: list[tuple[float, float]],
    boxes: list[tuple[tuple[float, float, float, float], str, str]],
    observed_queue_count: int,
    active_waiting_count: int,
    prediction_line: str,
) -> np.ndarray:
    canvas = draw_zones(frame, queue_polygon, service_polygon)
    for xyxy, text, state in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        contact_x = int(round((xyxy[0] + xyxy[2]) / 2))
        contact_y = int(round(xyxy[3]))
        cv2.circle(canvas, (contact_x, contact_y), 4, color, -1)
        _label(canvas, text, (x1, max(18, y1 - 6)), color)
    first, second, third = hud_baselines(canvas.shape[1], canvas.shape[0], 3)
    _label(canvas, f"Observed queue: {observed_queue_count}", (8, first), (255, 255, 255))
    _label(
        canvas,
        f"Active waiting visits (includes brief gaps): {active_waiting_count}",
        (8, second),
        (255, 255, 255),
    )
    _label(canvas, prediction_line, (8, third), (255, 255, 255))
    return canvas
