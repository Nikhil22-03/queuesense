"""Queue and service polygons, in normalized coordinates and in pixels."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

QUEUE = "queue"
SERVICE = "service"
OUTSIDE = "outside"


class ZoneError(ValueError):
    """A polygon or a zone file cannot be used."""


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Absolute shoelace area. The unit is the square of the point unit."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def is_self_intersecting(points: list[tuple[float, float]]) -> bool:
    """True when non-adjacent edges cross. Touching at a shared vertex is allowed."""
    count = len(points)
    if count < 4:
        return False
    for left in range(count):
        a = points[left]
        b = points[(left + 1) % count]
        for right in range(left + 1, count):
            adjacent = abs(left - right) <= 1 or (left == 0 and right == count - 1)
            if adjacent:
                continue
            c = points[right]
            d = points[(right + 1) % count]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def normalize_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        raise ZoneError("Cannot normalize coordinates without a positive frame size.")
    return [(float(x) / float(width), float(y) / float(height)) for x, y in points]


def denormalize_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        raise ZoneError("Cannot convert normalized coordinates without a positive frame size.")
    return [(float(x) * float(width), float(y) * float(height)) for x, y in points]


def click_to_normalized(
    x: float,
    y: float,
    reported_width: float,
    reported_height: float,
) -> tuple[float, float]:
    """Map a click into 0..1 using the width and height reported with that click.

    The click and the reported size must share one coordinate system. That is true
    both for source pixels and for a resized preview, so display scaling does not
    change the stored polygon.
    """
    if reported_width <= 0 or reported_height <= 0:
        raise ZoneError("The click was reported with an invalid image size.")
    nx = float(x) / float(reported_width)
    ny = float(y) / float(reported_height)
    if not (-0.02 <= nx <= 1.02 and -0.02 <= ny <= 1.02):
        raise ZoneError(f"Click ({x:.1f}, {y:.1f}) falls outside the image.")
    return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))


def validate_polygon(points: list, name: str) -> list[tuple[float, float]]:
    """Return a cleaned normalized polygon or raise ZoneError."""
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ZoneError(f"The {name} zone has a point that is not an x, y pair.")
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ZoneError(f"The {name} zone has a non-finite coordinate.")
        if x < -1e-6 or y < -1e-6 or x > 1 + 1e-6 or y > 1 + 1e-6:
            raise ZoneError(
                f"The {name} zone must use normalized coordinates between 0 and 1. Got ({x}, {y})."
            )
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        if cleaned and abs(cleaned[-1][0] - x) < 1e-9 and abs(cleaned[-1][1] - y) < 1e-9:
            continue
        cleaned.append((x, y))
    if (
        len(cleaned) >= 2
        and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-9
        and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-9
    ):
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        raise ZoneError(f"The {name} zone needs at least 3 distinct points.")
    if is_self_intersecting(cleaned):
        raise ZoneError(f"The {name} zone crosses itself. Click the corners in order around the boundary.")
    if polygon_area(cleaned) <= 1e-8:
        raise ZoneError(f"The {name} zone has zero area.")
    return cleaned


def validate_zones(
    queue: list,
    service: list | None,
    width: int,
    height: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Validate a queue polygon. Service may be omitted when no counter is visible."""
    queue_polygon = validate_polygon(queue, "queue")
    service_polygon = [] if not service else validate_polygon(service, "service")
    for name, polygon in (("queue", queue_polygon), ("service", service_polygon)):
        if not polygon:
            continue
        pixel_area = polygon_area(denormalize_points(polygon, width, height))
        if pixel_area < 4:
            raise ZoneError(f"The {name} zone covers fewer than 4 square pixels.")
    return queue_polygon, service_polygon


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """True for interior points and for points on the boundary."""
    if len(polygon) < 3:
        return False
    contour = np.array([[float(x), float(y)] for x, y in polygon], dtype=np.float32).reshape(-1, 1, 2)
    measure = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return float(measure) >= 0


def ground_contact_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a bounding box, used as a stand-in for where the feet meet the ground."""
    x1, _y1, x2, y2 = (float(value) for value in xyxy)
    return (x1 + x2) / 2.0, y2


def assign_zone(
    point: tuple[float, float],
    queue_polygon: list[tuple[float, float]],
    service_polygon: list[tuple[float, float]],
) -> str:
    """Service wins when a point is inside both polygons."""
    if point_in_polygon(point, service_polygon):
        return SERVICE
    if point_in_polygon(point, queue_polygon):
        return QUEUE
    return OUTSIDE


def zone_for_box(
    xyxy: tuple[float, float, float, float],
    queue_polygon: list[tuple[float, float]],
    service_polygon: list[tuple[float, float]],
) -> str:
    return assign_zone(ground_contact_point(xyxy), queue_polygon, service_polygon)


def zone_warnings(
    queue: list[tuple[float, float]],
    service: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[str]:
    queue_px = denormalize_points(queue, width, height)
    service_px = denormalize_points(service, width, height)
    if queue_px and all(point_in_polygon(point, service_px) for point in queue_px):
        return [
            "Every queue vertex lies inside the service zone. "
            "The service zone has priority, so those points are never counted as queue."
        ]
    return []


def load_zone_file(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    space = str(payload.get("coordinate_space", "normalized")).lower()
    if space != "normalized":
        raise ZoneError("Zone files must use coordinate_space 'normalized' with values from 0 to 1.")
    if "queue" not in payload:
        raise ZoneError("Zone file must contain a 'queue' point list.")
    queue = validate_polygon(payload["queue"], "queue")
    service_raw = payload.get("service")
    service = None if not service_raw else validate_polygon(service_raw, "service")
    return queue, service


def dump_zone_file(path: Path, queue: list, service: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coordinate_space": "normalized",
        "queue": [[float(x), float(y)] for x, y in queue],
        "service": [[float(x), float(y)] for x, y in service],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
