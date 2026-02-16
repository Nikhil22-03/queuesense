"""Detection records and the small amount of tracker bookkeeping we own.

ByteTrack itself lives inside Ultralytics. This module turns one tracking
result into person detections and asks an existing Ultralytics tracker to
forget its tracks between runs. Track ids are association hypotheses for the
current pass. They are not identities, and this project does not try to repair
an ID switch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def detections_from_result(result, person_class_ids: set[int]) -> list[Detection]:
    """Keep confirmed person tracks. Boxes without a track id are ignored."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    identities = _to_numpy(boxes.id)
    confidences = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    if xyxy is None or identities is None:
        return []
    best: dict[int, Detection] = {}
    for index in range(len(xyxy)):
        class_id = int(classes[index]) if classes is not None else -1
        if class_id not in person_class_ids:
            continue
        track_id = int(identities[index])
        x1, y1, x2, y2 = (float(value) for value in xyxy[index][:4])
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(confidences[index]) if confidences is not None else 0.0
        detection = Detection(track_id, (x1, y1, x2, y2), confidence, class_id)
        current = best.get(track_id)
        if current is None or detection.confidence >= current.confidence:
            best[track_id] = detection
    return [best[key] for key in sorted(best)]


def reset_model_trackers(model) -> None:
    """Clear ByteTrack memory on a model that has already tracked a video.

    The next track() call with persist=True then starts ids again. A model that
    has never tracked yet has nothing to clear.
    """
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None) or []
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()
