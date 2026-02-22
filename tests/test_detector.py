"""Parsing tests for detector output. They do not load a weight file."""

import numpy as np
import pytest

from queuesense.detector import resolve_person_class_ids
from queuesense.tracker import detections_from_result, reset_model_trackers


def test_person_class_comes_from_the_model_names():
    assert resolve_person_class_ids({0: "cat", 2: "person"}) == [2]
    assert resolve_person_class_ids(["Cat", "Person"]) == [1]
    with pytest.raises(RuntimeError, match="person"):
        resolve_person_class_ids({0: "head", 1: "bag"})


def test_only_person_tracks_with_ids_are_kept_and_duplicates_prefer_confidence():
    class Boxes:
        xyxy = np.array([[0, 0, 10, 10], [2, 2, 6, 8], [5, 5, 9, 12]], dtype=float)
        id = np.array([3, 3, 4])
        conf = np.array([0.2, 0.8, 0.95])
        cls = np.array([0, 0, 2])

        def __len__(self):
            return 3

    class Result:
        boxes = Boxes()

    kept = detections_from_result(Result(), {2})
    assert [item.track_id for item in kept] == [4]
    assert kept[0].class_id == 2

    class PersonBoxes(Boxes):
        cls = np.array([2, 2, 0])

    class PersonResult:
        boxes = PersonBoxes()

    kept = detections_from_result(PersonResult(), {2})
    chosen = {item.track_id: item for item in kept}
    assert set(chosen) == {3}
    assert chosen[3].confidence == pytest.approx(0.8)
    assert chosen[3].xyxy[0] == pytest.approx(2)


def test_reset_is_safe_before_the_first_prediction():
    class Tracker:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls += 1

    tracker = Tracker()

    class Model:
        predictor = type("Predictor", (), {"trackers": [tracker]})()

    reset_model_trackers(Model())
    assert tracker.calls == 1
    reset_model_trackers(object())
