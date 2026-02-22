"""Pipeline tests inject a scripted detector. Rectangles in the video are not people."""

from pathlib import Path

import pytest

from queuesense.analytics import summarize
from queuesense.config import Settings
from queuesense.pipeline import AnalysisSession, execute_run
from queuesense.storage import RunStore
from queuesense.tracker import Detection
from queuesense.video import is_readable_video, iter_video_frames, write_rectangle_video


class ScriptedDetector:
    def __init__(self, frames: list[list[Detection]]) -> None:
        self.frames = frames
        self.calls = 0
        self.resets = 0

    def reset_tracker(self) -> None:
        self.resets += 1
        self.calls = 0

    def track_frame(self, frame, confidence: float) -> list[Detection]:
        boxes = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return boxes

    @property
    def metadata(self) -> dict:
        return {
            "requested_model_path": "scripted",
            "resolved_model_path": "scripted",
            "weight_source": "test",
            "ultralytics_version": None,
            "class_names": {"1": "person"},
            "person_class_ids": [1],
            "device": "cpu",
            "image_size": 640,
            "tracker": "scripted",
        }


def _box(track_id: int, xyxy: tuple[float, float, float, float]) -> Detection:
    return Detection(track_id, xyxy, 0.9, 1)


def test_overlap_feet_count_as_service_and_not_as_queue():
    session = AnalysisSession(
        [(0.0, 0.0), (0.7, 0.0), (0.7, 1.0), (0.0, 1.0)],
        [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
        100,
        100,
        Settings(),
    )
    session.step(0.0, [_box(1, (50, 10, 70, 90))])
    assert session.metrics[0]["observed_queue_count"] == 0
    assert session.machine.visit_list() == []


def test_running_mean_does_not_rewrite_earlier_predictions():
    settings = Settings(
        join_confirm_sec=1,
        service_confirm_sec=1,
        exit_confirm_sec=1,
        missing_grace_sec=5,
        min_service_samples=3,
    )
    session = AnalysisSession(
        [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
        [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)],
        100,
        100,
        settings,
    )
    queue = _box(1, (10, 10, 30, 90))
    service = _box(1, (70, 10, 90, 90))
    outside = _box(1, (45, 10, 55, 90))
    zones = {0: queue, 1: queue, 2: service, 3: service, 4: outside, 5: outside}
    for cycle in range(3):
        base = cycle * 6
        for offset, detection in zones.items():
            session.step(float(base + offset), [detection])
    session.finish(17.0)
    by_time = {row["timestamp"]: row for row in session.metrics}
    assert by_time[5.0]["samples_used"] == 1
    assert by_time[5.0]["prediction_status"] == "insufficient_data"
    assert by_time[5.0]["prediction_sec"] is None
    assert by_time[11.0]["samples_used"] == 2
    assert by_time[11.0]["prediction_status"] == "insufficient_data"
    assert by_time[17.0]["samples_used"] == 3
    assert by_time[17.0]["prediction_status"] == "ok"
    assert by_time[17.0]["prediction_sec"] == pytest.approx(0.0)
    assert len(session.machine.visit_list()) == 3
    assert all(visit.status == "completed" for visit in session.machine.visit_list())


def test_scripted_video_run_is_isolated_and_resets_the_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr("queuesense.video.presentation_timestamps_reliable", lambda _samples: False)
    video = write_rectangle_video(tmp_path / "clip.mp4", frames=12, fps=10, size=(96, 64))
    packets = list(iter_video_frames(video, stride=1))
    queue = _box(1, (10, 10, 30, 50))
    service = _box(1, (60, 10, 84, 50))
    outside = _box(1, (40, 10, 56, 50))
    script = []
    for packet in packets:
        if packet.index <= 3:
            script.append([queue])
        elif packet.index <= 6:
            script.append([service])
        elif packet.index <= 9:
            script.append([outside])
        else:
            script.append([])
    settings = Settings(
        model_path="scripted",
        confidence=0.25,
        device="cpu",
        frame_stride=1,
        join_confirm_sec=0.3,
        service_confirm_sec=0.2,
        exit_confirm_sec=0.2,
        missing_grace_sec=1.0,
        min_service_samples=3,
    )
    zones = (
        [(0.0, 0.0), (0.45, 0.0), (0.45, 1.0), (0.0, 1.0)],
        [(0.55, 0.0), (1.0, 0.0), (1.0, 1.0), (0.55, 1.0)],
    )
    detector = ScriptedDetector(script)
    database = tmp_path / "runs.db"
    first = execute_run(
        video,
        zones[0],
        zones[1],
        settings,
        detector,
        output_root=tmp_path / "out",
        db_path=database,
    )
    assert first.status == "completed"
    assert first.annotated_video is not None
    assert is_readable_video(first.annotated_video)
    assert first.summary["completed"] == 1
    assert first.summary["mean_waiting_sec"] == pytest.approx(0.4)
    assert first.summary["last_prediction_status"] == "insufficient_data"
    assert first.summary["last_prediction_sec"] is None
    store = RunStore(database)
    visits = store.get_visits(first.run_id)
    assert len(visits) == 1
    assert visits[0]["status"] == "completed"
    assert visits[0]["waiting_time"] == pytest.approx(0.4)
    assert visits[0]["service_duration"] == pytest.approx(0.3)
    assert (Path(first.output_dir) / "visits.csv").is_file()

    second = execute_run(
        video,
        zones[0],
        zones[1],
        settings,
        detector,
        output_root=tmp_path / "out",
        db_path=database,
    )
    assert detector.resets == 2
    assert second.run_id != first.run_id
    assert [row["visit_id"] for row in store.get_visits(second.run_id)] == ["visit-0001"]
    assert len(store.get_visits(first.run_id)) == 1
    assert summarize(
        [],
        store.get_metrics(first.run_id),
    )["last_prediction_sec"] is None
