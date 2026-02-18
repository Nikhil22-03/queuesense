"""Run one video through detection, zones, the state machine, and storage.

The detector is injected. Tests pass a scripted detector so they never download
a model. The dashboard and CLI pass a PersonDetector. A new state machine and
a tracker reset happen inside every call.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from queuesense.analytics import summarize, write_result_tables
from queuesense.config import Settings, hardware_info, project_root
from queuesense.prediction import WaitEstimator
from queuesense.state_machine import COMPLETED, Observation, QueueStateMachine
from queuesense.storage import RunStore, allocate_run_dir, new_run_id, relativize, sanitize_filename
from queuesense.tracker import Detection
from queuesense.video import (
    VideoError,
    VideoInfo,
    even_size,
    fit_frame,
    is_readable_video,
    iter_video_frames,
    open_writer,
    output_fps,
    prefer_browser_mp4,
    probe_video,
)
from queuesense.visualization import draw_annotated_frame, prediction_overlay
from queuesense.zones import denormalize_points, ground_contact_point, validate_zones, zone_for_box, zone_warnings

logger = logging.getLogger(__name__)
_RUN_LOCK = threading.Lock()


@dataclass
class FrameStep:
    result: object
    prediction: object
    located: list[tuple[Detection, tuple[float, float], str]]


class AnalysisSession:
    """State, estimates, and per-frame metrics for a single video."""

    def __init__(self, queue, service, width: int, height: int, settings: Settings) -> None:
        self.queue_norm, self.service_norm = validate_zones(queue, service, width, height)
        self.width = width
        self.height = height
        self.queue_px = denormalize_points(self.queue_norm, width, height)
        self.service_px = denormalize_points(self.service_norm, width, height)
        self.machine = QueueStateMachine(settings.timing())
        self.estimator = WaitEstimator(settings.min_service_samples)
        self.metrics: list[dict] = []
        self._counted: set[str] = set()

    def step(self, timestamp: float, detections: list[Detection]) -> FrameStep:
        best: dict[int, Detection] = {}
        for detection in detections:
            current = best.get(detection.track_id)
            if current is None or detection.confidence >= current.confidence:
                best[detection.track_id] = detection
        observations: list[Observation] = []
        located: list[tuple[Detection, tuple[float, float], str]] = []
        for track_id in sorted(best):
            detection = best[track_id]
            point = ground_contact_point(detection.xyxy)
            zone = zone_for_box(detection.xyxy, self.queue_px, self.service_px)
            observations.append(Observation(track_id, zone))
            located.append((detection, point, zone))
        result = self.machine.update(timestamp, observations)
        for event in result.new_transitions:
            if event.to_state != COMPLETED or not event.visit_id or event.visit_id in self._counted:
                continue
            visit = self.machine.visits[event.visit_id]
            if visit.service_duration is None or visit.service_duration < 0:
                continue
            self.estimator.add_sample(visit.service_duration)
            self._counted.add(event.visit_id)
        prediction = self.estimator.predict(
            timestamp,
            result.active_waiting_count,
            result.in_service_elapsed,
        )
        self.metrics.append(
            {
                "timestamp": float(timestamp),
                "observed_queue_count": result.observed_queue_count,
                "active_waiting_count": result.active_waiting_count,
                "in_service_count": result.in_service_count,
                "prediction_sec": prediction.estimate_sec,
                "prediction_status": prediction.status,
                "mean_service_sec": prediction.mean_service_sec,
                "samples_used": prediction.samples_used,
                "remaining_service_sec": prediction.remaining_service_sec,
                "one_server_ok": prediction.one_server_ok,
            }
        )
        return FrameStep(result, prediction, located)

    def finish(self, end_time: float) -> None:
        self.machine.finalize(end_time)


@dataclass
class RunResult:
    run_id: str
    output_dir: Path
    annotated_video: Path | None
    summary: dict
    video_info: VideoInfo
    processing_fps: float | None
    status: str


def _render(frame, session: AnalysisSession, step: FrameStep):
    boxes = []
    views = step.result.tracks
    for detection, _point, _zone in step.located:
        view = views.get(detection.track_id)
        state = view.state if view else "OUTSIDE"
        label = f"track {detection.track_id} {state}"
        if view and view.visit_id:
            label += f" {view.visit_id}"
        boxes.append((detection.xyxy, label, state))
    line = prediction_overlay(step.prediction.status, step.prediction.estimate_sec)
    return draw_annotated_frame(
        frame,
        session.queue_px,
        session.service_px,
        boxes,
        step.result.observed_queue_count,
        step.result.active_waiting_count,
        line,
    )


def execute_run(
    video_path: Path,
    queue: list,
    service: list,
    settings: Settings,
    detector,
    output_root: Path | None = None,
    db_path: Path | None = None,
    progress=None,
) -> RunResult:
    """Process one video and store the run. Tracker and state start empty."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another analysis is already running in this process. Wait for it to finish.")
    try:
        return _execute_locked(video_path, queue, service, settings, detector, output_root, db_path, progress)
    finally:
        _RUN_LOCK.release()


def _execute_locked(video_path, queue, service, settings, detector, output_root, db_path, progress) -> RunResult:
    settings.validate()
    video_path = Path(video_path)
    info = probe_video(video_path)
    queue_norm, service_norm = validate_zones(queue, service, info.width, info.height)
    warnings = zone_warnings(queue_norm, service_norm, info.width, info.height)
    output_root = Path(output_root) if output_root else project_root() / "outputs"
    database = Path(db_path) if db_path else project_root() / "data" / "queuesense.db"
    run_id = new_run_id()
    output_dir = allocate_run_dir(output_root, run_id)
    store = RunStore(database)
    model_meta = dict(detector.metadata)
    hardware = hardware_info(str(model_meta.get("device", settings.device)))
    zones_payload = {"coordinate_space": "normalized", "queue": queue_norm, "service": service_norm}
    created = False
    try:
        store.create_run(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "running",
                "video_path": relativize(video_path),
                "video_filename": sanitize_filename(video_path.name),
                "output_dir": relativize(output_dir),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "reported_frame_count": info.reported_frame_count,
                "duration_sec": info.duration_sec,
                "timestamp_mode": info.timestamp_mode,
                "config_json": json.dumps(settings.to_dict()),
                "zones_json": json.dumps(zones_payload),
                "model_json": json.dumps(model_meta),
                "hardware_json": json.dumps(hardware),
            }
        )
        created = True
        (output_dir / "config.json").write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        (output_dir / "zones.json").write_text(json.dumps(zones_payload, indent=2), encoding="utf-8")
        detector.reset_tracker()
        session = AnalysisSession(queue_norm, service_norm, info.width, info.height, settings)
        writer_fps = output_fps(info.fps, settings.frame_stride)
        out_size = even_size(info.width, info.height)
        writer, annotated_path, codec = open_writer(output_dir / "annotated.mp4", writer_fps, out_size)
        processed = 0
        final_mode = info.timestamp_mode
        last_timestamp = 0.0
        expected = None
        if info.reported_frame_count:
            expected = max(1, (info.reported_frame_count + settings.frame_stride - 1) // settings.frame_stride)
        started = time.perf_counter()
        try:
            for packet in iter_video_frames(video_path, settings.frame_stride):
                final_mode = packet.timestamp_mode
                last_timestamp = packet.timestamp
                detections = detector.track_frame(packet.frame, settings.confidence)
                step = session.step(packet.timestamp, detections)
                writer.write(fit_frame(_render(packet.frame, session, step), out_size))
                processed += 1
                if progress is not None and (processed == 1 or processed % 10 == 0):
                    fraction = min(0.98, processed / expected) if expected else min(0.98, processed / 500)
                    progress(fraction, f"Frame {packet.index} at {packet.timestamp:.2f}s of video time")
        finally:
            writer.release()
        if processed == 0:
            raise VideoError("The video produced no frames to process.")
        session.finish(last_timestamp)
        wall = time.perf_counter() - started
        processing = processed / wall if wall > 0 else None
        playback_path, playback = (annotated_path, "opencv")
        video_error = None
        if is_readable_video(annotated_path):
            playback_path, playback = prefer_browser_mp4(annotated_path)
            if not is_readable_video(playback_path):
                video_error = f"Annotated video at {playback_path} exists but cannot be decoded."
                playback_path = None
        else:
            video_error = (
                f"Annotated video was not readable after writing with codec {codec}. "
                "The file path alone is not treated as success."
            )
            playback_path = None
        summary = summarize(session.machine.visit_list(), session.metrics)
        summary.update(
            {
                "warnings": warnings,
                "frames_processed": processed,
                "writer_codec": codec,
                "playback": playback if video_error is None else None,
                "output_fps": writer_fps,
                "timestamp_mode": final_mode,
                "frame_stride": settings.frame_stride,
            }
        )
        write_result_tables(
            output_dir,
            session.machine.visit_list(),
            session.metrics,
            [item.as_dict() for item in session.machine.transitions],
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        status = "completed" if video_error is None else "failed"
        store.finish_run(
            run_id,
            status=status,
            error=video_error,
            annotated_video_path=None if playback_path is None else relativize(playback_path),
            frames_processed=processed,
            wall_time_sec=wall,
            processing_fps=processing,
            timestamp_mode=final_mode,
            summary=summary,
            visits=[visit.as_dict() for visit in session.machine.visit_list()],
            transitions=[item.as_dict() for item in session.machine.transitions],
            metrics=session.metrics,
        )
        if video_error:
            raise VideoError(video_error)
        if progress is not None:
            progress(1.0, "Analysis stored.")
        return RunResult(run_id, output_dir, playback_path, summary, info, processing, status)
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        if created:
            current = store.get_run(run_id)
            if current and current["status"] == "running":
                store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
