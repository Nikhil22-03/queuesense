"""Storage tests use a temporary database and never touch the project database."""

import pytest

from queuesense.storage import RunStore, allocate_run_dir, relativize, resolve_stored_path, sanitize_filename


def _run_row(run_id: str, output: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": "2026-11-05T00:00:00+00:00",
        "status": "running",
        "video_path": "video.mp4",
        "video_filename": "video.mp4",
        "output_dir": output,
        "width": 64,
        "height": 48,
        "fps": 10,
        "timestamp_mode": "frame_index",
        "config_json": "{}",
        "zones_json": "{}",
        "model_json": "{}",
        "hardware_json": "{}",
    }


def _finish(store: RunStore, run_id: str, track_id: int, visit_id: str) -> None:
    store.finish_run(
        run_id,
        status="completed",
        error=None,
        annotated_video_path=None,
        frames_processed=1,
        wall_time_sec=1.0,
        processing_fps=1.0,
        timestamp_mode="frame_index",
        summary={"completed": 1},
        visits=[
            {
                "visit_id": visit_id,
                "track_id": track_id,
                "status": "completed",
                "join_time": 0.0,
                "service_start": 1.0,
                "service_end": 2.0,
                "waiting_time": 1.0,
                "service_duration": 1.0,
                "exit_interval_start": 2.0,
                "closed_at": 3.0,
            }
        ],
        transitions=[],
        metrics=[
            {
                "timestamp": 0.0,
                "observed_queue_count": 0,
                "active_waiting_count": 0,
                "in_service_count": 0,
                "prediction_sec": None,
                "prediction_status": "insufficient_data",
                "mean_service_sec": None,
                "samples_used": 0,
                "remaining_service_sec": None,
                "one_server_ok": True,
            }
        ],
    )


def test_runs_do_not_share_visits(tmp_path):
    store = RunStore(tmp_path / "queue.db")
    odd_id = "run-a' OR '1'='1"
    store.create_run(_run_row(odd_id, "out-a"))
    store.create_run(_run_row("run-b", "out-b"))
    _finish(store, odd_id, track_id=7, visit_id="visit-0001")
    _finish(store, "run-b", track_id=9, visit_id="visit-0001")
    first = store.get_visits(odd_id)
    second = store.get_visits("run-b")
    assert [row["track_id"] for row in first] == [7]
    assert [row["track_id"] for row in second] == [9]
    assert store.get_metrics(odd_id)[0]["prediction_sec"] is None
    assert store.get_metrics("run-b")[0]["one_server_ok"] is True


def test_failed_run_keeps_its_error(tmp_path):
    store = RunStore(tmp_path / "queue.db")
    store.create_run(_run_row("run-c", "out-c"))
    store.fail_run("run-c", "decoder failed")
    assert store.get_run("run-c")["status"] == "failed"
    assert "decoder" in store.get_run("run-c")["error"]
    assert store.get_visits("run-c") == []


def test_filenames_and_run_directories_are_contained(tmp_path):
    assert sanitize_filename(r"..\..\secret.mp4") == "secret.mp4"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my video.mp4") == "my_video.mp4"
    assert sanitize_filename("..") == "video.mp4"
    first = allocate_run_dir(tmp_path, "same-id")
    second = allocate_run_dir(tmp_path, "same-id")
    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path


def test_relative_paths_roundtrip_inside_the_project(tmp_path):
    from queuesense.config import project_root

    stored = relativize(project_root() / "configs" / "default.json")
    assert not Path_is_absolute(stored)
    resolved = resolve_stored_path(stored)
    assert resolved == (project_root() / "configs" / "default.json").resolve()
    outside = tmp_path / "clip.mp4"
    outside.write_bytes(b"x")
    assert Path_is_absolute(relativize(outside))


def Path_is_absolute(value: str) -> bool:
    from pathlib import Path

    return Path(value).is_absolute()
