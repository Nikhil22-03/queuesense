"""Local QueueSense dashboard.

Analysis runs only when Start analysis is pressed. Widget changes redraw the
page and keep the saved polygons, settings, and selected run. They do not call
the detector.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from queuesense import __version__
from queuesense.analytics import (
    format_count,
    format_seconds,
    metrics_dataframe,
    prediction_message,
    queue_chart,
    summarize,
    transitions_dataframe,
    visits_dataframe,
)
from queuesense.config import Settings, load_settings, project_root
from queuesense.pipeline import execute_run
from queuesense.state_machine import Visit
from queuesense.storage import RunStore, resolve_stored_path, sanitize_filename
from queuesense.video import VideoError, probe_video, read_first_frame, write_rectangle_video
from queuesense.zone_editor import edit_zones


def _store() -> RunStore:
    return RunStore(project_root() / "data" / "queuesense.db")


def _init_state() -> None:
    defaults = load_settings()
    initial = {
        "polygons": {"queue": [], "service": []},
        "zones_ready": False,
        "zone_error": None,
        "zone_warnings": [],
        "source_kind": "upload",
        "source_path": None,
        "upload_digest": None,
        "last_click_stamp": None,
        "last_error": None,
        "selected_run_id": None,
        "model_path": defaults.model_path,
        "confidence": defaults.confidence,
        "device": defaults.device,
        "frame_stride": defaults.frame_stride,
        "image_size": defaults.image_size,
        "join_confirm_sec": defaults.join_confirm_sec,
        "service_confirm_sec": defaults.service_confirm_sec,
        "exit_confirm_sec": defaults.exit_confirm_sec,
        "missing_grace_sec": defaults.missing_grace_sec,
        "min_service_samples": defaults.min_service_samples,
    }
    for key, value in initial.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _settings_from_state() -> Settings:
    settings = Settings(
        model_path=st.session_state.model_path,
        confidence=st.session_state.confidence,
        device=st.session_state.device,
        frame_stride=st.session_state.frame_stride,
        image_size=st.session_state.image_size,
        join_confirm_sec=st.session_state.join_confirm_sec,
        service_confirm_sec=st.session_state.service_confirm_sec,
        exit_confirm_sec=st.session_state.exit_confirm_sec,
        missing_grace_sec=st.session_state.missing_grace_sec,
        min_service_samples=st.session_state.min_service_samples,
    )
    settings.validate()
    return settings


@st.cache_resource(show_spinner="Loading the person-detection model...")
def _cached_detector(model_path: str, device: str, image_size: int):
    from queuesense.detector import PersonDetector

    return PersonDetector(model_path, device, image_size)


def _save_upload(upload) -> Path:
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:12]
    safe = sanitize_filename(upload.name)
    destination = project_root() / "data" / "uploads" / f"{digest}_{safe}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != len(data):
        destination.write_bytes(data)
    return destination


def _synthetic_path() -> Path:
    path = project_root() / "data" / "synthetic_rectangles.mp4"
    if not path.exists():
        write_rectangle_video(path, frames=30, fps=10, size=(320, 240))
    return path


def _selected_video() -> Path | None:
    if st.session_state.source_kind == "synthetic":
        return _synthetic_path()
    value = st.session_state.source_path
    if value and Path(value).is_file():
        return Path(value)
    return None


def _visits_from_rows(rows: list[dict]) -> list[Visit]:
    return [
        Visit(
            visit_id=row["visit_id"],
            track_id=int(row["track_id"]),
            status=row["status"],
            join_time=float(row["join_time"]),
            service_start=row["service_start"],
            service_end=row["service_end"],
            waiting_time=row["waiting_time"],
            service_duration=row["service_duration"],
            exit_interval_start=row["exit_interval_start"],
            closed_at=row["closed_at"],
        )
        for row in rows
    ]


def _render_setup() -> None:
    st.subheader("Video")
    st.radio(
        "Source",
        options=["upload", "synthetic"],
        format_func=lambda kind: "Uploaded video" if kind == "upload" else "Synthetic rectangle clip",
        horizontal=True,
        key="source_kind",
    )
    if st.session_state.source_kind == "synthetic":
        st.info(
            "This clip is a moving rectangle used to exercise video reading and the dashboard. "
            "It does not contain people, and a detector run on it does not measure detection accuracy."
        )
    else:
        upload = st.file_uploader(
            "Upload a fixed-camera video of one queue and one counter",
            type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
            max_upload_size=800,
        )
        if upload is not None:
            digest = hashlib.sha256(upload.getvalue()).hexdigest()
            if digest != st.session_state.upload_digest:
                path = _save_upload(upload)
                st.session_state.upload_digest = digest
                st.session_state.source_path = str(path)
                st.session_state.polygons = {"queue": [], "service": []}
                st.session_state.zones_ready = False
                st.session_state.zone_error = None
                st.session_state.last_click_stamp = None

    video = _selected_video()
    token = None if video is None else f"{st.session_state.source_kind}:{video.resolve()}"
    previous = st.session_state.get("video_token")
    if previous not in (None, token):
        st.session_state.polygons = {"queue": [], "service": []}
        st.session_state.zones_ready = False
        st.session_state.zone_error = None
        st.session_state.last_click_stamp = None
    st.session_state.video_token = token
    if video is None:
        st.info("Choose a video to see its metadata and draw zones.")
        return
    try:
        info = probe_video(video)
        frame = read_first_frame(video)
    except VideoError as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    left.metric("Frame size", f"{info.width}×{info.height}")
    right.metric("FPS", f"{info.fps:.3f}")
    left.metric("Reported frames", format_count(info.reported_frame_count))
    right.metric(
        "Estimated duration",
        "unavailable" if info.duration_sec is None else format_seconds(info.duration_sec),
    )
    st.caption(f"Timestamp source chosen from the file probe: {info.timestamp_mode.replace('_', ' ')}.")
    st.caption(f"File: {video}")

    edit_zones(frame, info.width, info.height)

    st.subheader("Model and timing")
    st.caption("These values are used the next time you press Start analysis. Changing them does not run the model.")
    st.text_input(
        "Model path",
        key="model_path",
        help="A local .pt file, or an Ultralytics name such as yolo26n.pt. The first use downloads that file.",
    )
    c1, c2, c3 = st.columns(3)
    c1.slider("Detection confidence", 0.05, 0.9, step=0.05, key="confidence")
    c2.selectbox("Device", ["auto", "cpu", "cuda"], key="device")
    c3.selectbox("Detector image size", [320, 480, 640, 960], key="image_size")
    st.number_input(
        "Frame stride",
        min_value=1,
        max_value=30,
        step=1,
        key="frame_stride",
        help=(
            "1 processes every frame. A larger stride keeps source timestamps and shortens the output "
            "so playback duration still matches the video, but skipped frames are invisible to tracking. "
            "A stride longer than the grace period can mark someone LOST after one missed sample."
        ),
    )
    t1, t2, t3, t4 = st.columns(4)
    t1.number_input("Join confirmation (seconds)", min_value=0.0, max_value=30.0, step=0.1, key="join_confirm_sec")
    t2.number_input("Service confirmation (seconds)", min_value=0.0, max_value=30.0, step=0.1, key="service_confirm_sec")
    t3.number_input("Exit confirmation (seconds)", min_value=0.0, max_value=30.0, step=0.1, key="exit_confirm_sec")
    t4.number_input("Missing-track grace (seconds)", min_value=0.0, max_value=30.0, step=0.1, key="missing_grace_sec")

    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)

    ready = video is not None and st.session_state.zones_ready
    if not st.session_state.zones_ready:
        st.info("Save a valid queue polygon and a valid service polygon before starting.")
    if st.button("Start analysis", disabled=not ready, type="primary"):
        _run_analysis(video)


def _run_analysis(video: Path) -> None:
    try:
        settings = _settings_from_state()
    except ValueError as exc:
        st.session_state.last_error = str(exc)
        st.error(str(exc))
        return
    progress = st.progress(0.0)
    status = st.empty()

    def _callback(fraction: float, message: str) -> None:
        progress.progress(min(1.0, max(0.0, float(fraction))))
        status.text(message)

    try:
        detector = _cached_detector(settings.model_path, settings.device, settings.image_size)
        result = execute_run(
            video,
            st.session_state.polygons["queue"],
            st.session_state.polygons["service"],
            settings,
            detector,
            progress=_callback,
        )
    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        st.error(st.session_state.last_error)
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return
    st.session_state.last_error = None
    st.session_state.selected_run_id = result.run_id
    progress.progress(1.0)
    status.text(f"Stored run {result.run_id}.")
    st.success(f"Analysis stored as {result.run_id}. Open the Results tab to inspect it.")


def _render_results() -> None:
    st.subheader("Saved runs")
    runs = _store().list_runs()
    if not runs:
        st.info("No runs yet. A completed analysis will appear here. This is an empty history, not a queue count of zero.")
        return
    labels = {run["run_id"]: f"{run['created_at']}  {run['run_id']}  ({run['status']})" for run in runs}
    preferred = st.session_state.selected_run_id
    options = list(labels)
    if preferred not in labels:
        preferred = options[0]
    selected = st.selectbox("Run", options=options, index=options.index(preferred), format_func=lambda item: labels[item])
    st.session_state.selected_run_id = selected
    run = _store().get_run(selected)
    if run is None:
        st.error("That run is no longer in the database.")
        return
    if run["status"] != "completed":
        st.warning(f"Run status: {run['status']}. {run['error'] or ''}")
    else:
        st.caption(f"Run status: {run['status']}.")

    video_path = resolve_stored_path(run.get("annotated_video_path"))
    if video_path is not None and video_path.is_file():
        st.video(str(video_path))
        st.caption(f"Annotated video: {video_path}")
        summary = json.loads(run["summary_json"] or "{}")
        if summary.get("playback") == "opencv":
            st.caption(
                "The file was checked by decoding a frame with OpenCV. "
                "If this player stays blank, the codec is one the browser does not play; open the same path in a desktop player."
            )
    else:
        st.warning("No annotated video is available for this run. A missing video is not reported as a successful export.")

    visits = _visits_from_rows(_store().get_visits(selected))
    metrics = _store().get_metrics(selected)
    transitions = _store().get_transitions(selected)
    summary = summarize(visits, metrics)
    stored_summary = json.loads(run["summary_json"] or "{}")

    st.markdown("**Last processed frame**")
    a, b, c = st.columns(3)
    a.metric("Observed queue count", format_count(summary["last_observed_queue_count"]))
    b.metric("Active waiting visits", format_count(summary["last_active_waiting_count"]))
    c.metric("People in service", format_count(summary["last_in_service_count"]))
    st.caption(
        "Observed queue count is how many detected ground points were inside the queue zone on the last processed frame. "
        "Active waiting visits also include tracks missing inside the grace period and people inside the exit-confirmation window. "
        "A displayed 0 means the count was measured as zero. Unavailable means this run has no frame metrics."
    )

    chart = queue_chart(metrics)
    if chart is None:
        st.info("No queue-length samples were stored for this run.")
    else:
        st.plotly_chart(chart, width="stretch")

    st.markdown("**Measured waits from completed visits**")
    d, e, f = st.columns(3)
    d.metric("Mean wait", format_seconds(summary["mean_waiting_sec"]))
    e.metric("Median wait", format_seconds(summary["median_waiting_sec"]))
    f.metric("Completed visits with a wait", summary["waiting_samples"])
    if summary["waiting_samples"] == 0:
        st.caption("Mean and median are unavailable because no completed visit has a measured wait. That is different from a wait of 0 seconds.")

    st.markdown("**Visit outcomes**")
    g, h, i, j = st.columns(4)
    g.metric("Completed", summary["completed"])
    h.metric("Abandoned", summary["abandoned"])
    i.metric("Lost", summary["lost"])
    j.metric("Incomplete", summary["incomplete"])
    st.caption("These count visits after the video ended. Zero means none of that outcome was recorded.")

    st.markdown("**Service durations from completed visits**")
    k, m, n = st.columns(3)
    k.metric("Mean service", format_seconds(summary["mean_service_sec"]))
    m.metric("Median service", format_seconds(summary["median_service_sec"]))
    n.metric("Service samples", summary["service_samples"])
    if summary["service_samples"] == 0:
        st.caption("Service mean and median are unavailable until a visit completes service.")

    level, message = prediction_message(summary)
    st.markdown("**Estimated wait for a new arrival**")
    if level == "ok":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "info":
        st.info(message)
    else:
        st.info(message)
    for warning in stored_summary.get("warnings") or []:
        st.warning(warning)

    fps = run["processing_fps"]
    st.caption(
        "Processing speed: "
        + ("unavailable" if fps is None else f"{float(fps):.2f} frames per second of wall-clock analysis time")
        + f". Timestamp mode used: {run['timestamp_mode'] or 'unspecified'}."
    )
    with st.expander("Model, device, and stored configuration"):
        st.json(
            {
                "model": json.loads(run["model_json"] or "{}"),
                "hardware": json.loads(run["hardware_json"] or "{}"),
                "config": json.loads(run["config_json"] or "{}"),
            }
        )

    st.markdown("**Visits**")
    visit_frame = visits_dataframe(visits)
    if visit_frame.empty:
        st.info("No visits were recorded. The table is empty; it is not a visit of length zero.")
    else:
        st.dataframe(visit_frame, width="stretch", hide_index=True)
    metric_frame = metrics_dataframe(metrics)
    transition_frame = transitions_dataframe(transitions)
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download visits CSV",
        data=visit_frame.to_csv(index=False, na_rep=""),
        file_name=f"{selected}_visits.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download queue metrics CSV",
        data=metric_frame.to_csv(index=False, na_rep=""),
        file_name=f"{selected}_metrics.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Download transitions CSV",
        data=transition_frame.to_csv(index=False, na_rep=""),
        file_name=f"{selected}_transitions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="QueueSense", layout="wide")
    _init_state()
    st.title("QueueSense")
    st.caption(
        f"Version {__version__}. One uploaded video, one fixed camera, one queue, and one service counter. "
        "Tracker ids are not identities."
    )
    setup, results = st.tabs(["Setup", "Results"])
    with setup:
        _render_setup()
    with results:
        _render_results()


if __name__ == "__main__":
    main()
