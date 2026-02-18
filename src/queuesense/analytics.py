"""Tables and the queue-length chart. Empty measurements stay empty."""

from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from queuesense.state_machine import Visit

VISIT_COLUMNS = [
    "visit_id",
    "track_id",
    "status",
    "join_time_sec",
    "service_start_sec",
    "service_end_sec",
    "waiting_time_sec",
    "service_duration_sec",
    "exit_interval_start_sec",
    "closed_at_sec",
]
METRIC_COLUMNS = [
    "timestamp_sec",
    "observed_queue_count",
    "active_waiting_count",
    "in_service_count",
    "prediction_sec",
    "prediction_status",
    "mean_service_sec",
    "samples_used",
    "remaining_service_sec",
    "one_server_ok",
]
TRANSITION_COLUMNS = [
    "timestamp_sec",
    "track_id",
    "visit_id",
    "from_state",
    "to_state",
    "zone",
]


def format_seconds(value: float | None, digits: int = 1) -> str:
    """Format a duration. None is an em dash. A numeric zero stays numeric."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f} s"


def format_count(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return str(int(value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def summarize(visits: list[Visit], metrics: list[dict]) -> dict:
    completed_waits = [
        float(visit.waiting_time)
        for visit in visits
        if visit.status == "completed" and visit.waiting_time is not None
    ]
    service_durations = [
        float(visit.service_duration)
        for visit in visits
        if visit.status == "completed" and visit.service_duration is not None
    ]
    latest = metrics[-1] if metrics else None
    return {
        "completed": sum(1 for visit in visits if visit.status == "completed"),
        "abandoned": sum(1 for visit in visits if visit.status == "abandoned"),
        "lost": sum(1 for visit in visits if visit.status == "lost"),
        "incomplete": sum(1 for visit in visits if visit.status == "incomplete"),
        "mean_waiting_sec": _mean(completed_waits),
        "median_waiting_sec": _median(completed_waits),
        "waiting_samples": len(completed_waits),
        "mean_service_sec": _mean(service_durations),
        "median_service_sec": _median(service_durations),
        "service_samples": len(service_durations),
        "last_observed_queue_count": None if latest is None else int(latest["observed_queue_count"]),
        "last_active_waiting_count": None if latest is None else int(latest["active_waiting_count"]),
        "last_in_service_count": None if latest is None else int(latest["in_service_count"]),
        "last_prediction_sec": None if latest is None else latest["prediction_sec"],
        "last_prediction_status": None if latest is None else latest["prediction_status"],
        "last_mean_service_sec": None if latest is None else latest["mean_service_sec"],
        "last_prediction_samples": None if latest is None else int(latest["samples_used"]),
    }


def prediction_message(summary: dict) -> tuple[str, str]:
    """Return a level (empty, info, warning, ok) and the sentence shown in the dashboard."""
    status = summary.get("last_prediction_status")
    if status is None:
        return "empty", "No prediction was stored for this run."
    samples = summary.get("last_prediction_samples")
    sample_text = "an unknown number of" if samples is None else str(samples)
    if status == "insufficient_data":
        return (
            "info",
            "Estimated wait: insufficient data. "
            f"{sample_text} completed service samples are available, and 3 are required. "
            "This line is an estimate slot, separate from measured waits in the visit table.",
        )
    if status == "one_server_violation":
        return (
            "warning",
            "Estimated wait suppressed. More than one person is in service at this time, "
            "so the single-server assumption does not hold.",
        )
    if status == "ok":
        return (
            "ok",
            f"Estimated wait for a new arrival: {format_seconds(summary.get('last_prediction_sec'))}. "
            "This is an estimate, not a measured wait. It assumes one server and first-come service. "
            "The number is active waiting visits times the running mean of service durations completed "
            "so far, plus the estimated time left for the person in service. "
            "Time left is max(mean service − elapsed service, 0). "
            "Active waiting visits include tracks that are missing inside the grace period and people "
            "still inside the exit-confirmation window. Measured waits in the table are historical "
            "observations and are not this estimate.",
        )
    return "warning", f"Prediction status '{status}' is not one of the expected values."


def visits_dataframe(visits: list[Visit]) -> pd.DataFrame:
    rows = [
        {
            "visit_id": visit.visit_id,
            "track_id": visit.track_id,
            "status": visit.status,
            "join_time_sec": visit.join_time,
            "service_start_sec": visit.service_start,
            "service_end_sec": visit.service_end,
            "waiting_time_sec": visit.waiting_time,
            "service_duration_sec": visit.service_duration,
            "exit_interval_start_sec": visit.exit_interval_start,
            "closed_at_sec": visit.closed_at,
        }
        for visit in visits
    ]
    return pd.DataFrame(rows, columns=VISIT_COLUMNS)


def metrics_dataframe(metrics: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "timestamp_sec": item["timestamp"],
            "observed_queue_count": item["observed_queue_count"],
            "active_waiting_count": item["active_waiting_count"],
            "in_service_count": item["in_service_count"],
            "prediction_sec": item["prediction_sec"],
            "prediction_status": item["prediction_status"],
            "mean_service_sec": item["mean_service_sec"],
            "samples_used": item["samples_used"],
            "remaining_service_sec": item["remaining_service_sec"],
            "one_server_ok": item["one_server_ok"],
        }
        for item in metrics
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def transitions_dataframe(transitions: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "timestamp_sec": item["timestamp"],
            "track_id": item["track_id"],
            "visit_id": item.get("visit_id"),
            "from_state": item["from_state"],
            "to_state": item["to_state"],
            "zone": item.get("zone"),
        }
        for item in transitions
    ]
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def queue_chart(metrics: list[dict]):
    """Plot observed occupancy and active waiting visits. None when there are no rows."""
    if not metrics:
        return None
    frame = metrics_dataframe(metrics)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp_sec"],
            y=frame["observed_queue_count"],
            mode="lines",
            name="Observed in queue zone",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp_sec"],
            y=frame["active_waiting_count"],
            mode="lines",
            name="Active waiting visits (includes brief gaps)",
        )
    )
    figure.update_layout(
        title="Queue length over video time",
        xaxis_title="Video time (seconds)",
        yaxis_title="People",
        legend_title="Count",
        margin={"l": 48, "r": 24, "t": 56, "b": 48},
        hovermode="x unified",
    )
    return figure


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def write_result_tables(directory: Path, visits: list[Visit], metrics: list[dict], transitions: list[dict]) -> dict:
    directory = Path(directory)
    paths = {
        "visits": directory / "visits.csv",
        "metrics": directory / "metrics.csv",
        "transitions": directory / "transitions.csv",
    }
    write_csv(visits_dataframe(visits), paths["visits"])
    write_csv(metrics_dataframe(metrics), paths["metrics"])
    write_csv(transitions_dataframe(transitions), paths["transitions"])
    return paths
