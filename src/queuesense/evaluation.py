"""Compare a stored run with manual annotations.

Queue counts use the latest stored observation at or before the annotated time.
They never use a later frame. Waiting-time error uses matched visits only.
Prediction error uses the latest stored row at or before the check time, and
only when that row's status is ok. A suppressed or warmup row is skipped.
An older numeric estimate is not reused, and a later frame is not consulted.
"""

from __future__ import annotations

import json
from pathlib import Path

STATEMENT = (
    "These scores compare the run with the supplied manual annotations. "
    "They are not a person-detection benchmark and they are not a claim about real-world accuracy."
)


def load_ground_truth(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ground truth must be a JSON object.")
    for key in ("queue_counts", "visits", "prediction_checks"):
        payload.setdefault(key, [])
        if not isinstance(payload[key], list):
            raise ValueError(f"Ground truth field '{key}' must be a list.")
    return payload


def _mae(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def evaluate_run(
    metrics: list[dict],
    visits: list[dict],
    ground_truth: dict,
    run_info: dict | None = None,
) -> dict:
    info = run_info or {}
    tolerance = float(ground_truth.get("match_tolerance_sec", 2.0))
    ordered_metrics = sorted(metrics, key=lambda item: float(item["timestamp"]))
    predicted = [visit for visit in visits if visit.get("join_time") is not None]
    remaining = {visit["visit_id"]: visit for visit in predicted}

    queue_errors: list[float] = []
    queue_unscored = 0
    for item in ground_truth.get("queue_counts", []):
        moment = float(item["time_sec"])
        prior = [row for row in ordered_metrics if float(row["timestamp"]) <= moment + 1e-9]
        if not prior:
            queue_unscored += 1
            continue
        queue_errors.append(abs(int(prior[-1]["observed_queue_count"]) - int(item["count"])))

    matched: list[tuple[dict, dict]] = []
    missed: list[str] = []
    for ground in ground_truth.get("visits", []):
        explicit = ground.get("predicted_visit_id")
        if explicit:
            found = remaining.pop(explicit, None)
            if found is None:
                missed.append(str(ground.get("ground_truth_id")))
            else:
                matched.append((ground, found))
            continue
        target = float(ground["join_time_sec"])
        best_id = None
        best_delta = None
        for visit_id, visit in remaining.items():
            delta = abs(float(visit["join_time"]) - target)
            if delta > tolerance + 1e-9:
                continue
            better = best_delta is None or delta < best_delta - 1e-12
            tie = best_delta is not None and abs(delta - best_delta) <= 1e-12 and visit_id < best_id
            if better or tie:
                best_delta = delta
                best_id = visit_id
        if best_id is None:
            missed.append(str(ground.get("ground_truth_id")))
        else:
            matched.append((ground, remaining.pop(best_id)))

    wait_errors: list[float] = []
    for ground, visit in matched:
        if ground.get("service_start_sec") is None or visit.get("waiting_time") is None:
            continue
        actual = float(ground["service_start_sec"]) - float(ground["join_time_sec"])
        wait_errors.append(abs(actual - float(visit["waiting_time"])))

    prediction_errors: list[float] = []
    prediction_skipped = 0
    for check in ground_truth.get("prediction_checks", []):
        moment = float(check["time_sec"])
        prior = [row for row in ordered_metrics if float(row["timestamp"]) <= moment + 1e-9]
        latest = prior[-1] if prior else None
        if (
            latest is None
            or latest.get("prediction_status") != "ok"
            or latest.get("prediction_sec") is None
        ):
            # The estimate on display at this time was missing or suppressed.
            # An older numeric estimate is not reused, and a later frame is not consulted.
            prediction_skipped += 1
            continue
        prediction_errors.append(abs(float(latest["prediction_sec"]) - float(check["actual_wait_sec"])))

    return {
        "statement": STATEMENT,
        "notes": ground_truth.get("notes"),
        "run_status": info.get("status"),
        "queue_count_mae": _mae(queue_errors),
        "queue_count_samples": len(queue_errors),
        "queue_count_unscored": queue_unscored,
        "waiting_time_mae": _mae(wait_errors),
        "waiting_time_samples": len(wait_errors),
        "matched_visits": len(matched),
        "missed_visits": len(missed),
        "unmatched_visits": len(remaining),
        "missed_ids": missed,
        "unmatched_ids": sorted(remaining),
        "prediction_mae": _mae(prediction_errors),
        "prediction_samples": len(prediction_errors),
        "prediction_skipped": prediction_skipped,
        "processing_fps": info.get("processing_fps"),
        "hardware": info.get("hardware"),
    }


def format_report(report: dict) -> str:
    def metric_line(label: str, value: float | None, count: int, unit: str) -> str:
        if value is None:
            return f"{label}: unavailable (n={count})"
        return f"{label}: {value:.3f} {unit} (n={count})"

    lines = [
        report["statement"],
        "",
        f"Run status: {report.get('run_status') or 'unspecified'}",
        metric_line("Queue-count MAE", report["queue_count_mae"], report["queue_count_samples"], "people"),
        f"Queue-count checks with no earlier observation: {report['queue_count_unscored']}",
        metric_line(
            "Waiting-time MAE",
            report["waiting_time_mae"],
            report["waiting_time_samples"],
            "seconds",
        ),
        f"Matched visits: {report['matched_visits']}",
        f"Missed visits: {report['missed_visits']}",
        f"Unmatched visits: {report['unmatched_visits']}",
        metric_line(
            "Prediction MAE",
            report["prediction_mae"],
            report["prediction_samples"],
            "seconds",
        ),
        f"Prediction checks skipped because no valid stored estimate existed yet: {report['prediction_skipped']}",
    ]
    if report.get("processing_fps") is None:
        lines.append("Processing FPS: unavailable")
    else:
        lines.append(f"Processing FPS: {float(report['processing_fps']):.2f}")
    hardware = report.get("hardware") or {}
    if hardware:
        lines.append(
            "Hardware: "
            f"device={hardware.get('selected_device')}, "
            f"torch={hardware.get('torch')}, "
            f"cuda={hardware.get('cuda_available')}, "
            f"gpu={hardware.get('gpu_name')}, "
            f"opencv={hardware.get('opencv')}"
        )
    else:
        lines.append("Hardware: unavailable")
    if report.get("notes"):
        lines.extend(["", f"Annotation notes: {report['notes']}"])
    return "\n".join(lines) + "\n"
