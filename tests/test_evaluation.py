"""Evaluation uses hand-written rows. It does not score the synthetic rectangle clip."""

import pytest

from queuesense.evaluation import STATEMENT, evaluate_run, format_report


def test_scores_match_only_the_supplied_rows_and_ignore_the_future():
    metrics = [
        {
            "timestamp": 0.5,
            "observed_queue_count": 1,
            "prediction_status": "ok",
            "prediction_sec": 12.0,
        },
        {
            "timestamp": 1.5,
            "observed_queue_count": 0,
            "prediction_status": "insufficient_data",
            "prediction_sec": None,
        },
        {
            "timestamp": 9.0,
            "observed_queue_count": 4,
            "prediction_status": "ok",
            "prediction_sec": 99.0,
        },
    ]
    visits = [
        {
            "visit_id": "visit-0001",
            "join_time": 0.2,
            "service_start": 5.4,
            "waiting_time": 5.2,
            "status": "completed",
        },
        {
            "visit_id": "visit-0002",
            "join_time": 9.0,
            "service_start": None,
            "waiting_time": None,
            "status": "abandoned",
        },
    ]
    ground_truth = {
        "notes": "unit-test annotations",
        "match_tolerance_sec": 2.0,
        "queue_counts": [
            {"time_sec": 1.0, "count": 1},
            {"time_sec": 2.0, "count": 0},
            {"time_sec": -1.0, "count": 0},
        ],
        "visits": [
            {
                "ground_truth_id": "g1",
                "join_time_sec": 0.0,
                "service_start_sec": 5.0,
                "predicted_visit_id": None,
            },
            {
                "ground_truth_id": "g2",
                "join_time_sec": 30.0,
                "service_start_sec": 40.0,
                "predicted_visit_id": "visit-0009",
            },
        ],
        "prediction_checks": [
            {"time_sec": 1.0, "actual_wait_sec": 10.0},
            {"time_sec": 1.5, "actual_wait_sec": 1.0},
        ],
    }
    report = evaluate_run(metrics, visits, ground_truth, {"processing_fps": 8.5, "status": "completed"})
    assert report["statement"] == STATEMENT
    assert report["queue_count_mae"] == 0
    assert report["queue_count_samples"] == 2
    assert report["queue_count_unscored"] == 1
    assert report["waiting_time_mae"] == pytest.approx(0.2)
    assert report["matched_visits"] == 1
    assert report["missed_visits"] == 1
    assert report["unmatched_visits"] == 1
    assert report["unmatched_ids"] == ["visit-0002"]
    assert report["prediction_mae"] == 2.0
    assert report["prediction_samples"] == 1
    assert report["prediction_skipped"] == 1
    text = format_report(report)
    assert "not a person-detection benchmark" in text
    assert "Processing FPS: 8.50" in text


def test_empty_annotations_do_not_become_a_zero_error():
    report = evaluate_run([], [], {"queue_counts": [], "visits": [], "prediction_checks": []}, {})
    assert report["queue_count_mae"] is None
    assert report["waiting_time_mae"] is None
    assert report["prediction_mae"] is None
    assert "unavailable" in format_report(report)
