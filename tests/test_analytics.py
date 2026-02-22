"""Empty results stay empty. A measured zero stays zero."""

from queuesense.analytics import format_count, format_seconds, prediction_message, queue_chart, summarize


def _metric(**overrides):
    row = {
        "timestamp": 1.0,
        "observed_queue_count": 0,
        "active_waiting_count": 0,
        "in_service_count": 0,
        "prediction_sec": None,
        "prediction_status": "insufficient_data",
        "mean_service_sec": None,
        "samples_used": 1,
        "remaining_service_sec": None,
        "one_server_ok": True,
    }
    row.update(overrides)
    return row


def test_missing_measurements_are_not_reported_as_zero():
    empty = summarize([], [])
    assert empty["mean_waiting_sec"] is None
    assert empty["median_waiting_sec"] is None
    assert empty["mean_service_sec"] is None
    assert empty["last_observed_queue_count"] is None
    assert empty["completed"] == 0
    assert format_seconds(None) == "—"
    assert format_seconds(0) == "0.0 s"
    assert format_count(None) == "unavailable"
    assert format_count(0) == "0"
    assert queue_chart([]) is None

    measured_zero = summarize([], [_metric()])
    assert measured_zero["last_observed_queue_count"] == 0
    assert measured_zero["mean_waiting_sec"] is None
    level, message = prediction_message(measured_zero)
    assert level == "info"
    assert "insufficient data" in message
