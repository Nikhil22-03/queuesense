"""Estimator tests. Samples are supplied directly, so the model is not involved."""

import pytest

from queuesense.prediction import WaitEstimator


def test_warmup_suppresses_the_number_until_three_samples():
    estimator = WaitEstimator(3)
    estimator.add_sample(10)
    estimator.add_sample(20)
    early = estimator.predict(1.0, waiting_count=4, in_service_elapsed=[1.0])
    assert early.status == "insufficient_data"
    assert early.estimate_sec is None
    estimator.add_sample(30)
    assert estimator.mean() == pytest.approx(20.0)


def test_formula_uses_waiting_count_and_remaining_service():
    estimator = WaitEstimator(3)
    for _ in range(3):
        estimator.add_sample(10)
    prediction = estimator.predict(5.0, waiting_count=2, in_service_elapsed=[4])
    assert prediction.status == "ok"
    assert prediction.remaining_service_sec == pytest.approx(6.0)
    assert prediction.estimate_sec == pytest.approx(26.0)

    overrun = estimator.predict(6.0, waiting_count=2, in_service_elapsed=[15])
    assert overrun.remaining_service_sec == pytest.approx(0.0)
    assert overrun.estimate_sec == pytest.approx(20.0)

    idle = estimator.predict(7.0, waiting_count=0, in_service_elapsed=[])
    assert idle.status == "ok"
    assert idle.estimate_sec == pytest.approx(0.0)


def test_two_people_in_service_suppresses_the_estimate():
    estimator = WaitEstimator(3)
    for duration in (10, 10, 10):
        estimator.add_sample(duration)
    blocked = estimator.predict(1.0, waiting_count=1, in_service_elapsed=[1.0, 2.0])
    assert blocked.status == "one_server_violation"
    assert blocked.estimate_sec is None
    assert blocked.one_server_ok is False


def test_too_few_samples_wins_over_the_server_flag_for_the_status():
    estimator = WaitEstimator(3)
    estimator.add_sample(10)
    prediction = estimator.predict(1.0, waiting_count=1, in_service_elapsed=[1.0, 2.0])
    assert prediction.status == "insufficient_data"
    assert prediction.estimate_sec is None
    assert prediction.one_server_ok is False
