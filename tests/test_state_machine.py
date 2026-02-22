"""State-machine tests use synthetic observations and timestamps. No detector."""

import pytest

from queuesense.config import Timing
from queuesense.state_machine import IN_SERVICE, WAITING, Observation, QueueStateMachine


def see(*pairs: tuple[int, str]) -> list[Observation]:
    return [Observation(track_id, zone) for track_id, zone in pairs]


def machine() -> QueueStateMachine:
    return QueueStateMachine()


def test_passerby_never_confirms_a_visit():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.0, see((1, "queue")))
    sm.update(1.2, see((1, "outside")))
    sm.finalize(1.2)
    assert sm.visit_list() == []


def test_many_frames_do_not_join_before_the_time_threshold():
    sm = machine()
    for timestamp in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        sm.update(timestamp, see((1, "queue")))
    assert sm.visits == {}


def test_two_frames_join_when_elapsed_time_is_enough_and_join_time_is_the_start():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    visit = sm.visits["visit-0001"]
    assert visit.join_time == pytest.approx(0.0)
    assert visit.track_id == 1
    assert visit.visit_id != str(visit.track_id)
    assert visit.status == "waiting"


def test_join_then_service_uses_interval_starts_for_both_durations():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(10.0, see((1, "service")))
    sm.update(11.0, see((1, "service")))
    sm.update(20.0, see((1, "outside")))
    result = sm.update(22.0, see((1, "outside")))
    visit = sm.visits["visit-0001"]
    assert visit.status == "completed"
    assert visit.service_start == pytest.approx(10.0)
    assert visit.waiting_time == pytest.approx(10.0)
    assert visit.service_end == pytest.approx(20.0)
    assert visit.service_duration == pytest.approx(10.0)
    assert result.tracks[1].state == "COMPLETED"


def test_confirmed_queue_exit_is_abandonment_without_a_wait():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(4.0, see((1, "outside")))
    sm.update(6.0, see((1, "outside")))
    visit = sm.visits["visit-0001"]
    assert visit.status == "abandoned"
    assert visit.waiting_time is None
    assert visit.service_start is None
    assert visit.exit_interval_start == pytest.approx(4.0)


def test_brief_disappearance_keeps_the_same_waiting_visit():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(2.0, see((1, "queue")))
    missing = sm.update(3.0, [])
    returned = sm.update(4.0, see((1, "queue")))
    assert missing.observed_queue_count == 0
    assert missing.active_waiting_count == 1
    assert returned.active_waiting_count == 1
    assert len(sm.visit_list()) == 1
    assert sm.visits["visit-0001"].status == "waiting"
    assert returned.tracks[1].state == WAITING


def test_gap_beyond_grace_is_lost_and_not_abandoned():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(2.0, see((1, "queue")))
    sm.update(6.1, [])
    visit = sm.visits["visit-0001"]
    assert visit.status == "lost"
    assert visit.service_end is None
    sm.finalize(6.1)
    assert visit.status == "lost"


def test_missing_observations_do_not_satisfy_join_confirmation():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(0.5, see((1, "queue")))
    sm.update(1.0, [])
    sm.update(1.6, see((1, "queue")))
    assert sm.visits == {}
    sm.update(3.1, see((1, "queue")))
    assert sm.visits["visit-0001"].join_time == pytest.approx(1.6)


def test_missing_observations_do_not_satisfy_exit_confirmation():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(2.0, see((1, "service")))
    sm.update(3.0, see((1, "service")))
    sm.update(4.0, see((1, "outside")))
    sm.update(5.0, [])
    midway = sm.update(6.0, see((1, "outside")))
    assert sm.visits["visit-0001"].status == "in_service"
    assert midway.tracks[1].state == IN_SERVICE
    sm.update(8.0, see((1, "outside")))
    visit = sm.visits["visit-0001"]
    assert visit.status == "completed"
    assert visit.service_end == pytest.approx(6.0)


def test_rejoin_after_completion_is_a_new_visit():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(2.0, see((1, "service")))
    sm.update(3.0, see((1, "service")))
    sm.update(3.5, see((1, "outside")))
    sm.update(5.5, see((1, "outside")))
    sm.update(6.0, see((1, "queue")))
    sm.update(7.5, see((1, "queue")))
    assert sm.visits["visit-0001"].status == "completed"
    assert sm.visits["visit-0002"].status == "waiting"
    assert sm.visits["visit-0002"].track_id == 1
    assert sm.visits["visit-0002"].join_time == pytest.approx(6.0)


def test_rejoin_after_lost_keeps_the_old_visit_lost():
    sm = machine()
    sm.update(0.0, see((1, "queue")))
    sm.update(1.5, see((1, "queue")))
    sm.update(6.0, [])
    sm.update(7.0, see((1, "queue")))
    sm.update(8.5, see((1, "queue")))
    assert sm.visits["visit-0001"].status == "lost"
    assert sm.visits["visit-0002"].status == "waiting"


def test_open_visits_at_the_end_are_incomplete():
    waiting = machine()
    waiting.update(0.0, see((1, "queue")))
    waiting.update(1.5, see((1, "queue")))
    waiting.finalize(2.0)
    assert waiting.visits["visit-0001"].status == "incomplete"
    assert waiting.visits["visit-0001"].service_end is None
    assert all(event.to_state != "COMPLETED" for event in waiting.transitions)

    serving = machine()
    serving.update(0.0, see((1, "queue")))
    serving.update(1.5, see((1, "queue")))
    serving.update(2.0, see((1, "service")))
    serving.update(3.0, see((1, "service")))
    serving.finalize(3.2)
    visit = serving.visits["visit-0001"]
    assert visit.status == "incomplete"
    assert visit.waiting_time == pytest.approx(2.0)
    assert visit.service_end is None
    serving.finalize(3.2)
    assert sum(1 for event in serving.transitions if event.to_state == "INCOMPLETE") == 1


def test_observed_queue_and_active_waiting_are_different_counts():
    sm = machine()
    both = see((1, "queue"), (2, "queue"))
    sm.update(0.0, both)
    sm.update(1.5, both)
    result = sm.update(2.0, see((1, "queue")))
    assert result.observed_queue_count == 1
    assert result.active_waiting_count == 2


def test_timestamps_must_increase():
    sm = machine()
    sm.update(1.0, see((1, "outside")))
    with pytest.raises(ValueError, match="increase"):
        sm.update(1.0, see((1, "outside")))


def test_custom_timing_still_uses_seconds():
    sm = QueueStateMachine(Timing(0.4, 0.4, 0.4, 1.0))
    sm.update(0.0, see((1, "queue")))
    sm.update(0.4, see((1, "queue")))
    assert sm.visits["visit-0001"].join_time == pytest.approx(0.0)
