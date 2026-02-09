"""Timestamp state machine for one queue and one service counter.

Rules implemented here:

- A state change that needs a confirmation uses source-video time, not frame counts.
- The timer starts at the first frame of an uninterrupted run of observations.
- A processed frame that does not contain the track breaks that run. The missing
  time does not count toward join, service, or exit confirmation.
- Returning within the grace period keeps the open visit. Exceeding the grace
  period marks that visit LOST. A person who never confirmed a join does not
  become a visit.
- Join time and service-start time are the beginnings of the confirmed intervals.
- Service ends at the beginning of the confirmed exit interval.
- Waiting time is service start minus join time. It is empty when service never starts.
- The same track can open a later visit. Track ids are not visit ids.
- finalize() marks visits that are still open as incomplete. It does not invent
  a completion or an abandonment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from queuesense.config import Timing

OUTSIDE = "OUTSIDE"
JOIN_CANDIDATE = "JOIN_CANDIDATE"
WAITING = "WAITING"
SERVICE_CANDIDATE = "SERVICE_CANDIDATE"
IN_SERVICE = "IN_SERVICE"
COMPLETED = "COMPLETED"
ABANDONED = "ABANDONED"
LOST = "LOST"
INCOMPLETE = "INCOMPLETE"

QUEUE = "queue"
SERVICE = "service"
OUTSIDE_ZONE = "outside"
ALLOWED_ZONES = {QUEUE, SERVICE, OUTSIDE_ZONE}
ACTIVE_STATES = {JOIN_CANDIDATE, WAITING, SERVICE_CANDIDATE, IN_SERVICE}
TERMINAL_STATES = {COMPLETED, ABANDONED, LOST}


def _due(start: float, now: float, duration: float) -> bool:
    return (now - start) + 1e-6 >= duration


@dataclass
class Visit:
    visit_id: str
    track_id: int
    status: str
    join_time: float
    service_start: float | None = None
    service_end: float | None = None
    waiting_time: float | None = None
    service_duration: float | None = None
    exit_interval_start: float | None = None
    closed_at: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Transition:
    timestamp: float
    track_id: int
    visit_id: str | None
    from_state: str
    to_state: str
    zone: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    track_id: int
    zone: str


@dataclass(frozen=True)
class TrackView:
    track_id: int
    state: str
    visit_id: str | None
    zone: str | None


@dataclass
class UpdateResult:
    observed_queue_count: int
    active_waiting_count: int
    in_service_count: int
    in_service_elapsed: list[float]
    tracks: dict[int, TrackView]
    new_transitions: list[Transition]


@dataclass
class _Track:
    track_id: int
    state: str = OUTSIDE
    visit_id: str | None = None
    last_seen: float | None = None
    seen_last_frame: bool = False
    interval_start: float | None = None
    exit_interval_start: float | None = None
    service_start: float | None = None


class QueueStateMachine:
    """Per-run track memory. Construct a new machine for every video."""

    def __init__(self, timing: Timing | None = None) -> None:
        self.timing = timing or Timing()
        self.tracks: dict[int, _Track] = {}
        self.visits: dict[str, Visit] = {}
        self.transitions: list[Transition] = []
        self._sequence = 0
        self._last_time: float | None = None
        self._current_new: list[Transition] | None = None

    def visit_list(self) -> list[Visit]:
        return list(self.visits.values())

    def update(self, timestamp: float, observations: list[Observation]) -> UpdateResult:
        timestamp = float(timestamp)
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError(
                f"Timestamps must increase. Previous timestamp was {self._last_time}, got {timestamp}."
            )
        unique: dict[int, Observation] = {}
        for observation in observations:
            if observation.zone not in ALLOWED_ZONES:
                raise ValueError(f"Unknown zone {observation.zone!r}.")
            unique[int(observation.track_id)] = Observation(int(observation.track_id), observation.zone)
        ordered = [unique[key] for key in sorted(unique)]
        seen = set(unique)
        new_transitions: list[Transition] = []
        self._current_new = new_transitions
        for track_id in sorted(self.tracks):
            if track_id in seen:
                continue
            track = self.tracks[track_id]
            if track.state in ACTIVE_STATES:
                self._handle_missing(track, timestamp)
        for observation in ordered:
            track = self.tracks.get(observation.track_id)
            if track is None:
                track = _Track(track_id=observation.track_id)
                self.tracks[observation.track_id] = track
            self._on_observation(track, timestamp, observation.zone)
        for track in self.tracks.values():
            track.seen_last_frame = track.track_id in seen
        self._last_time = timestamp
        self._current_new = None

        active_waiting = 0
        in_service = 0
        elapsed: list[float] = []
        views: dict[int, TrackView] = {}
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            zone = unique[track_id].zone if track_id in unique else None
            views[track_id] = TrackView(track_id, track.state, track.visit_id, zone)
            if track.state in {WAITING, SERVICE_CANDIDATE}:
                active_waiting += 1
            elif track.state == IN_SERVICE and track.service_start is not None:
                in_service += 1
                elapsed.append(max(0.0, timestamp - track.service_start))
        observed = sum(1 for observation in ordered if observation.zone == QUEUE)
        return UpdateResult(observed, active_waiting, in_service, elapsed, views, new_transitions)

    def finalize(self, end_time: float) -> None:
        """Close visits that never reached a terminal state. Safe to call once."""
        end_time = float(end_time)
        for track in self.tracks.values():
            if track.visit_id is None:
                continue
            visit = self.visits.get(track.visit_id)
            if visit is None or visit.status not in {"waiting", "in_service"}:
                continue
            previous = track.state
            visit.status = "incomplete"
            visit.closed_at = end_time
            self.transitions.append(
                Transition(end_time, track.track_id, track.visit_id, previous, INCOMPLETE, None)
            )

    def _handle_missing(self, track: _Track, timestamp: float) -> None:
        gap = 0.0 if track.last_seen is None else timestamp - track.last_seen
        if gap > self.timing.missing_grace_sec + 1e-6:
            track.interval_start = None
            track.exit_interval_start = None
            if track.visit_id is None:
                self._set_state(track, OUTSIDE, timestamp, None)
            else:
                self._lose(track, timestamp)
            return
        # The absence itself must not finish a confirmation interval.
        track.interval_start = None
        track.exit_interval_start = None

    def _on_observation(self, track: _Track, timestamp: float, zone: str) -> None:
        if track.state in TERMINAL_STATES:
            track.visit_id = None
            track.service_start = None
            track.interval_start = None
            track.exit_interval_start = None
            track.state = OUTSIDE
        if track.last_seen is not None and not track.seen_last_frame:
            track.interval_start = None
            track.exit_interval_start = None
        if track.state == OUTSIDE:
            self._from_outside(track, timestamp, zone)
        elif track.state == JOIN_CANDIDATE:
            self._from_join(track, timestamp, zone)
        elif track.state == WAITING:
            self._from_waiting(track, timestamp, zone)
        elif track.state == SERVICE_CANDIDATE:
            self._from_service(track, timestamp, zone)
        elif track.state == IN_SERVICE:
            self._from_in_service(track, timestamp, zone)
        else:
            raise RuntimeError(f"Unhandled state {track.state}")
        track.last_seen = timestamp

    def _from_outside(self, track: _Track, timestamp: float, zone: str) -> None:
        if zone == QUEUE:
            track.interval_start = timestamp
            self._set_state(track, JOIN_CANDIDATE, timestamp, zone)

    def _from_join(self, track: _Track, timestamp: float, zone: str) -> None:
        if zone == QUEUE:
            if track.interval_start is None:
                track.interval_start = timestamp
            elif _due(track.interval_start, timestamp, self.timing.join_confirm_sec):
                self._confirm_join(track, timestamp, zone)
            return
        track.interval_start = None
        self._set_state(track, OUTSIDE, timestamp, zone)

    def _from_waiting(self, track: _Track, timestamp: float, zone: str) -> None:
        if zone == QUEUE:
            track.exit_interval_start = None
            return
        if zone == SERVICE:
            track.exit_interval_start = None
            track.interval_start = timestamp
            self._set_state(track, SERVICE_CANDIDATE, timestamp, zone)
            return
        if track.exit_interval_start is None:
            track.exit_interval_start = timestamp
            return
        if _due(track.exit_interval_start, timestamp, self.timing.exit_confirm_sec):
            self._abandon(track, timestamp, zone)

    def _from_service(self, track: _Track, timestamp: float, zone: str) -> None:
        if zone == SERVICE:
            if track.interval_start is None:
                track.interval_start = timestamp
            elif _due(track.interval_start, timestamp, self.timing.service_confirm_sec):
                self._confirm_service(track, timestamp, zone)
            return
        track.interval_start = None
        if zone == QUEUE:
            self._set_state(track, WAITING, timestamp, zone)
            track.exit_interval_start = None
            return
        self._set_state(track, WAITING, timestamp, zone)
        track.exit_interval_start = timestamp

    def _from_in_service(self, track: _Track, timestamp: float, zone: str) -> None:
        if zone == SERVICE:
            track.exit_interval_start = None
            return
        if track.exit_interval_start is None:
            track.exit_interval_start = timestamp
            return
        if _due(track.exit_interval_start, timestamp, self.timing.exit_confirm_sec):
            self._complete(track, timestamp, zone)

    def _confirm_join(self, track: _Track, timestamp: float, zone: str) -> None:
        self._sequence += 1
        visit_id = f"visit-{self._sequence:04d}"
        start = timestamp if track.interval_start is None else track.interval_start
        track.visit_id = visit_id
        track.service_start = None
        track.interval_start = None
        track.exit_interval_start = None
        self.visits[visit_id] = Visit(
            visit_id=visit_id,
            track_id=track.track_id,
            status="waiting",
            join_time=start,
        )
        self._set_state(track, WAITING, timestamp, zone)

    def _confirm_service(self, track: _Track, timestamp: float, zone: str) -> None:
        if track.visit_id is None:
            return
        start = timestamp if track.interval_start is None else track.interval_start
        visit = self.visits[track.visit_id]
        visit.service_start = start
        visit.waiting_time = start - visit.join_time
        visit.status = "in_service"
        track.service_start = start
        track.interval_start = None
        track.exit_interval_start = None
        self._set_state(track, IN_SERVICE, timestamp, zone)

    def _complete(self, track: _Track, timestamp: float, zone: str) -> None:
        if track.visit_id is None:
            return
        visit = self.visits[track.visit_id]
        end = timestamp if track.exit_interval_start is None else track.exit_interval_start
        visit.exit_interval_start = end
        visit.service_end = end
        if visit.service_start is not None:
            visit.service_duration = end - visit.service_start
        visit.closed_at = timestamp
        visit.status = "completed"
        track.interval_start = None
        track.exit_interval_start = None
        self._set_state(track, COMPLETED, timestamp, zone)

    def _abandon(self, track: _Track, timestamp: float, zone: str) -> None:
        if track.visit_id is None:
            return
        visit = self.visits[track.visit_id]
        visit.status = "abandoned"
        visit.exit_interval_start = track.exit_interval_start
        visit.closed_at = timestamp
        track.interval_start = None
        track.exit_interval_start = None
        self._set_state(track, ABANDONED, timestamp, zone)

    def _lose(self, track: _Track, timestamp: float) -> None:
        if track.visit_id is None:
            return
        visit = self.visits[track.visit_id]
        visit.status = "lost"
        visit.closed_at = timestamp
        track.interval_start = None
        track.exit_interval_start = None
        self._set_state(track, LOST, timestamp, None)

    def _set_state(self, track: _Track, new_state: str, timestamp: float, zone: str | None) -> None:
        old = track.state
        if old == new_state:
            return
        track.state = new_state
        event = Transition(timestamp, track.track_id, track.visit_id, old, new_state, zone)
        self.transitions.append(event)
        if self._current_new is not None:
            self._current_new.append(event)
