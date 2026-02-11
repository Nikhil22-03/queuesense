"""Causal single-server FIFO wait estimate.

The estimate uses only service durations that have already completed by the
current video timestamp. It is a running mean of those durations (an expanding
window over this run). A short clip often has only a few completed services, so
the window keeps every valid sample instead of dropping early ones and delaying
the three-sample warmup.

For a hypothetical new arrival:

    waiting visits × mean service duration
    + max(mean service duration − elapsed service of the person in service, 0)

"Waiting visits" is the active-waiting count supplied by the caller. That count
includes tracks missing inside the grace period. It is not the observed queue count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Prediction:
    timestamp: float
    status: str
    estimate_sec: float | None
    mean_service_sec: float | None
    samples_used: int
    waiting_count: int
    remaining_service_sec: float | None
    in_service_count: int
    one_server_ok: bool

    def as_dict(self) -> dict:
        return asdict(self)


class WaitEstimator:
    def __init__(self, min_samples: int = 3) -> None:
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        self.min_samples = int(min_samples)
        self.samples: list[float] = []

    def add_sample(self, duration: float) -> None:
        value = float(duration)
        if value < 0:
            raise ValueError("A service duration cannot be negative.")
        self.samples.append(value)

    def mean(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def predict(
        self,
        timestamp: float,
        waiting_count: int,
        in_service_elapsed: list[float],
    ) -> Prediction:
        elapsed = [max(0.0, float(value)) for value in in_service_elapsed]
        used = len(self.samples)
        mean = self.mean()
        one_server_ok = len(elapsed) <= 1
        waiting = int(waiting_count)
        if used < self.min_samples or mean is None:
            return Prediction(
                float(timestamp),
                "insufficient_data",
                None,
                mean,
                used,
                waiting,
                None,
                len(elapsed),
                one_server_ok,
            )
        if not one_server_ok:
            return Prediction(
                float(timestamp),
                "one_server_violation",
                None,
                mean,
                used,
                waiting,
                None,
                len(elapsed),
                False,
            )
        remaining = max(mean - elapsed[0], 0.0) if elapsed else 0.0
        estimate = waiting * mean + remaining
        return Prediction(
            float(timestamp),
            "ok",
            estimate,
            mean,
            used,
            waiting,
            remaining,
            len(elapsed),
            True,
        )
