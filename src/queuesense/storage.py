"""SQLite storage for runs. Each run id is unique and queries are parameterized."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from queuesense.config import project_root

_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    video_path TEXT NOT NULL,
    video_filename TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    annotated_video_path TEXT,
    width INTEGER,
    height INTEGER,
    fps REAL,
    reported_frame_count INTEGER,
    duration_sec REAL,
    timestamp_mode TEXT,
    frames_processed INTEGER,
    wall_time_sec REAL,
    processing_fps REAL,
    config_json TEXT NOT NULL,
    zones_json TEXT NOT NULL,
    model_json TEXT NOT NULL,
    hardware_json TEXT NOT NULL,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    visit_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    join_time REAL,
    service_start REAL,
    service_end REAL,
    waiting_time REAL,
    service_duration REAL,
    exit_interval_start REAL,
    closed_at REAL,
    UNIQUE (run_id, visit_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    track_id INTEGER NOT NULL,
    visit_id TEXT,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    zone TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    observed_queue_count INTEGER NOT NULL,
    active_waiting_count INTEGER NOT NULL,
    in_service_count INTEGER NOT NULL,
    prediction_sec REAL,
    prediction_status TEXT NOT NULL,
    mean_service_sec REAL,
    samples_used INTEGER NOT NULL,
    remaining_service_sec REAL,
    one_server_ok INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_time ON metrics(run_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_visits_run ON visits(run_id);
CREATE INDEX IF NOT EXISTS idx_transitions_run ON transitions(run_id);
"""


def sanitize_filename(name: str) -> str:
    """Keep the final path component and a small set of filename characters."""
    base = Path(str(name).replace("\x00", "")).name
    cleaned = _FILENAME.sub("_", base).strip("._")
    if not cleaned:
        cleaned = "video.mp4"
    return cleaned[:120]


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def allocate_run_dir(root: Path, run_id: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / run_id
    if path.exists():
        path = root / f"{run_id}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True)
    return path


def relativize(path: Path) -> str:
    resolved = Path(path).resolve()
    root = project_root().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def resolve_stored_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root() / path


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_run(self, row: dict) -> None:
        columns = [
            "run_id",
            "created_at",
            "status",
            "error",
            "video_path",
            "video_filename",
            "output_dir",
            "annotated_video_path",
            "width",
            "height",
            "fps",
            "reported_frame_count",
            "duration_sec",
            "timestamp_mode",
            "frames_processed",
            "wall_time_sec",
            "processing_fps",
            "config_json",
            "zones_json",
            "model_json",
            "hardware_json",
            "summary_json",
        ]
        payload = {column: row.get(column) for column in columns}
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO runs ({', '.join(columns)}) VALUES ({placeholders})"
        with self._connect() as connection:
            connection.execute(sql, [payload[column] for column in columns])

    def fail_run(self, run_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, error = ? WHERE run_id = ?",
                ("failed", message[:2000], run_id),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None,
        annotated_video_path: str | None,
        frames_processed: int,
        wall_time_sec: float,
        processing_fps: float | None,
        timestamp_mode: str,
        summary: dict,
        visits: list[dict],
        transitions: list[dict],
        metrics: list[dict],
    ) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM visits WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM transitions WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            connection.executemany(
                """
                INSERT INTO visits (
                    run_id, visit_id, track_id, status, join_time, service_start, service_end,
                    waiting_time, service_duration, exit_interval_start, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item["visit_id"],
                        item["track_id"],
                        item["status"],
                        item.get("join_time"),
                        item.get("service_start"),
                        item.get("service_end"),
                        item.get("waiting_time"),
                        item.get("service_duration"),
                        item.get("exit_interval_start"),
                        item.get("closed_at"),
                    )
                    for item in visits
                ],
            )
            connection.executemany(
                """
                INSERT INTO transitions (run_id, timestamp, track_id, visit_id, from_state, to_state, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item["timestamp"],
                        item["track_id"],
                        item.get("visit_id"),
                        item["from_state"],
                        item["to_state"],
                        item.get("zone"),
                    )
                    for item in transitions
                ],
            )
            connection.executemany(
                """
                INSERT INTO metrics (
                    run_id, timestamp, observed_queue_count, active_waiting_count, in_service_count,
                    prediction_sec, prediction_status, mean_service_sec, samples_used,
                    remaining_service_sec, one_server_ok
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item["timestamp"],
                        item["observed_queue_count"],
                        item["active_waiting_count"],
                        item["in_service_count"],
                        item.get("prediction_sec"),
                        item["prediction_status"],
                        item.get("mean_service_sec"),
                        item["samples_used"],
                        item.get("remaining_service_sec"),
                        1 if item.get("one_server_ok") else 0,
                    )
                    for item in metrics
                ],
            )
            connection.execute(
                """
                UPDATE runs
                SET status = ?, error = ?, annotated_video_path = ?, frames_processed = ?,
                    wall_time_sec = ?, processing_fps = ?, timestamp_mode = ?, summary_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    error,
                    annotated_video_path,
                    frames_processed,
                    wall_time_sec,
                    processing_fps,
                    timestamp_mode,
                    json.dumps(summary),
                    run_id,
                ),
            )

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_visits(self, run_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM visits WHERE run_id = ? ORDER BY join_time, visit_id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_transitions(self, run_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM transitions WHERE run_id = ? ORDER BY timestamp, id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_metrics(self, run_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY timestamp, id",
                (run_id,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["one_server_ok"] = bool(item["one_server_ok"])
            records.append(item)
        return records
