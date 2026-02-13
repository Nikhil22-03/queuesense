"""Video probing, timestamps, and writers.

Presentation timestamps from OpenCV (CAP_PROP_POS_MSEC) are used when a short
probe shows they increase. Otherwise each frame is dated as frame_index / FPS.
Event logic always receives one of those source-video times, never the wall
clock used to measure processing speed.

Frame stride keeps the timestamp of each processed frame. Output FPS is
source FPS divided by the stride, so the annotated file lasts about as long
as the source. Frames that are skipped are invisible to the detector and the
state machine, and a stride longer than the grace period can mark a track LOST
after a single missed sample. The default stride is 1.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_CODEC_CANDIDATES = (
    ("mp4", "avc1"),
    ("mp4", "H264"),
    ("mp4", "mp4v"),
    ("avi", "XVID"),
    ("avi", "MJPG"),
)


class VideoError(RuntimeError):
    """A video could not be read or a readable output video could not be written."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    reported_frame_count: int | None
    duration_sec: float | None
    timestamp_mode: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FramePacket:
    index: int
    timestamp: float
    frame: np.ndarray
    timestamp_mode: str


def presentation_timestamps_reliable(samples_msec: list[float]) -> bool:
    """True when probe timestamps move forward and stay inside a one-day clip."""
    if len(samples_msec) < 3:
        return False
    if any(not math.isfinite(value) for value in samples_msec):
        return False
    if samples_msec[-1] <= 0 or samples_msec[-1] > 24 * 3600 * 1000:
        return False
    return all(later > earlier + 0.1 for earlier, later in zip(samples_msec, samples_msec[1:]))


def next_timestamp(
    mode: str,
    frame_index: int,
    fps: float,
    pos_msec: float,
    last_ts: float,
) -> tuple[str, float]:
    """Return the mode to keep using and a strictly increasing source timestamp."""
    step = 1.0 / fps if fps > 0 else 1e-3
    fallback = frame_index / fps if fps > 0 else float(frame_index)
    if mode == "presentation":
        candidate = float(pos_msec) / 1000.0
        jump_ok = last_ts < 0 or candidate - last_ts <= 5.0
        if math.isfinite(candidate) and candidate > last_ts + 1e-6 and jump_ok:
            return "presentation", candidate
        mode = "frame_index"
    timestamp = fallback
    if timestamp <= last_ts:
        timestamp = last_ts + step
    return mode, timestamp


def output_fps(source_fps: float, stride: int) -> float:
    if source_fps <= 0 or stride < 1:
        raise VideoError("Output FPS needs a positive source FPS and a stride of at least 1.")
    return float(source_fps) / float(stride)


def even_size(width: int, height: int) -> tuple[int, int]:
    return int(width) + (int(width) % 2), int(height) + (int(height) % 2)


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pad or resize a BGR frame to the even size a codec accepted."""
    width, height = size
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame_h, frame_w = frame.shape[:2]
    if frame_w > width or frame_h > height:
        return cv2.resize(frame, (width, height))
    pad_bottom = height - frame_h
    pad_right = width - frame_w
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))
    return np.ascontiguousarray(frame)


def is_readable_video(path: Path) -> bool:
    """True only when OpenCV can decode a non-empty frame from the file."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 128:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
    finally:
        capture.release()


def probe_video(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)))
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise VideoError(f"Could not read the first frame: {path}")
        samples = [float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)]
        for _ in range(7):
            ok, _extra = capture.read()
            if not ok:
                break
            samples.append(float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise VideoError(f"Video has invalid dimensions: {width}x{height}")
    if not math.isfinite(fps) or fps < 1 or fps > 240:
        raise VideoError(f"Video FPS must be between 1 and 240. This file reports {fps}.")
    mode = "presentation" if presentation_timestamps_reliable(samples) else "frame_index"
    frame_count = reported if reported > 0 else None
    duration = (frame_count / fps) if frame_count else None
    return VideoInfo(str(path), width, height, fps, frame_count, duration, mode)


def read_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(f"Could not open video: {path}")
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise VideoError(f"Could not read the first frame: {path}")
    return frame


def iter_video_frames(path: Path, stride: int = 1):
    """Yield processed frames with monotonic source timestamps."""
    if stride < 1:
        raise VideoError("Frame stride must be at least 1.")
    info = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoError(f"Could not open video: {path}")
    mode = info.timestamp_mode
    last_ts = -1.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            mode, timestamp = next_timestamp(mode, index, info.fps, pos_msec, last_ts)
            last_ts = timestamp
            if index % stride == 0:
                if frame.shape[1] != info.width or frame.shape[0] != info.height:
                    frame = cv2.resize(frame, (info.width, info.height))
                yield FramePacket(index, timestamp, frame, mode)
            index += 1
    finally:
        capture.release()


def _probe_codec(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> bool:
    if path.exists():
        path.unlink()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
    try:
        if not writer.isOpened():
            return False
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        writer.write(blank)
    finally:
        writer.release()
    readable = is_readable_video(path)
    if path.exists():
        path.unlink()
    return readable


def open_writer(destination: Path, fps: float, size: tuple[int, int]):
    """Open a VideoWriter whose codec can be read back. Raise if none can."""
    if fps <= 0:
        raise VideoError(f"Output FPS must be positive. Got {fps}.")
    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise VideoError(f"Output size must be positive even numbers. Got {width}x{height}.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    stem = destination.with_suffix("")
    for suffix, fourcc in _CODEC_CANDIDATES:
        out = Path(str(stem) + f".{suffix}")
        if not _probe_codec(out, fourcc, fps, size):
            errors.append(f"{fourcc} ({suffix}) could not write a readable probe")
            continue
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        if writer.isOpened():
            return writer, out, fourcc
        writer.release()
        errors.append(f"{fourcc} ({suffix}) did not stay open")
    detail = "; ".join(errors) if errors else "no codecs were attempted"
    raise VideoError(f"No working video codec. {detail}")


def prefer_browser_mp4(path: Path) -> tuple[Path, str]:
    """Transcode with ffmpeg when it is installed, so the dashboard can play H.264.

    OpenCV's mp4v files are valid and are what the readability check accepts.
    Some browsers still show a blank player for mp4v. A missing ffmpeg is not
    a failed analysis; the OpenCV file remains the output.
    """
    path = Path(path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return path, "opencv"
    target = path.with_name(f"{path.stem}_h264.mp4")
    if target.exists():
        target.unlink()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0 and is_readable_video(target):
        return target, "ffmpeg-libx264"
    if target.exists():
        target.unlink()
    return path, "opencv"


def write_rectangle_video(
    path: Path,
    *,
    frames: int = 8,
    fps: float = 10.0,
    size: tuple[int, int] = (96, 64),
) -> Path:
    """Write a short clip of a moving rectangle.

    The rectangles are a video-container fixture. They are not people and they
    do not measure detector accuracy.
    """
    if frames < 1:
        raise VideoError("A synthetic clip needs at least one frame.")
    path = Path(path)
    width, height = size
    out_size = even_size(width, height)
    writer, actual, _codec = open_writer(path, fps, out_size)
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 4 + index * max(1, width // max(frames, 1))
            x = min(x, max(0, width - 20))
            cv2.rectangle(frame, (x, 8), (min(width - 1, x + 18), min(height - 1, 40)), (0, 0, 255), -1)
            writer.write(fit_frame(frame, out_size))
    finally:
        writer.release()
    if not is_readable_video(actual):
        raise VideoError(f"Synthetic video was created at {actual} but OpenCV cannot read it back.")
    return actual
