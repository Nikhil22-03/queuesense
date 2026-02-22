"""Video container tests. The moving rectangle is not a person and does not score the detector."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from queuesense.video import (
    VideoError,
    is_readable_video,
    iter_video_frames,
    next_timestamp,
    open_writer,
    output_fps,
    presentation_timestamps_reliable,
    probe_video,
    write_rectangle_video,
)


def test_timestamp_helpers_prefer_monotonic_presentation_time():
    assert presentation_timestamps_reliable([0.0, 40.0, 80.0])
    assert not presentation_timestamps_reliable([0.0, 0.0, 40.0])
    assert not presentation_timestamps_reliable([0.0, 40.0])
    mode, first = next_timestamp("presentation", 0, 10, 0.0, -1.0)
    assert (mode, first) == ("presentation", 0.0)
    mode, second = next_timestamp("presentation", 1, 10, 100.0, first)
    assert second == pytest.approx(0.1)
    mode, third = next_timestamp(mode, 2, 10, 50.0, second)
    assert mode == "frame_index"
    assert third > second


def test_rectangle_clip_roundtrip_does_not_claim_to_be_people(tmp_path):
    path = write_rectangle_video(tmp_path / "rects.mp4", frames=5, fps=10, size=(64, 48))
    info = probe_video(path)
    packets = list(iter_video_frames(path, stride=1))
    assert info.fps == pytest.approx(10)
    assert info.width == 64
    assert info.height == 48
    assert len(packets) == 5
    assert all(packets[i].timestamp < packets[i + 1].timestamp for i in range(4))
    assert packets[0].frame.shape[1] == 64
    assert int(packets[0].frame.reshape(-1, 3)[:, 2].max()) > 0
    assert is_readable_video(path)
    note = Path(path).read_bytes()[:4]
    assert note != b""  # the file has a header; readability is the real check


def test_stride_keeps_source_indexes_and_output_duration_math():
    # 6 source frames at 10 fps last 0.6 s. Stride 2 writes 3 frames at 5 fps, also 0.6 s.
    assert output_fps(10, 2) == pytest.approx(5)
    # The pure duration identity is what the writer uses; the fixture below checks indexes.
    assert 3 / output_fps(10, 2) == pytest.approx(6 / 10)


def test_stride_on_a_real_clip_processes_every_nth_frame(tmp_path):
    path = write_rectangle_video(tmp_path / "stride.mp4", frames=6, fps=10, size=(64, 48))
    packets = list(iter_video_frames(path, stride=2))
    assert [packet.index for packet in packets] == [0, 2, 4]
    assert all(left.timestamp < right.timestamp for left, right in zip(packets, packets[1:]))


def test_empty_file_is_not_treated_as_a_video(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"this is not a video container")
    assert is_readable_video(path) is False
    with pytest.raises(VideoError):
        probe_video(path)


def test_codec_failure_is_an_error_not_a_path(monkeypatch, tmp_path):
    class DeadWriter:
        def __init__(self, *args, **kwargs):
            pass

        def isOpened(self):
            return False

        def write(self, _frame):
            return False

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoWriter", DeadWriter)
    with pytest.raises(VideoError, match="codec"):
        open_writer(tmp_path / "out.mp4", 10, (64, 48))


def test_unreadable_probe_does_not_count_as_success(tmp_path, monkeypatch):
    # A writer that claims to open but produces nothing must be rejected.
    class BlankWriter:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)

        def isOpened(self):
            return True

        def write(self, _frame):
            self.path.write_bytes(b"")

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoWriter", BlankWriter)
    with pytest.raises(VideoError, match="codec"):
        open_writer(tmp_path / "out.mp4", 10, (64, 48))
    assert not any(tmp_path.glob("*.mp4")) or all(not is_readable_video(path) for path in tmp_path.glob("*.mp4"))
