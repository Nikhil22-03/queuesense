"""Overlay layout. Does not load a detector."""

from queuesense.visualization import hud_baselines


def test_status_lines_stay_separated_on_a_720p_frame():
    baselines = hud_baselines(1280, 720, 3)
    assert baselines == sorted(baselines)
    gaps = [baselines[index + 1] - baselines[index] for index in range(2)]
    assert min(gaps) >= 24
    assert baselines[-1] < 720
