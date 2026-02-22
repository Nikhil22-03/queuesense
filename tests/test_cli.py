"""CLI and import checks that do not download model weights."""

from queuesense.cli import build_parser
from queuesense.config import load_settings, project_root


def test_default_settings_match_the_documented_durations():
    settings = load_settings()
    assert settings.join_confirm_sec == 1.5
    assert settings.service_confirm_sec == 1.0
    assert settings.exit_confirm_sec == 2.0
    assert settings.missing_grace_sec == 3.0
    assert settings.min_service_samples == 3
    assert settings.frame_stride == 1
    assert settings.device == "auto"
    assert (project_root() / "configs" / "default.json").is_file()


def test_help_lists_process_and_evaluate():
    help_text = build_parser().format_help()
    assert "process" in help_text
    assert "evaluate" in help_text


def test_app_imports_without_running_the_page():
    import app

    assert callable(app.main)
