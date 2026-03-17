import json
import pytest
from pathlib import Path
from app.settings import Settings


@pytest.fixture
def settings_file(tmp_path):
    return tmp_path / "settings.json"


def test_defaults_when_no_file(settings_file):
    s = Settings(settings_file)
    assert s.language == "en"
    assert s.output_dir is None
    assert s.chunk_size_sec == 600
    assert s.default_output_formats == ["txt", "srt"]


def test_save_and_reload(settings_file):
    s = Settings(settings_file)
    s.language = "it"
    s.save()
    s2 = Settings(settings_file)
    assert s2.language == "it"


def test_output_dir_stored_as_string(settings_file):
    s = Settings(settings_file)
    s.output_dir = Path("/some/dir")
    s.save()
    raw = json.loads(settings_file.read_text())
    assert raw["output_dir"] == "/some/dir"


def test_output_dir_loaded_as_path(settings_file):
    settings_file.write_text(json.dumps({"output_dir": "/some/dir"}))
    s = Settings(settings_file)
    assert isinstance(s.output_dir, Path)


def test_api_key_not_stored_in_settings(settings_file):
    s = Settings(settings_file)
    s.save()
    raw = json.loads(settings_file.read_text())
    assert "api_key" not in raw
    assert "OPENAI_API_KEY" not in raw
