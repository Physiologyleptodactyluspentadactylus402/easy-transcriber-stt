from __future__ import annotations
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "settings.json"
_DB_PATH = _PROJECT_ROOT / "app" / "transcriber.db"
_LIVE_OUTPUT_DEFAULT = Path.home() / "Documents" / "Transcriber"


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _SETTINGS_PATH
        data: dict = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        self.language: str = data.get("language", "en")
        raw_dir = data.get("output_dir")
        self.output_dir: Path | None = Path(raw_dir) if raw_dir else None
        self.chunk_size_sec: int = data.get("chunk_size_sec", 600)
        self.default_provider: str = data.get("default_provider", "openai")
        self.default_model: str = data.get("default_model", "whisper-1")
        self.default_output_formats: list[str] = data.get(
            "default_output_formats", ["txt", "srt"]
        )
        self.wizard_complete: bool = data.get("wizard_complete", False)

    def save(self) -> None:
        data = {
            "language": self.language,
            "output_dir": self.output_dir.as_posix() if self.output_dir else None,
            "chunk_size_sec": self.chunk_size_sec,
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "default_output_formats": self.default_output_formats,
            "wizard_complete": self.wizard_complete,
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def elevenlabs_api_key(self) -> str | None:
        return os.getenv("ELEVENLABS_API_KEY")

    @property
    def db_path(self) -> Path:
        return _DB_PATH

    def resolve_output_dir(self, input_file: Path | None = None) -> Path:
        """Return the effective output directory for a job."""
        if self.output_dir:
            return self.output_dir
        if input_file:
            return input_file.parent
        path = _LIVE_OUTPUT_DEFAULT
        path.mkdir(parents=True, exist_ok=True)
        return path
