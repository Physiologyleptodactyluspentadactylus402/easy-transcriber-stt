from __future__ import annotations
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "settings.json"
_ENV_PATH = _PROJECT_ROOT / ".env"
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
        self.denoise_engine: str = data.get("denoise_engine", "ffmpeg")

    def save(self) -> None:
        data = {
            "language": self.language,
            "output_dir": self.output_dir.as_posix() if self.output_dir else None,
            "chunk_size_sec": self.chunk_size_sec,
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "default_output_formats": self.default_output_formats,
            "wizard_complete": self.wizard_complete,
            "denoise_engine": self.denoise_engine,
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

    # ------------------------------------------------------------------
    # API-key persistence  (.env file)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_env() -> dict[str, str]:
        """Parse the .env file into a dict (simple KEY=VALUE format)."""
        env: dict[str, str] = {}
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    # Strip optional surrounding quotes
                    value = value.strip().strip("'\"")
                    env[key.strip()] = value
        return env

    @staticmethod
    def _write_env(env: dict[str, str]) -> None:
        """Write the dict back to .env (preserving simple KEY=VALUE format)."""
        lines = [f"{k}={v}" for k, v in env.items() if v]
        _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def set_api_key(self, provider: str, key: str) -> None:
        """Persist an API key to .env and update the current process env."""
        var_map = {
            "openai": "OPENAI_API_KEY",
            "elevenlabs": "ELEVENLABS_API_KEY",
        }
        env_var = var_map.get(provider)
        if not env_var:
            raise ValueError(f"Unknown API-key provider: {provider}")

        # Update .env file
        env = self._read_env()
        if key:
            env[env_var] = key
        else:
            env.pop(env_var, None)
        self._write_env(env)

        # Also set in current process so it takes effect immediately
        if key:
            os.environ[env_var] = key
        else:
            os.environ.pop(env_var, None)

    def resolve_output_dir(self, input_file: Path | None = None) -> Path:
        """Return the effective output directory for a job."""
        if self.output_dir:
            return self.output_dir
        if input_file:
            return input_file.parent
        path = _LIVE_OUTPUT_DEFAULT
        path.mkdir(parents=True, exist_ok=True)
        return path
