# Contributing to easy-transcriber-stt

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Development Setup

**Requirements:** Python 3.10+, ffmpeg, Git

```bash
git clone https://github.com/agiuseppe28/easy-transcriber-stt.git
cd easy-transcriber-stt

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with at least one provider API key
```

---

## Running Tests

```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_audio.py -v
```

Tests use `pytest-asyncio` for async routes. No mocking of external providers — integration tests use real API calls where applicable (requires valid `.env`).

---

## Adding a New Provider

1. Create `app/providers/your_provider.py`
2. Extend the abstract base class in `app/providers/base.py`:
   ```python
   from app.providers.base import BaseProvider

   class YourProvider(BaseProvider):
       async def transcribe(self, audio_path: str, **kwargs) -> str:
           ...
   ```
3. Register the provider in `app/main.py` (follow the pattern of existing providers)
4. Add locale strings for the provider name/models in `app/locales/en.json` and `app/locales/it.json`
5. Add tests in `tests/providers/test_your_provider.py`

---

## Adding a Language

1. Copy `app/locales/en.json` to `app/locales/<lang>.json`
2. Translate all values (keep keys unchanged)
3. Register the new locale in `app/core/i18n.py` (follow the existing pattern)
4. Add the language option to the settings UI in `app/templates/index.html`

---

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Maintenance, tooling, config |
| `test:` | Tests only |
| `refactor:` | Code change with no behavior change |

Example: `feat: add Whisper.cpp local provider`

---

## Pull Request Checklist

Before opening a PR:

- [ ] Tests pass: `pytest tests/ -v`
- [ ] New features have tests
- [ ] `README.md` updated if user-facing behavior changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No API keys, secrets, or personal data in commits
- [ ] No CDN links added (all assets must be vendored in `app/static/`)
- [ ] `start.bat` / `start.sh` double-click startup still works
