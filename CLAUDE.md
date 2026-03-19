# CLAUDE.md — easy-transcriber-stt

Instructions for Claude Code sessions in this repository.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + uvicorn (Python 3.10+) |
| Frontend | Alpine.js 3.x + Tailwind CSS 3.x (vendored in `app/static/`) |
| Database | SQLite via `app/transcriber.db` |
| Transport | HTTP + WebSocket (real-time progress) |
| i18n | JSON locale files in `app/locales/` |

---

## Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, routes, WebSocket handlers |
| `app/settings.py` | Settings schema and manager |
| `app/core/audio.py` | Audio chunking and conversion |
| `app/core/history.py` | SQLite history management |
| `app/core/i18n.py` | Internationalization utilities |
| `app/core/live.py` | Live transcription stream handling |
| `app/core/output.py` | Output format generation (TXT, SRT, etc.) |
| `app/core/queue.py` | Transcription job queue |
| `app/core/preprocess.py` | Audio preprocessing and denoise dispatch |
| `app/providers/base.py` | Abstract base class for all providers |
| `app/templates/index.html` | Main Jinja2 template (Alpine.js UI) |
| `app/static/app.js` | Frontend application logic |
| `app/locales/en.json` | English translations |
| `app/locales/it.json` | Italian translations |
| `start.py` | Bootstrap script (dependency install, port discovery, browser launch) |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Hard Constraints

These rules must never be violated:

1. **No CDN links** — all JS/CSS assets must be vendored in `app/static/` and committed
2. **No Node.js required** — no npm, no webpack, no build step for end users
3. **No API keys in commits** — `.env` and `API KEY.txt` are gitignored; keep them local
4. **No breaking double-click startup** — `start.bat` / `start.sh` must always work without prior setup
5. **Python 3.10+ only** — use match/case, `|` union types, and other 3.10+ features freely
6. **Settings in `settings.json`, secrets in `.env`** — never mix them

---

## Conventions

- Commit messages: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`
- CHANGELOG.md: update `[Unreleased]` section before each release
- New providers: extend `app/providers/base.py`
- New languages: copy `app/locales/en.json`, translate, register in `app/core/i18n.py`
