# GitHub Publication Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and publish `easy-transcriber-stt` on GitHub with a complete documentation suite, clean git history, MIT license, and v0.9.0 release.

**Architecture:** Sequential cleanup + file creation + GitHub setup. No code changes — all work is git operations, file writes, and `gh` CLI commands. Each chunk is independent and can be committed separately.

**Tech Stack:** Git, GitHub CLI (`gh`), Markdown, MIT License

**Spec:** `docs/superpowers/specs/2026-03-19-github-publication-design.md`

---

## Chunk 1: Pre-Publication Cleanup

Files changed:
- Deleted: `docs/superpowers/` (entire directory — tracked in git)
- Deleted: `.superpowers/` (entire directory — tracked in git, contains brainstorm HTML/logs)
- Modified: `.gitignore` (add `test_audio.*`)

Note: `test_audio.mp4` and `test_audio.wav` are untracked — only need `.gitignore` entry. No `git rm` needed.
Note: The spec file itself (`docs/superpowers/specs/2026-03-19-github-publication-design.md`) is untracked and will be physically deleted along with the directory.

---

- [ ] **Step 1.1: Rename branch master → main**

```bash
git branch -m master main
```

Verify:
```bash
git branch
```
Expected: `* main`

- [ ] **Step 1.2: Remove tracked internal directories from git**

```bash
git rm -r docs/superpowers/
git rm -r .superpowers/
```

Expected output: lines like `rm 'docs/superpowers/specs/...'` and `rm '.superpowers/brainstorm/...'` (around 15 files total).

- [ ] **Step 1.3: Physically delete any remaining untracked files in those dirs**

```bash
rm -rf docs/superpowers/
rm -rf .superpowers/
```

- [ ] **Step 1.4: Add test audio files to .gitignore**

Open `.gitignore` and add at the end:

```
# Test audio files
test_audio.*
```

- [ ] **Step 1.5: Stage .gitignore and commit**

Note: `git rm` in Step 1.2 automatically stages the deletions. You only need to additionally stage the `.gitignore` change. The commit will include all three changes together (two deletions + gitignore update).

```bash
git add .gitignore
git commit -m "chore: remove internal dev docs and add test audio to gitignore"
```

Verify:
```bash
git ls-files docs/ .superpowers/
```
Expected: empty output (no tracked files remaining in those dirs).

```bash
git status
```
Expected: clean working tree (test_audio.* listed as ignored, not untracked).

---

## Chunk 2: LICENSE + README

Files changed:
- Create: `LICENSE`
- Modify: `README.md` (full rewrite)

---

- [ ] **Step 2.0: Confirm GitHub username**

`git config user.name` is not set in this repo. Confirm your GitHub username before writing the README badges:

```bash
gh api user --jq .login
```

Expected: your GitHub username (e.g. `agiuseppe28`). Use this value to replace `YOUR_GITHUB_USERNAME` in Step 2.3.

- [ ] **Step 2.1: Create LICENSE**

Note: `git config user.name` is not configured in this repo. The author name is hardcoded as "Giuseppe Ardoselli" based on the commit history.

Create file `LICENSE` with this exact content:

```
MIT License

Copyright (c) 2026 Giuseppe Ardoselli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2.2: Commit LICENSE**

```bash
git add LICENSE
git commit -m "chore: add MIT license"
```

- [ ] **Step 2.3: Rewrite README.md**

Replace the entire content of `README.md` with:

Replace `YOUR_GITHUB_USERNAME` with the value obtained in Step 2.0.

```markdown
# easy-transcriber-stt

[![Version](https://img.shields.io/badge/version-v0.9.0-blue)](https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()

> Local-first audio transcription tool. Upload audio or video, get a transcript in seconds. No cloud required if you use local providers.

---

## Features

- **Multi-provider transcription** — OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe), ElevenLabs (scribe_v2)
- **Local providers** *(optional)* — faster-whisper, Qwen3-ASR, Ollama — no API key required
- **Audio Lab** — advanced preprocessing: noise reduction, vocal isolation (Demucs), HQ pipeline
- **Dual denoise engine** — DeepFilterNet (AI-based) or ffmpeg afftdn (fast, no GPU required)
- **Real-time progress** — WebSocket-powered stepper with ETA
- **Transcription history** — stored locally in SQLite
- **Multiple output formats** — TXT, SRT, VTT, JSON
- **Bilingual UI** — English and Italian
- **No CDN, no Node.js** — all static assets vendored, works fully offline
- **Double-click startup** — `start.bat` (Windows) or `start.sh` (macOS/Linux)

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- At least one provider API key **or** a local provider installed (see [docs/providers.md](docs/providers.md))

---

## Quick Start

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt.git
cd easy-transcriber-stt
python start.py
```

Or double-click `start.bat` (Windows) / `start.sh` (macOS/Linux).

The app opens automatically in your browser. On first launch, a setup wizard will guide you through provider configuration.

---

## Providers

| Provider | Models | Requires API Key | Notes |
|----------|--------|-----------------|-------|
| OpenAI | whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe | Yes | Best accuracy |
| ElevenLabs | scribe_v2, scribe_v1 | Yes | Speaker diarization |
| faster-whisper | tiny → large-v3 | No | Local, GPU optional |
| Qwen3-ASR | 0.6B, 1.7B | No | Local, lightweight |
| Ollama | any speech model | No | Local, requires Ollama |

Full setup instructions: [docs/providers.md](docs/providers.md)

---

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

```env
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...
```

Settings (language, default provider, output formats) are managed through the UI and saved to `settings.json`.

---

## Documentation

- [Installation guide](docs/installation.md) — detailed setup for all platforms
- [Providers](docs/providers.md) — compare providers, get API keys, configure local models
- [FAQ](docs/faq.md) — common questions

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs are welcome.

---

## License

MIT — see [LICENSE](LICENSE).

---

---

## Italiano

> Strumento di trascrizione audio locale. Carica audio o video, ottieni una trascrizione in pochi secondi. Non richiede cloud se usi i provider locali.

### Caratteristiche

- **Trascrizione multi-provider** — OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe), ElevenLabs (scribe_v2)
- **Provider locali** *(opzionali)* — faster-whisper, Qwen3-ASR, Ollama — senza chiave API
- **Audio Lab** — preprocessing avanzato: riduzione rumore, isolamento voce (Demucs), pipeline HQ
- **Progresso in tempo reale** — stepper WebSocket con ETA
- **Storico trascrizioni** — salvato localmente in SQLite
- **Formati di output** — TXT, SRT, VTT, JSON
- **Interfaccia bilingue** — italiano e inglese
- **Nessun CDN, nessun Node.js** — funziona completamente offline
- **Avvio con doppio click** — `start.bat` (Windows) / `start.sh` (macOS/Linux)

### Requisiti

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installato e nel PATH
- Almeno una chiave API provider **oppure** un provider locale installato

### Avvio rapido

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt.git
cd easy-transcriber-stt
python start.py
```

Oppure doppio click su `start.bat` (Windows) / `start.sh` (macOS/Linux).

L'app si apre automaticamente nel browser. Al primo avvio, un wizard guida la configurazione del provider.

Per la documentazione completa vedi la [sezione inglese](#easy-transcriber-stt) sopra.
```

- [ ] **Step 2.4: Verify README renders correctly**

Open `README.md` in a markdown preview or check that all section headers are present:
```bash
grep "^## " README.md
```
Expected output:
```
## Features
## Requirements
## Quick Start
## Providers
## Configuration
## Documentation
## Contributing
## License
## Italiano
```

- [ ] **Step 2.5: Commit README**

```bash
git add README.md
git commit -m "docs: rewrite README bilingual EN+IT"
```

---

## Chunk 3: CHANGELOG + CONTRIBUTING + CLAUDE.md

Files changed:
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `CLAUDE.md`

---

- [ ] **Step 3.1: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-03-19

### Added

#### Core Transcription
- Multi-provider transcription: OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe) and ElevenLabs (scribe_v2, scribe_v1)
- Optional local providers: faster-whisper, Qwen3-ASR (0.6B / 1.7B), Ollama
- Multiple output formats: TXT, SRT, VTT, JSON
- Transcription history stored in local SQLite database
- Job queue for managing concurrent transcription requests

#### Audio Lab
- Advanced audio preprocessing pipeline (Audio Lab)
- HQ pipeline: chunked Demucs vocal separation + Polish step for high-quality output
- Dual denoise engine: DeepFilterNet (AI-based, GPU optional) and ffmpeg afftdn (fast, no GPU)
- Denoise engine selector in the UI with install guide for DeepFilterNet

#### UI & Progress
- Pipeline progress stepper with per-step ETA via WebSocket
- Real-time progress updates during transcription and audio processing
- Bilingual interface: English and Italian (i18n via JSON locale files)
- Setup wizard for first-run provider configuration

#### Infrastructure
- FastAPI backend with uvicorn ASGI server
- Alpine.js 3.x + Tailwind CSS 3.x frontend — both vendored (no CDN, no Node.js required)
- WebSocket channel for real-time progress and live transcription
- Settings manager: `settings.json` for preferences, `.env` for API keys
- Double-click startup: `start.bat` (Windows), `start.sh` (macOS/Linux), `start.py` (Python direct)
- Automatic dependency installation and browser launch on startup
```

- [ ] **Step 3.2: Create CONTRIBUTING.md**

Verify file paths before writing:
- `app/providers/base.py` — abstract provider base class
- `app/locales/en.json` — English locale file
- `app/core/i18n.py` — i18n utilities

```bash
ls app/providers/base.py app/locales/en.json app/core/i18n.py
```
Expected: all three files exist.

Then create `CONTRIBUTING.md`:

```markdown
# Contributing to easy-transcriber-stt

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Development Setup

**Requirements:** Python 3.10+, ffmpeg, Git

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt.git
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
```

- [ ] **Step 3.3: Create CLAUDE.md**

```markdown
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
```

- [ ] **Step 3.4: Commit CHANGELOG separately**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG for v0.9.0"
```

- [ ] **Step 3.5: Commit CONTRIBUTING and CLAUDE.md**

```bash
git add CONTRIBUTING.md CLAUDE.md
git commit -m "docs: add CONTRIBUTING.md and CLAUDE.md"
```

---

## Chunk 4: GitHub Issue & PR Templates

Files changed:
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/pull_request_template.md`

---

- [ ] **Step 4.1: Create `.github/ISSUE_TEMPLATE/bug_report.md`**

```markdown
---
name: Bug report
about: Report a problem with easy-transcriber-stt
title: '[BUG] '
labels: bug
assignees: ''
---

## Description

A clear description of what went wrong.

## Steps to Reproduce

1. Go to '...'
2. Upload file '...'
3. Click '...'
4. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include error messages or screenshots if possible.

## Environment

- **OS:** (e.g. Windows 11, macOS 14, Ubuntu 22.04)
- **Python version:** (run `python --version`)
- **Provider used:** (e.g. OpenAI whisper-1, ElevenLabs scribe_v2, faster-whisper)
- **easy-transcriber-stt version:** (e.g. v0.9.0)

## Error Log

Paste the terminal output or browser console error here:

```
(paste log here)
```
```

- [ ] **Step 4.2: Create `.github/ISSUE_TEMPLATE/feature_request.md`**

```markdown
---
name: Feature request
about: Suggest a new feature or improvement
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Problem

What problem does this feature solve? Who is affected?

## Proposed Solution

Describe the feature you'd like to see. Be as specific as possible.

## Alternatives Considered

Have you tried any workarounds? Are there other ways to solve this?

## Additional Context

Any mockups, links, or extra context that might help.
```

- [ ] **Step 4.3: Create `.github/pull_request_template.md`**

```markdown
## Summary

Brief description of what this PR does.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactor (no behavior change)
- [ ] Chore / maintenance

## Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] New/changed behavior has tests
- [ ] `README.md` updated if user-facing behavior changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No API keys or secrets in the diff
- [ ] No CDN links added (assets vendored in `app/static/`)
- [ ] Double-click startup (`start.bat` / `start.sh`) still works
```

- [ ] **Step 4.4: Commit templates**

```bash
git add .github/
git commit -m "chore: add GitHub issue and PR templates"
```

Verify:
```bash
git ls-files .github/
```
Expected:
```
.github/ISSUE_TEMPLATE/bug_report.md
.github/ISSUE_TEMPLATE/feature_request.md
.github/pull_request_template.md
```

---

## Chunk 5: User Documentation

Files changed:
- Create: `docs/installation.md`
- Create: `docs/providers.md`
- Create: `docs/faq.md`

---

- [ ] **Step 5.1: Create `docs/installation.md`**

```markdown
# Installation Guide

## Prerequisites

Before installing easy-transcriber-stt, make sure you have:

### Python 3.10+

Check your version:
```bash
python --version   # or python3 --version on macOS/Linux
```

Download from [python.org](https://www.python.org/downloads/) if needed.

### ffmpeg

ffmpeg is required for audio conversion and preprocessing.

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)
2. Extract the archive
3. Add the `bin/` folder to your PATH, or use [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

Verify: `ffmpeg -version`

### Git

Download from [git-scm.com](https://git-scm.com/) if not already installed.

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt.git
cd easy-transcriber-stt
```

### Step 2: Configure your API keys

```bash
cp .env.example .env
```

Edit `.env` with your provider keys (see [providers.md](providers.md) for where to get them):

```env
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=sk_...
```

You need **at least one** key, or a local provider installed (see [providers.md](providers.md)).

### Step 3: Start the app

**Option A — Python:**
```bash
python start.py
```

**Option B — Double-click (no terminal needed):**
- Windows: double-click `start.bat`
- macOS/Linux: double-click `start.sh` (may need to mark as executable first: `chmod +x start.sh`)

The app will:
1. Check Python version
2. Install dependencies automatically (first run only)
3. Open your browser at `http://localhost:<port>`

---

## Verifying the Installation

The browser should open to the easy-transcriber-stt interface. If it's your first run, a setup wizard will appear to configure your provider.

To verify everything works:
1. Upload a short audio or video file
2. Select a provider
3. Click "Transcribe"
4. You should see real-time progress and then a transcript

---

## Troubleshooting

### `ffmpeg: command not found`

ffmpeg is not in your PATH. See the ffmpeg installation section above. On Windows, restart your terminal after updating PATH.

### Port already in use

`start.py` automatically finds an available port. If you see a port conflict error, close any other running instance of the app and retry.

### Dependencies not installing

If `pip` fails during startup:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python start.py
```

### App opens but transcription fails

Check that your `.env` file has valid API keys. See [providers.md](providers.md).

### `ModuleNotFoundError` for optional packages

Some features (DeepFilterNet, Demucs, local providers) require optional dependencies. Install them only if you need them — see [providers.md](providers.md).
```

- [ ] **Step 5.2: Create `docs/providers.md`**

```markdown
# Providers Guide

## Comparison

| Provider | Type | API Key Required | Offline | Quality | Speed |
|----------|------|-----------------|---------|---------|-------|
| OpenAI whisper-1 | Cloud | Yes | No | Good | Fast |
| OpenAI gpt-4o-transcribe | Cloud | Yes | No | Excellent | Medium |
| OpenAI gpt-4o-mini-transcribe | Cloud | Yes | No | Very Good | Fast |
| ElevenLabs scribe_v2 | Cloud | Yes | No | Excellent + diarization | Medium |
| ElevenLabs scribe_v1 | Cloud | Yes | No | Good | Fast |
| faster-whisper | Local | No | Yes | Good–Excellent (size-dependent) | Medium–Fast |
| Qwen3-ASR 0.6B | Local | No | Yes | Good | Fast (CPU-friendly) |
| Qwen3-ASR 1.7B | Local | No | Yes | Very Good | Medium |
| Ollama | Local | No | Yes | Depends on model | Varies |

---

## Cloud Providers

### OpenAI

**Get an API key:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

Add to `.env`:
```env
OPENAI_API_KEY=sk-...
```

**Models:**
- `whisper-1` — Fast and reliable general-purpose transcription
- `gpt-4o-transcribe` — Highest accuracy, understands context better
- `gpt-4o-mini-transcribe` — Good accuracy at lower cost

### ElevenLabs

**Get an API key:** [elevenlabs.io](https://elevenlabs.io) → Profile → API Keys

Add to `.env`:
```env
ELEVENLABS_API_KEY=sk_...
```

**Models:**
- `scribe_v2` — Best model, includes speaker diarization (identifies who is speaking)
- `scribe_v1` — Legacy model, faster but less accurate

---

## Local Providers

> **Coming Soon** — Local providers (faster-whisper, Qwen3-ASR, Ollama) are available in the current build but full setup documentation and UI integration improvements are planned for the next release.

Local providers run entirely on your machine. No API key required. Audio never leaves your device.

**Hardware requirements vary:**
- faster-whisper large-v3: ~4GB RAM, GPU recommended
- Qwen3-ASR 0.6B: runs on CPU, ~2GB RAM
- Ollama: depends on the model

Installation instructions will be added in the next documentation update.

---

## Which Provider Should I Use?

- **Best accuracy:** OpenAI gpt-4o-transcribe or ElevenLabs scribe_v2
- **Best cost/quality ratio:** OpenAI whisper-1
- **Speaker identification needed:** ElevenLabs scribe_v2
- **Privacy-sensitive audio:** faster-whisper or Qwen3-ASR (local, no cloud)
- **No API key / fully offline:** faster-whisper, Qwen3-ASR, or Ollama
```

- [ ] **Step 5.3: Create `docs/faq.md`**

```markdown
# Frequently Asked Questions

## How does offline mode work?

Offline transcription requires a **local provider** (faster-whisper, Qwen3-ASR, or Ollama). Cloud providers (OpenAI, ElevenLabs) always need an internet connection. See [providers.md](providers.md) for setup instructions.

## Where are transcriptions saved?

Transcriptions are saved in two ways:
1. **Output files** — in the output directory configured in Settings (defaults to your Downloads folder or a folder you specify)
2. **History** — in a local SQLite database at `app/transcriber.db`, accessible from the History tab in the UI

Nothing is sent to any cloud service except what is strictly necessary to call the transcription API.

## How do I change the interface language?

Go to **Settings** in the app and select your language. Currently supported: English and Italian. The setting is saved immediately.

## What output formats are supported?

- **TXT** — plain text transcript
- **SRT** — SubRip subtitle format (with timestamps)
- **VTT** — WebVTT subtitle format
- **JSON** — structured output with timestamps and metadata

You can select one or more output formats before transcribing.

## How do I use the Audio Lab?

The Audio Lab is accessible from the main interface. It allows you to:
- Apply noise reduction before transcribing (recommended for recordings with background noise)
- Run vocal isolation with Demucs (removes music and background sounds)
- Use the HQ pipeline for the best possible audio quality before transcription

Note: Demucs requires additional dependencies. See [providers.md](providers.md).

## Is my audio private?

- **With cloud providers (OpenAI, ElevenLabs):** audio is sent to their servers for transcription, subject to their privacy policies
- **With local providers (faster-whisper, Qwen3-ASR, Ollama):** audio never leaves your machine
- **History and output files** are always stored locally only, never uploaded anywhere

## How do I update the app?

```bash
git pull
python start.py   # or double-click start.bat / start.sh
```

`start.py` automatically installs any new dependencies.

## Can I use this without an internet connection?

Yes, if you use a local provider. The app itself runs locally — only the transcription API calls require internet. See [providers.md](providers.md) for local provider setup.

## Something is broken. How do I get help?

Open an issue on GitHub: [github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt/issues](https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt/issues)

Please include your OS, Python version, provider used, and any error messages from the terminal.
```

- [ ] **Step 5.4: Commit user docs**

```bash
git add docs/
git commit -m "docs: add user guides (installation, providers, faq)"
```

Verify:
```bash
git ls-files docs/
```
Expected:
```
docs/faq.md
docs/installation.md
docs/providers.md
```

---

## Chunk 6: GitHub Repository Setup

No files changed. All operations are `git` and `gh` CLI commands.

Prerequisites: `gh` CLI installed and authenticated (`gh auth login`).

---

- [ ] **Step 6.1: Verify gh CLI is installed and authenticated**

```bash
gh auth status
```
Expected: `Logged in to github.com as <username>`

If not authenticated:
```bash
gh auth login
```
Follow the interactive prompt.

- [ ] **Step 6.2: Tag v0.9.0 locally**

```bash
git tag v0.9.0
```

Verify:
```bash
git tag
```
Expected: `v0.9.0` appears in the list.

- [ ] **Step 6.3: Create GitHub repository and push**

```bash
gh repo create easy-transcriber-stt --public --source=. --remote=origin --push
```

Expected: repo created, all commits and the `main` branch pushed. `gh` will print the repo URL.

- [ ] **Step 6.4: Push the tag**

```bash
git push origin v0.9.0
```

- [ ] **Step 6.5: Set default branch (safety)**

```bash
gh repo edit --default-branch main
```

- [ ] **Step 6.6: Set repo description**

```bash
gh repo edit --description "🎙️ Local-first audio transcription. OpenAI Whisper, ElevenLabs, offline providers. No cloud required."
```

- [ ] **Step 6.7: Add topics**

```bash
gh repo edit \
  --add-topic python \
  --add-topic fastapi \
  --add-topic whisper \
  --add-topic speech-to-text \
  --add-topic transcription \
  --add-topic alpine-js \
  --add-topic students \
  --add-topic university \
  --add-topic productivity \
  --add-topic offline \
  --add-topic italian \
  --add-topic multilingual
```

- [ ] **Step 6.8: Create GitHub release**

```bash
gh release create v0.9.0 \
  --title "v0.9.0 — Initial public release" \
  --notes "Initial public release of easy-transcriber-stt.

## Highlights
- Multi-provider transcription: OpenAI (whisper-1, gpt-4o-transcribe), ElevenLabs (scribe_v2)
- Optional local providers: faster-whisper, Qwen3-ASR, Ollama
- Audio Lab with dual denoise engine (DeepFilterNet + ffmpeg afftdn) and Demucs HQ pipeline
- Real-time progress stepper with ETA via WebSocket
- Bilingual UI (English + Italian), no CDN, no Node.js required
- Double-click startup on Windows, macOS, Linux

See [CHANGELOG.md](https://github.com/YOUR_GITHUB_USERNAME/easy-transcriber-stt/blob/main/CHANGELOG.md) for the full feature list." \
  --latest
```

- [ ] **Step 6.9: Verify the published repository**

```bash
gh repo view --web
```

Check manually:
- [ ] Description is set
- [ ] Topics are visible
- [ ] README renders correctly (bilingual, badges show)
- [ ] LICENSE file is detected by GitHub (shows "MIT license" badge)
- [ ] Release v0.9.0 appears under Releases
- [ ] `docs/superpowers/` and `.superpowers/` are NOT visible in the file tree
