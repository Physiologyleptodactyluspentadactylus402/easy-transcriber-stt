# GitHub Publication Design — easy-transcriber-stt

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Prepare the `easy-transcriber-stt` project for public GitHub publication with full documentation suite.

---

## 1. Repository Identity

- **Name:** `easy-transcriber-stt`
- **Account:** Personal GitHub account
- **Visibility:** Public
- **Initial tag:** `v0.9.0`
- **GitHub Topics:** `python`, `fastapi`, `whisper`, `speech-to-text`, `transcription`, `alpine-js`, `students`, `university`, `productivity`, `offline`, `italian`, `multilingual`

---

## 2. Pre-Publication Cleanup

### 2.1 Branch Rename

The local branch is `master`. Rename to `main` before pushing:

```bash
git branch -m master main
```

This must happen before `gh repo create` so the pushed branch is already `main`.

### 2.2 Files to Remove from Git Tracking

The following directories are **currently tracked** by git and must be removed and the deletion committed:

```bash
git rm -r docs/superpowers/
git rm -r .superpowers/
```

`.superpowers/` contains brainstorming HTML and server log artifacts from internal tooling (confirmed: 6 tracked files). These must not be published.

After running `git rm`, also physically delete any remaining untracked files in those directories to prevent accidental staging:

```bash
rm -rf docs/superpowers/
rm -rf .superpowers/
```

> Note: two plan files (`2026-03-17-live-transcription.md`, `2026-03-17-local-providers.md`) inside `docs/superpowers/plans/` are untracked but will be deleted by the `rm -rf` above. No separate action needed.

Stage and commit alongside the `.gitignore` update (see below):

```bash
git commit -m "chore: remove internal dev docs and add test audio to gitignore"
```

### 2.3 Test Audio Files

`test_audio.mp4` (33MB) and `test_audio.wav` are **untracked** (never committed). Add to `.gitignore` and stage the change as part of the cleanup commit:

```
# Test audio files
test_audio.*
```

```bash
git add .gitignore
# (alongside the git rm steps above, in one commit)
```

No `git rm` needed for the audio files themselves.

---

## 3. New Files to Create

### 3.1 `LICENSE`
MIT License, year 2026. Author name: read from `git config user.name` at implementation time.

### 3.2 `README.md` (rewrite)
Bilingual document — English first, Italian second.

**English sections:**
1. Project name + one-line description
2. Badges: version, license, Python version, platform
3. Features (bullet list)
4. Requirements (Python 3.10+, ffmpeg, at least one provider API key)
5. Quick Start (clone → `python start.py` or double-click `start.bat`/`start.sh`)
6. Providers table (Provider | Models | Requires API Key | Notes)
7. Configuration (`.env` setup, `settings.json` overview)
8. Documentation (links to `docs/`)
9. Contributing (link to `CONTRIBUTING.md`)
10. License

**Italian section** (after a horizontal rule):
- Same structure, more concise, defers to English for technical details.

### 3.3 `CHANGELOG.md`
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

Content at publication:
- `[Unreleased]` section (empty)
- `[0.9.0] - 2026-03-19` — comprehensive list of all features derived from git history

From v0.9.0 onwards, maintained manually before each release/tag.

### 3.4 `CONTRIBUTING.md`
Sections:
1. Development setup (clone, venv, pip install, ffmpeg)
2. Running tests (`pytest tests/`)
3. Adding a new provider — note: verify path `app/providers/base.py` exists at implementation time
4. Adding a language — note: verify path `app/locales/en.json` and `app/core/i18n.py` at implementation time
5. Commit conventions: `feat:`, `fix:`, `docs:`, `chore:`
6. PR checklist

### 3.5 `CLAUDE.md`
Instructions for Claude Code sessions in this repo:
- Stack overview
- Coding conventions
- How to run tests
- Hard constraints (no CDN, no Node.js, no API keys in commits, no breaking double-click startup)
- Key files map

### 3.6 `.github/ISSUE_TEMPLATE/bug_report.md`
Fields: description, steps to reproduce, expected vs actual behavior, OS, Python version, provider used, error log.

### 3.7 `.github/ISSUE_TEMPLATE/feature_request.md`
Fields: problem it solves, proposed solution, alternatives considered.

### 3.8 `.github/pull_request_template.md`
Checklist: change type, tests added/updated, docs updated, CHANGELOG updated.

---

## 4. `docs/` Public User Guides

All three files in English only.

### `docs/installation.md`
- Prerequisites (Python 3.10+, ffmpeg, Git)
- Step-by-step installation: Windows / macOS / Linux
- Verifying the installation
- Troubleshooting (ffmpeg not found, port in use, venv issues)

### `docs/providers.md`
- Comparison table (cost, quality, offline capability, supported languages)
- API key setup per provider (where to get it, how to add to `.env`)
- OpenAI model comparison (whisper-1 vs gpt-4o-transcribe vs gpt-4o-mini-transcribe)
- ElevenLabs model comparison (scribe_v2 vs scribe_v1)
- Local providers section (faster-whisper, Qwen3-ASR, Ollama): marked as **"Coming Soon"** — these are Plan 2 features not yet fully released. Section exists with a callout box only, no installation instructions yet.

### `docs/faq.md`
- How does offline mode work?
- Where are transcriptions saved?
- How do I change the interface language?
- What output formats are supported?
- How do I use the Audio Lab?
- Is my audio private?

---

## 5. GitHub Repository Setup

Execute in order after all local commits and the `main` rename:

```bash
# 1. Create repo and push
gh repo create easy-transcriber-stt --public --source=. --remote=origin --push

# 2. Set description
gh repo edit --description "🎙️ Easy, local-first audio transcription. OpenAI Whisper, ElevenLabs, offline providers. No cloud required."

# 3. Add topics (one per --add-topic flag)
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

# 4. Set default branch explicitly (safety measure — account default may differ)
gh repo edit --default-branch main

# 5. Create tag locally and push
git tag v0.9.0
git push origin v0.9.0

# 6. Create GitHub release from tag (inline notes, not --notes-file, to avoid
#    publishing the [Unreleased] header from CHANGELOG.md verbatim)
gh release create v0.9.0 \
  --title "v0.9.0 — Initial public release" \
  --notes "Initial public release. See [CHANGELOG.md](CHANGELOG.md) for full feature list." \
  --latest
```

Note: `gh repo create` with `--push` will push the current branch (`main` after rename). The `--default-branch main` step is added as a safety measure in case the GitHub account's default branch setting is not `main`.

---

## 6. Commit Strategy

All preparation work done locally on `main` (after rename), then pushed via `gh repo create`.

Commit order:
1. `chore: remove internal dev docs and add test audio to gitignore`
   - `git rm -r docs/superpowers/ .superpowers/` + `rm -rf docs/superpowers/ .superpowers/`
   - Edit `.gitignore` to add `test_audio.*`
   - `git add .gitignore && git commit`
2. `chore: add MIT license`
3. `docs: rewrite README bilingual EN+IT`
4. `docs: add CHANGELOG for v0.9.0`
5. `docs: add CONTRIBUTING.md and CLAUDE.md`
6. `chore: add GitHub issue and PR templates`
7. `docs: add user guides (installation, providers, faq)`

Tag `v0.9.0` applied after all commits, before push.

---

## 7. Out of Scope

- GitHub Actions / CI pipeline (can be added post-publication)
- `SECURITY.md` (can be added when security policy is defined)
- `CODE_OF_CONDUCT.md` (can be added if community grows)
- Automated CHANGELOG generation
