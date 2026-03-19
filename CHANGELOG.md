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
