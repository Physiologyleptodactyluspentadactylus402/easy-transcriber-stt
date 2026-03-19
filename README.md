# easy-transcriber-stt

[![Version](https://img.shields.io/badge/version-v0.9.0-blue)](https://github.com/agiuseppe28/easy-transcriber-stt/releases)
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
git clone https://github.com/agiuseppe28/easy-transcriber-stt.git
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
git clone https://github.com/agiuseppe28/easy-transcriber-stt.git
cd easy-transcriber-stt
python start.py
```

Oppure doppio click su `start.bat` (Windows) / `start.sh` (macOS/Linux).

L'app si apre automaticamente nel browser. Al primo avvio, un wizard guida la configurazione del provider.

Per la documentazione completa vedi la [sezione inglese](#easy-transcriber-stt) sopra.
