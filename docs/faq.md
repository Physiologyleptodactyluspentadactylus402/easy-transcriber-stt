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

Open an issue on GitHub: [github.com/agiuseppe28/easy-transcriber-stt/issues](https://github.com/agiuseppe28/easy-transcriber-stt/issues)

Please include your OS, Python version, provider used, and any error messages from the terminal.
