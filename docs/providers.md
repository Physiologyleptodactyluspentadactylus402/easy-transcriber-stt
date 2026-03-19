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
