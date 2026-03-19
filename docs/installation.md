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
git clone https://github.com/agiuseppe28/easy-transcriber-stt.git
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
