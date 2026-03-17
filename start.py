#!/usr/bin/env python3
"""
Transcriber bootstrapper.
Run this script to install dependencies and open the app in your browser.
"""
import subprocess
import sys
import socket
import webbrowser
import time
from pathlib import Path

MIN_PYTHON = (3, 10)
THIS_DIR = Path(__file__).parent.resolve()
STATIC_DIR = THIS_DIR / "app" / "static"

REQUIRED_ASSETS = [
    STATIC_DIR / "alpine.min.js",
    STATIC_DIR / "tailwind.cdn.min.js",
]


def check_python():
    if sys.version_info < MIN_PYTHON:
        print(f"[error] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
              f"You have {sys.version_info.major}.{sys.version_info.minor}.")
        print("  Download: https://www.python.org/downloads/")
        sys.exit(1)
    print(f"[ok] Python {sys.version_info.major}.{sys.version_info.minor}")


def install_requirements():
    req_file = THIS_DIR / "requirements.txt"
    print("[setup] Installing/verifying Python packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[error] Package installation failed. Check the output above.")
        sys.exit(1)
    print("[ok] Packages installed")


def check_static_assets():
    missing = [a for a in REQUIRED_ASSETS if not a.exists()]
    if missing:
        print(f"[error] Missing static assets: {[str(a.name) for a in missing]}")
        print("  Run: git checkout app/static/ OR re-clone the repository.")
        sys.exit(1)
    print("[ok] Static assets present")


def check_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        print("[ok] ffmpeg found")
    else:
        print("[warning] ffmpeg not found — audio conversion will not work.")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt-get install ffmpeg")
        print("  Windows: choco install ffmpeg   (or download from ffmpeg.org)")


def find_free_port(start: int = 8000, end: int = 8010) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range 8000-8010")


def load_dotenv():
    env_file = THIS_DIR / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv as _load
            _load(env_file)
            print("[ok] .env loaded")
        except ImportError:
            pass


def main():
    check_python()
    install_requirements()
    check_static_assets()
    check_ffmpeg()
    load_dotenv()

    port = find_free_port()
    url = f"http://localhost:{port}"
    print(f"\n[run] Starting Transcriber on {url}")
    print("      Press Ctrl+C to stop.\n")

    # Small delay then open browser
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Configure logging so job/install messages appear in terminal
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep uvicorn quiet (only warnings), but show our app logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Import here so packages are already installed above
    import uvicorn
    from app.main import create_app
    app = create_app(port=port)
    uvicorn.run(app, host="localhost", port=port, log_level="info")


if __name__ == "__main__":
    main()
