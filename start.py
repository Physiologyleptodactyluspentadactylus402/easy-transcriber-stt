#!/usr/bin/env python3
import sys
import subprocess
import importlib
import shutil
import os
from pathlib import Path

REQUIRED_PACKAGES = [
    "customtkinter",
    "pydub",
    "python-dotenv",
    "openai",
]

THIS_DIR = Path(__file__).parent.resolve()
MAIN_APP = THIS_DIR / "main_ui.py"

def ensure_pip():
    try:
        import pip
        return True
    except Exception:
        print("[setup] pip non trovato. Provo a installarlo con ensurepip...")
        try:
            import ensurepip
            ensurepip.bootstrap()
            return True
        except Exception as e:
            print(f"[errore] impossibile installare pip automaticamente: {e}")
            return False

def install_missing(packages):
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except Exception:
            missing.append(pkg)

    if not missing:
        print("[ok] Tutti i pacchetti Python richiesti sono già presenti.")
        return True

    print(f"[setup] Installo pacchetti mancanti: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.run(cmd, check=False)

    cmd = [sys.executable, "-m", "pip", "install", *missing]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[errore] Installazione pacchetti fallita: {e}")
        return False

def check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        print(f"[ok] ffmpeg trovato: {ffmpeg}")
        return True
    print("[attenzione] ffmpeg e/o ffprobe non trovati nel PATH.")
    print(" - Installa ffmpeg (ad es. macOS: brew install ffmpeg, Ubuntu: sudo apt-get install ffmpeg, Windows: choco install ffmpeg)")
    return False

def load_env():
    try:
        from dotenv import load_dotenv
        env_path = THIS_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print("[ok] .env caricato")
    except Exception:
        pass

def launch_app():
    if not MAIN_APP.exists():
        print(f"[errore] File principale non trovato: {MAIN_APP}")
        sys.exit(1)
    env = os.environ.copy()
    print("[run] Avvio applicazione GUI…")
    subprocess.run([sys.executable, str(MAIN_APP)], env=env)

def main():
    if not ensure_pip():
        sys.exit(1)
    if not install_missing(REQUIRED_PACKAGES):
        sys.exit(1)
    ff_ok = check_ffmpeg()
    load_env()
    launch_app()
    if not ff_ok:
        print("\n[nota] L'app è partita, ma per processare audio serve installare ffmpeg.")

if __name__ == "__main__":
    main()
