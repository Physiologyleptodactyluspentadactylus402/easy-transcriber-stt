from __future__ import annotations
import asyncio
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field as dc_field
from pathlib import Path

logger = logging.getLogger("transcriber")
from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from app.settings import Settings
from app.core.history import History
from app.core.i18n import load_locale
from app.core.queue import JobQueue
from app.providers.base import Job, TranscribeOptions
from app.providers.openai import OpenAIProvider
from app.providers.elevenlabs import ElevenLabsProvider

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_AUDIO_CHUNKS_DIR = Path(__file__).parent.parent / "audio_chunks"


def create_app(
    settings_path: Path | None = None,
    db_path: Path | None = None,
    port: int = 8000,
) -> FastAPI:
    settings = Settings(settings_path)
    history = History(db_path or settings.db_path)
    queue = JobQueue()
    from app.core.live import LiveSessionManager
    _live_sessions = LiveSessionManager()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _get_providers() -> dict[str, object]:
        from app.providers.faster_whisper import FasterWhisperProvider
        from app.providers.qwen3_asr import Qwen3ASRProvider
        from app.providers.ollama import OllamaProvider
        return {
            "openai": OpenAIProvider(api_key=settings.openai_api_key),
            "elevenlabs": ElevenLabsProvider(api_key=settings.elevenlabs_api_key),
            "faster_whisper": FasterWhisperProvider(),
            "qwen3_asr": Qwen3ASRProvider(),
            "ollama": OllamaProvider(),
        }

    app = FastAPI(title="Transcriber")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "port": port,
            "language": settings.language,
            "wizard_complete": settings.wizard_complete,
        })

    @app.get("/api/locale")
    async def get_locale(lang: str = "en"):
        return load_locale(lang)

    @app.get("/api/providers")
    async def get_providers():
        providers = _get_providers()
        result = []
        for name, p in providers.items():
            result.append({
                "name": name,
                "available": p.is_available(),
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "hardware_hint": m.hardware_hint.value,
                        "supports_live": m.supports_live,
                        "supports_speaker_labels": m.supports_speaker_labels,
                        "supports_timestamps": m.supports_timestamps,
                    }
                    for m in p.models
                ],
            })
        return result

    @app.get("/api/settings")
    async def get_settings():
        return {
            "language": settings.language,
            "output_dir": str(settings.output_dir) if settings.output_dir else None,
            "chunk_size_sec": settings.chunk_size_sec,
            "default_provider": settings.default_provider,
            "default_model": settings.default_model,
            "default_output_formats": settings.default_output_formats,
            "wizard_complete": settings.wizard_complete,
            "denoise_engine": settings.denoise_engine,
            "openai_key_set": bool(settings.openai_api_key),
            "elevenlabs_key_set": bool(settings.elevenlabs_api_key),
        }

    @app.patch("/api/settings")
    async def update_settings(body: dict):
        allowed = {
            "language", "output_dir", "chunk_size_sec",
            "default_provider", "default_model",
            "default_output_formats", "wizard_complete",
            "denoise_engine",
        }
        for key, value in body.items():
            if key in allowed:
                setattr(settings, key, value)
        settings.save()
        return {"ok": True}

    @app.post("/api/apikey")
    async def set_api_key(body: dict):
        """Save an API key to .env and update the current process."""
        provider = body.get("provider", "")
        key = body.get("key", "").strip()
        try:
            settings.set_api_key(provider, key)
        except ValueError as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "ok": True,
            "openai_key_set": bool(settings.openai_api_key),
            "elevenlabs_key_set": bool(settings.elevenlabs_api_key),
        }

    @app.get("/api/history")
    async def get_history():
        return history.list()

    @app.delete("/api/history/{session_id}")
    async def delete_history(session_id: str):
        history.delete(session_id)
        return {"ok": True}

    @app.post("/api/install/{provider_name}")
    async def install_provider(provider_name: str):
        from fastapi import HTTPException
        providers = _get_providers()
        provider = providers.get(provider_name)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider not found")
        asyncio.create_task(
            _run_install(provider_name, provider, _ws_manager)
        )
        return {"ok": True}

    @app.get("/api/ffmpeg")
    async def check_ffmpeg():
        import shutil
        return {"available": shutil.which("ffmpeg") is not None}

    # ── Audio Lab dependency management ───────────────────────────
    @app.get("/api/audiolab/deps")
    async def audiolab_deps():
        """Return installation status of optional Audio Lab tools."""
        from app.core.preprocess import _DEMUCS_AVAILABLE, _DEEPFILTER_AVAILABLE
        return {
            "demucs": _DEMUCS_AVAILABLE,
            "deepfilter": _DEEPFILTER_AVAILABLE,
        }

    _AUDIOLAB_PACKAGES = {
        "demucs": {"pip": "demucs>=4.0.0", "label": "Demucs", "size": "~2 GB (includes PyTorch)"},
        "deepfilter": {"pip": "deepfilternet>=0.5.0", "label": "DeepFilterNet", "size": "~200 MB — requires Rust compiler (rustup.rs)"},
    }

    @app.post("/api/audiolab/install/{tool_name}")
    async def audiolab_install(tool_name: str):
        from fastapi import HTTPException
        pkg = _AUDIOLAB_PACKAGES.get(tool_name)
        if not pkg:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        asyncio.create_task(_run_audiolab_install(tool_name, pkg, _ws_manager))
        return {"ok": True}

    @app.get("/api/download")
    async def download_file(path: str):
        from fastapi import HTTPException
        import tempfile
        file_path = Path(path).resolve()
        # Only serve files within known output roots (prevent path traversal)
        allowed_roots = [
            *(
                [settings.output_dir.resolve()] if settings.output_dir else []
            ),
            (Path.home() / "Documents" / "Transcriber").resolve(),
            # Also allow system temp dir — older jobs may have written there
            Path(tempfile.gettempdir()).resolve(),
        ]
        if not any(_path_within(file_path, root) for root in allowed_roots):
            logger.warning(
                "Download denied: %s not within allowed roots %s",
                file_path,
                [str(r) for r in allowed_roots],
            )
            raise HTTPException(status_code=403, detail="Access denied")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(file_path), filename=file_path.name)

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile],
        provider_name: str = Form(...),
        model_id: str = Form(...),
        output_formats: str = Form('["txt"]'),
        merge_output: str = Form("false"),
        prompt: str = Form(""),
        language: str = Form(""),
        speaker_labels: str = Form("false"),
    ):
        # Save uploaded files to temp dir
        tmp_dir = Path(tempfile.mkdtemp())
        saved_files: list[Path] = []
        for upload in files:
            dest = tmp_dir / upload.filename
            dest.write_bytes(await upload.read())
            saved_files.append(dest)

        opts = TranscribeOptions(
            language=language or None,
            prompt=prompt,
            speaker_labels=speaker_labels.lower() == "true",
            output_formats=json.loads(output_formats),
            chunk_size_sec=settings.chunk_size_sec,
            model_id=model_id,
        )
        job = Job(
            input_files=saved_files,
            opts=opts,
            provider_name=provider_name,
            model_id=model_id,
            status="pending",
            merge_output=merge_output.lower() == "true",
        )
        queue.add(job)

        # Run job in background
        asyncio.create_task(_run_job(job, settings, _get_providers(), history, _ws_manager))
        return {"job_id": job.id}

    # WebSocket connection manager
    class ConnectionManager:
        def __init__(self):
            self.all_connections: set[WebSocket] = set()
            self.job_connections: dict[str, set[WebSocket]] = {}

        async def connect(self, ws: WebSocket) -> None:
            """Register a new connection globally (before any job subscription)."""
            self.all_connections.add(ws)

        def subscribe(self, ws: WebSocket, job_id: str) -> None:
            """Subscribe a connection to a specific job's updates."""
            self.job_connections.setdefault(job_id, set()).add(ws)

        def disconnect(self, ws: WebSocket, job_id: str | None = None) -> None:
            self.all_connections.discard(ws)
            if job_id and job_id in self.job_connections:
                self.job_connections[job_id].discard(ws)

        async def broadcast(self, job_id: str, msg: dict) -> None:
            for ws in list(self.job_connections.get(job_id, [])):
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

        async def broadcast_global(self, msg: dict) -> None:
            """Send a message to every connected WebSocket (e.g. install progress)."""
            for ws in list(self.all_connections):
                try:
                    await ws.send_json(msg)
                except Exception:
                    pass

    _ws_manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        import base64
        import uuid
        await ws.accept()
        await _ws_manager.connect(ws)
        job_id = None
        live_session_id = None

        # Background task: send keepalive pings every 25s to prevent
        # browser/proxy from closing idle connections during long jobs.
        async def _keepalive():
            try:
                while True:
                    await asyncio.sleep(25)
                    await ws.send_json({"type": "pong"})
            except Exception:
                pass

        keepalive_task = asyncio.create_task(_keepalive())

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "subscribe":
                    job_id = data["job_id"]
                    _ws_manager.subscribe(ws, job_id)

                elif msg_type == "cancel" and job_id:
                    queue.cancel(job_id)

                elif msg_type == "start_live":
                    live_session_id = str(uuid.uuid4())
                    opts_raw = data.get("opts", {})
                    opts = TranscribeOptions(
                        language=opts_raw.get("language"),
                        prompt=opts_raw.get("prompt", ""),
                        speaker_labels=bool(opts_raw.get("speaker_labels", False)),
                        output_formats=opts_raw.get("output_formats", ["txt"]),
                        chunk_size_sec=settings.chunk_size_sec,
                        model_id=data.get("model_id", "tiny"),
                    )
                    provider = _get_providers().get(data.get("provider_name", "faster_whisper"))
                    if provider is None or not provider.is_available():
                        await ws.send_json({
                            "type": "error",
                            "message": f"Provider '{data.get('provider_name')}' is not available.",
                        })
                        continue
                    out_dir = settings.resolve_output_dir()
                    from app.core.live import LiveSession
                    session = LiveSession(live_session_id, provider, opts, out_dir)
                    _live_sessions.add(session)
                    await session.start(_ws_manager)
                    await ws.send_json({
                        "type": "live_session_started",
                        "session_id": live_session_id,
                    })

                elif msg_type == "audio_chunk":
                    sid = data.get("session_id")
                    session = _live_sessions.get(sid) if sid else None
                    if session:
                        raw = base64.b64decode(data.get("data", ""))
                        session.add_chunk(raw)

                elif msg_type == "stop_live":
                    sid = data.get("session_id")
                    session = _live_sessions.get(sid) if sid else None
                    if session:
                        output_files = await session.stop(_ws_manager)
                        _live_sessions.remove(sid)
                        await ws.send_json({
                            "type": "live_session_stopped",
                            "session_id": sid,
                            "output_files": [str(p) for p in output_files],
                        })

        except WebSocketDisconnect:
            keepalive_task.cancel()
            _ws_manager.disconnect(ws, job_id)
            if live_session_id:
                session = _live_sessions.get(live_session_id)
                if session:
                    try:
                        await session.stop(_ws_manager)
                    finally:
                        _live_sessions.remove(live_session_id)

    # --- Audio Lab ---
    @dataclass
    class AudioLabJob:
        id: str
        status: str = "processing"
        original_path: Path | None = None
        processed_path: Path | None = None
        processed_48k_path: Path | None = None
        progress: float = 0.0
        current_step: str = "decode"
        message: str = ""
        cancelled: bool = False
        stats: dict | None = None
        original_filename: str = ""

    _audiolab_jobs: dict[str, AudioLabJob] = {}

    @app.post("/api/audiolab/process")
    async def audiolab_process(
        file: UploadFile,
        preset: str = Form("lecture"),
        loudnorm: bool = Form(True),
        loudnorm_target: float = Form(-16.0),
        voice_isolation: bool = Form(True),
        denoise: bool = Form(True),
        denoise_engine: str = Form("ffmpeg"),
        polish: bool = Form(False),
    ):
        import uuid
        job_id = str(uuid.uuid4())

        # Save uploaded file to temp dir
        cache_dir = Path(tempfile.gettempdir()) / "transcriber_audiolab" / job_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        input_path = cache_dir / (file.filename or "upload.wav")
        content = await file.read()
        input_path.write_bytes(content)

        # Resolve preset to config
        from app.core.preprocess import PreprocessConfig
        if preset == "lecture":
            config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                                      voice_isolation=True, denoise=True,
                                      denoise_engine=denoise_engine, polish=False)
        elif preset == "hq":
            config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                                      voice_isolation=True, denoise=True,
                                      denoise_engine=denoise_engine, polish=True)
        elif preset == "clean":
            config = PreprocessConfig(loudnorm=True, loudnorm_target=-16.0,
                                      voice_isolation=False, denoise=False,
                                      denoise_engine=denoise_engine, polish=False)
        else:  # custom
            config = PreprocessConfig(
                loudnorm=loudnorm,
                loudnorm_target=loudnorm_target,
                voice_isolation=voice_isolation,
                denoise=denoise,
                denoise_engine=denoise_engine,
                polish=polish,
            )

        job = AudioLabJob(id=job_id, original_filename=file.filename or "upload")
        _audiolab_jobs[job_id] = job

        # Run pipeline in background
        asyncio.create_task(_run_audiolab(job, input_path, cache_dir, config))

        return {"job_id": job_id, "status": "processing"}

    @app.post("/api/audiolab/cancel/{job_id}")
    async def audiolab_cancel(job_id: str):
        from fastapi import HTTPException
        job = _audiolab_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in ("done", "error"):
            raise HTTPException(status_code=409, detail="Job already finished")
        job.cancelled = True
        return {"status": "cancelling"}

    @app.get("/api/audiolab/preview/{job_id}")
    async def audiolab_preview(job_id: str, which: str = "original"):
        from fastapi import HTTPException
        job = _audiolab_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if which == "processed" and job.processed_48k_path and job.processed_48k_path.exists():
            return FileResponse(str(job.processed_48k_path), media_type="audio/wav")
        elif which == "original" and job.original_path and job.original_path.exists():
            return FileResponse(str(job.original_path), media_type="audio/wav")
        raise HTTPException(status_code=404, detail="File not ready")

    @app.get("/api/audiolab/download/{job_id}")
    async def audiolab_download(job_id: str):
        from fastapi import HTTPException
        job = _audiolab_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.processed_path or not job.processed_path.exists():
            raise HTTPException(status_code=404, detail="Processed file not ready")
        name = f"{Path(job.original_filename).stem} (processed).wav"
        return FileResponse(str(job.processed_path), filename=name, media_type="audio/wav")

    @app.post("/api/audiolab/send-to-transcribe")
    async def audiolab_send_to_transcribe(request: Request):
        from fastapi import HTTPException
        body = await request.json()
        al_job_id = body.get("job_id")
        al_job = _audiolab_jobs.get(al_job_id)
        if not al_job or not al_job.processed_path:
            raise HTTPException(status_code=404, detail="Audio Lab job not found or not done")

        # Create a transcription job using the processed file
        opts = TranscribeOptions(
            model_id=settings.default_model or "whisper-1",
            output_formats=settings.default_output_formats,
            chunk_size_sec=settings.chunk_size_sec,
        )
        job = Job(
            provider_name=settings.default_provider or "openai",
            model_id=settings.default_model or "whisper-1",
            input_files=[al_job.processed_path],
            opts=opts,
        )
        queue.add(job)
        asyncio.create_task(_run_job(job, settings, _get_providers(), history, _ws_manager))

        filename = f"{Path(al_job.original_filename).stem} (processed).wav"
        return {"transcription_job_id": job.id, "filename": filename}

    async def _run_audiolab(job, input_path, cache_dir, config):
        """Run the preprocessing pipeline in a background thread."""
        from app.core.preprocess import run_pipeline, build_step_list
        loop = asyncio.get_running_loop()

        step_list = build_step_list(config)
        await _ws_manager.broadcast_global({
            "type": "audiolab_steps",
            "job_id": job.id,
            "steps": step_list,
        })

        def _progress(frac, step, msg, **kwargs):
            job.progress = frac
            job.current_step = step
            job.message = msg
            loop.call_soon_threadsafe(
                asyncio.ensure_future,
                _ws_manager.broadcast_global({
                    "type": "audiolab_progress",
                    "job_id": job.id,
                    "progress": frac,
                    "step": step,
                    "message": msg,
                    "sub_progress": kwargs.get("sub_progress"),
                    "sub_label": kwargs.get("sub_label"),
                    "elapsed_sec": kwargs.get("elapsed_sec", 0),
                    "eta_sec": kwargs.get("eta_sec"),
                    "completed_step": kwargs.get("completed_step"),
                    "completed_step_elapsed": kwargs.get("completed_step_elapsed"),
                }),
            )

        cancel_flag = {"cancelled": False}

        try:
            def _run():
                import threading, time
                def _sync():
                    while not cancel_flag.get("done"):
                        cancel_flag["cancelled"] = job.cancelled
                        time.sleep(0.5)
                t = threading.Thread(target=_sync, daemon=True)
                t.start()
                try:
                    return run_pipeline(
                        input_path, cache_dir, config,
                        progress_callback=_progress,
                        cancel_flag=cancel_flag,
                    )
                finally:
                    cancel_flag["done"] = True

            result = await loop.run_in_executor(None, _run)

            job.original_path = result.original_path
            job.processed_path = result.processed_path
            job.processed_48k_path = result.processed_48k_path
            job.stats = result.stats

            if result.cancelled:
                job.status = "cancelled"
                await _ws_manager.broadcast_global({
                    "type": "audiolab_error",
                    "job_id": job.id,
                    "message": "Cancelled by user",
                })
            else:
                job.status = "done"
                await _ws_manager.broadcast_global({
                    "type": "audiolab_done",
                    "job_id": job.id,
                    "original_url": f"/api/audiolab/preview/{job.id}?which=original",
                    "processed_url": f"/api/audiolab/preview/{job.id}?which=processed",
                    "stats": result.stats,
                })

        except Exception as exc:
            logger.error("AudioLab %s failed: %s", job.id, exc, exc_info=True)
            job.status = "error"
            job.message = str(exc)
            await _ws_manager.broadcast_global({
                "type": "audiolab_error",
                "job_id": job.id,
                "message": str(exc),
            })

    # Audiolab cache cleanup
    @app.on_event("startup")
    async def _cleanup_old_audiolab_cache():
        """Remove audiolab cache entries older than 24h."""
        import time as _time
        cache_root = Path(tempfile.gettempdir()) / "transcriber_audiolab"
        if cache_root.exists():
            cutoff = _time.time() - 86400
            for d in cache_root.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)

    @app.on_event("shutdown")
    async def _cleanup_audiolab_cache():
        """Remove all audiolab cache on shutdown."""
        cache_root = Path(tempfile.gettempdir()) / "transcriber_audiolab"
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)

    return app


async def _run_install(
    provider_name: str,
    provider: object,
    ws_manager,
) -> None:
    """Run provider.install_deps() in a thread pool and stream progress via WebSocket."""
    loop = asyncio.get_running_loop()

    def _progress(prog: float, msg: str) -> None:
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            ws_manager.broadcast_global({
                "type": "install_progress",
                "package": provider_name,
                "progress": prog,
                "message": msg,
            }),
        )

    try:
        await loop.run_in_executor(
            None, lambda: provider.install_deps(progress_callback=_progress)
        )
        await ws_manager.broadcast_global({
            "type": "install_done",
            "package": provider_name,
            "success": True,
            "error": None,
        })
    except Exception as exc:
        logger.error("Install %s failed: %s", provider_name, exc, exc_info=True)
        await ws_manager.broadcast_global({
            "type": "install_done",
            "package": provider_name,
            "success": False,
            "error": str(exc),
        })


async def _run_audiolab_install(
    tool_name: str,
    pkg: dict,
    ws_manager,
) -> None:
    """Install an Audio Lab dependency (demucs / deepfilter) via pip.

    Streams pip output line-by-line so the user sees real download progress
    instead of a frozen spinner.  Uses Popen (not check_call) to avoid the
    classic deadlock where an unread stderr pipe fills its 64 KB buffer.
    """
    import subprocess, sys
    loop = asyncio.get_running_loop()

    def _send(prog, msg):
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            ws_manager.broadcast_global({
                "type": "install_progress",
                "package": tool_name,
                "progress": prog,
                "message": msg,
            }),
        )

    def _do_install():
        _send(0.05, f"Installing {pkg['label']}...")

        # Ensure Rust/Cargo is in PATH if installed (rustup adds to system
        # PATH but already-running processes don't see it until restarted)
        import os
        env = os.environ.copy()
        cargo_bin = Path.home() / ".cargo" / "bin"
        if cargo_bin.is_dir() and str(cargo_bin) not in env.get("PATH", ""):
            env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")

        # Stream pip output line-by-line for real progress updates
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", pkg["pip"],
             "--progress-bar=off"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout to avoid deadlock
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )
        lines_seen = 0
        last_lines = []  # keep last few lines for error diagnosis
        for line in proc.stdout:
            lines_seen += 1
            line = line.strip()
            if not line:
                continue
            last_lines.append(line)
            if len(last_lines) > 10:
                last_lines.pop(0)
            # Show the user what pip is doing (download, install, etc.)
            # Progress goes from 0.05 → 0.85, proportional to output lines
            frac = min(0.05 + lines_seen * 0.02, 0.85)
            _send(frac, line[:120])  # truncate very long lines

        returncode = proc.wait()
        if returncode != 0:
            # Build a helpful error message from pip output
            all_output = "\n".join(last_lines).lower()
            # DeepFilterNet needs Rust — detect various failure signatures
            needs_rust = (
                "cargo" in all_output
                or "rust" in all_output
                or "deepfilterlib" in all_output
                or "maturin" in all_output
            )
            if needs_rust and tool_name == "deepfilter":
                raise RuntimeError(
                    "DeepFilterNet requires the Rust compiler to build.\n"
                    "1. Install Rust: https://rustup.rs\n"
                    "2. Restart the app\n"
                    "3. Try again"
                )
            output_tail = "\n".join(last_lines[-5:])
            raise RuntimeError(
                f"pip install failed (exit code {returncode}):\n{output_tail}"
            )

        _send(0.9, "Verifying import...")

        # Verify the package was installed (lightweight check)
        import importlib
        if tool_name == "demucs":
            importlib.import_module("demucs")
        elif tool_name == "deepfilter":
            # Import deepfilterlib (the native extension) rather than df.enhance
            # which pulls in torchaudio.backend (removed in newer torchaudio)
            importlib.import_module("deepfilterlib")

        _send(1.0, "Done")

    try:
        await loop.run_in_executor(None, _do_install)
        # Update the flags in preprocess module so they reflect new state
        import app.core.preprocess as _pp
        if tool_name == "demucs":
            _pp._DEMUCS_AVAILABLE = True
        elif tool_name == "deepfilter":
            _pp._DEEPFILTER_AVAILABLE = True

        await ws_manager.broadcast_global({
            "type": "install_done",
            "package": tool_name,
            "success": True,
            "error": None,
        })
    except Exception as exc:
        logger.error("Install audiolab tool %s failed: %s", tool_name, exc, exc_info=True)
        await ws_manager.broadcast_global({
            "type": "install_done",
            "package": tool_name,
            "success": False,
            "error": str(exc),
        })


def _path_within(path: Path, root: Path) -> bool:
    """Return True if path is inside root (prevents path traversal)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


async def _run_job(job: Job, settings: Settings, providers: dict, history: History, ws_manager) -> None:
    """Execute a transcription job in the background."""
    import time
    from app.core.audio import split_audio
    from app.core.output import format_transcript, merge_transcripts

    async def emit(msg: dict):
        await ws_manager.broadcast(job.id, msg)

    job.status = "running"
    logger.info("Job %s started: provider=%s model=%s files=%d",
                job.id, job.provider_name, job.model_id, len(job.input_files))
    await emit({"type": "progress", "job_id": job.id, "progress": 0, "message": "Starting..."})

    chunk_dir = Path("audio_chunks") / job.id
    try:
        provider = providers.get(job.provider_name)
        if not provider or not provider.is_available():
            raise RuntimeError(f"Provider '{job.provider_name}' is not available.")

        all_results = []
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for file_idx, input_file in enumerate(job.input_files):
            if job.status == "cancelled":
                await emit({"type": "error", "job_id": job.id, "message": "Cancelled", "retryable": False})
                return

            await emit({"type": "progress", "job_id": job.id,
                        "progress": file_idx / len(job.input_files),
                        "message": f"Splitting {input_file.name}..."})

            file_chunk_dir = chunk_dir / f"file_{file_idx}"
            chunks = split_audio(input_file, file_chunk_dir, job.opts.chunk_size_sec)

            if job.status == "cancelled":
                await emit({"type": "error", "job_id": job.id, "message": "Cancelled", "retryable": False})
                return

            await emit({"type": "progress", "job_id": job.id,
                        "progress": (file_idx + 0.1) / len(job.input_files),
                        "message": f"Transcribing {input_file.name} ({len(chunks)} chunks)…"})

            # Progress callback: maps provider chunk progress (0-1) into the
            # file-level range [file_idx+0.1 … file_idx+1) / num_files.
            n_files = len(job.input_files)
            async def _emit_chunk_progress(frac: float, msg: str,
                                           _fidx=file_idx, _nf=n_files):
                overall = (_fidx + 0.1 + frac * 0.9) / _nf
                await emit({"type": "progress", "job_id": job.id,
                            "progress": overall, "message": msg})

            def _sync_progress(frac: float, msg: str):
                """Thread-safe bridge: schedule the async emit on the event loop."""
                import asyncio as _aio
                loop = _aio.get_event_loop()
                loop.call_soon_threadsafe(
                    _aio.ensure_future, _emit_chunk_progress(frac, msg)
                )

            result = await provider.transcribe_batch(
                chunks, job.opts, progress_callback=_sync_progress
            )
            all_results.append((result, input_file.name))

        if job.status == "cancelled":
            await emit({"type": "error", "job_id": job.id, "message": "Cancelled", "retryable": False})
            return

        # Write output files
        output_files: list[Path] = []
        # Always use a known output directory (never the temp upload dir)
        # so the download endpoint's allow-list can serve the files.
        output_dir = settings.resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = job.input_files[0].stem if len(job.input_files) == 1 else "merged"

        if job.merge_output and len(all_results) > 1:
            for fmt in job.opts.output_formats:
                content = merge_transcripts(all_results, fmt)
                out_path = output_dir / f"{base_name}.{fmt}"
                out_path.write_text(content, encoding="utf-8")
                output_files.append(out_path)
        else:
            for result, filename in all_results:
                stem = Path(filename).stem
                for fmt in job.opts.output_formats:
                    content = format_transcript(result, fmt)
                    out_path = output_dir / f"{stem}.{fmt}"
                    out_path.write_text(content, encoding="utf-8")
                    output_files.append(out_path)

        job.status = "done"
        job.output_files = output_files

        # Compute total audio duration from the last segment end time across all results
        total_duration = 0.0
        for result, _ in all_results:
            if result.segments:
                total_duration += result.segments[-1].end

        history.save(job, duration_sec=total_duration)

        logger.info("Job %s done: %d output files", job.id, len(output_files))
        await emit({"type": "done", "job_id": job.id,
                    "output_files": [str(p) for p in output_files]})

    except Exception as e:
        logger.error("Job %s failed: %s", job.id, e, exc_info=True)
        job.status = "error"
        job.error_message = str(e)
        history.save(job, duration_sec=0.0)
        await emit({"type": "error", "job_id": job.id, "message": str(e), "retryable": False})

    finally:
        # Clean up audio chunks and uploaded temp files
        shutil.rmtree(str(chunk_dir), ignore_errors=True)
        if job.input_files:
            shutil.rmtree(str(job.input_files[0].parent), ignore_errors=True)
