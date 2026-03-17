import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(settings_path=tmp_path / "settings.json", db_path=tmp_path / "test.db")
    return TestClient(app)


def test_root_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_locale_endpoint_english(client):
    r = client.get("/api/locale?lang=en")
    assert r.status_code == 200
    data = r.json()
    assert data["nav_transcribe"] == "Transcribe"


def test_locale_endpoint_italian(client):
    r = client.get("/api/locale?lang=it")
    assert r.status_code == 200
    assert r.json()["nav_transcribe"] == "Trascrizione"


def test_providers_endpoint(client):
    r = client.get("/api/providers")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert "openai" in names
    assert "elevenlabs" in names


def test_settings_get(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert "language" in data
    assert "default_output_formats" in data


def test_settings_update(client):
    r = client.patch("/api/settings", json={"language": "it"})
    assert r.status_code == 200
    r2 = client.get("/api/settings")
    assert r2.json()["language"] == "it"


def test_history_empty_on_fresh_db(client):
    r = client.get("/api/history")
    assert r.status_code == 200
    assert r.json() == []


def test_upload_and_queue_job(client, tmp_path):
    dummy_audio = tmp_path / "test.mp3"
    dummy_audio.write_bytes(b"fake")
    with open(dummy_audio, "rb") as f:
        r = client.post(
            "/api/jobs",
            files={"files": ("test.mp3", f, "audio/mpeg")},
            data={
                "provider_name": "openai",
                "model_id": "whisper-1",
                "output_formats": '["txt"]',
                "merge_output": "false",
            },
        )
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_providers_endpoint_includes_local(client):
    r = client.get("/api/providers")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()]
    assert "faster_whisper" in names
    assert "qwen3_asr" in names
    assert "ollama" in names


def test_install_unknown_provider_returns_404(client):
    r = client.post("/api/install/nonexistent")
    assert r.status_code == 404


def test_install_known_provider_returns_200(client):
    r = client.post("/api/install/faster_whisper")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_ws_global_connection_tracked(tmp_path):
    """Every WebSocket connection is registered globally (for broadcast_global)."""
    from app.main import create_app
    app = create_app(settings_path=tmp_path / "s.json", db_path=tmp_path / "db.db")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    # Open a WebSocket without subscribing to any job — must not error
    with client.websocket_connect("/ws") as ws:
        # Send a non-subscribe message — server should handle without crashing
        ws.send_json({"type": "ping"})
        # No response expected, just verify no exception and connection stays open


def test_ffmpeg_endpoint_returns_bool(client):
    r = client.get("/api/ffmpeg")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data
    assert isinstance(data["available"], bool)


def test_websocket_start_live_responds_with_session_started(tmp_path):
    from app.main import create_app
    app = create_app(settings_path=tmp_path / "s.json", db_path=tmp_path / "db.db")
    from fastapi.testclient import TestClient
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "start_live",
            "provider_name": "faster_whisper",
            "model_id": "tiny",
            "opts": {"language": None, "speaker_labels": False, "output_formats": ["txt"]},
        })
        msg = ws.receive_json()
        assert msg["type"] == "live_session_started"
        assert "session_id" in msg


def test_websocket_stop_live_responds_with_session_stopped(tmp_path):
    from app.main import create_app
    app = create_app(settings_path=tmp_path / "s.json", db_path=tmp_path / "db.db")
    from fastapi.testclient import TestClient
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({
            "type": "start_live",
            "provider_name": "faster_whisper",
            "model_id": "tiny",
            "opts": {},
        })
        start_msg = ws.receive_json()
        session_id = start_msg["session_id"]

        ws.send_json({"type": "stop_live", "session_id": session_id})
        stop_msg = ws.receive_json()
        assert stop_msg["type"] == "live_session_stopped"
        assert stop_msg["session_id"] == session_id
