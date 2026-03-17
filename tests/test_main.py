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
