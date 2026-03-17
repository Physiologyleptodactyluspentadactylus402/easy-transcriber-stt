import json
import pytest
import time
from pathlib import Path
from app.core.history import History
from app.providers.base import Job, TranscribeOptions


@pytest.fixture
def db(tmp_path):
    return History(tmp_path / "test.db")


def make_done_job(input_files=None, output_files=None):
    return Job(
        input_files=input_files or [Path("lecture.mp3")],
        opts=TranscribeOptions(),
        provider_name="openai",
        model_id="whisper-1",
        status="done",
        output_files=output_files or [Path("/out/lecture.srt")],
    )


def test_save_and_retrieve(db):
    job = make_done_job()
    db.save(job, duration_sec=120.0)
    sessions = db.list()
    assert len(sessions) == 1
    assert sessions[0]["id"] == job.id


def test_list_ordered_by_date_descending(db):
    j1 = make_done_job()
    time.sleep(0.01)
    j2 = make_done_job()
    db.save(j1, duration_sec=60.0)
    db.save(j2, duration_sec=90.0)
    sessions = db.list()
    assert sessions[0]["id"] == j2.id  # most recent first


def test_input_filenames_stored_as_json(db):
    job = make_done_job(input_files=[Path("a.mp3"), Path("b.mp3")])
    db.save(job, duration_sec=0.0)
    session = db.list()[0]
    assert session["input_filenames"] == ["a.mp3", "b.mp3"]


def test_output_paths_stored_as_json(db):
    job = make_done_job(output_files=[Path("/out/a.srt"), Path("/out/a.txt")])
    db.save(job, duration_sec=0.0)
    session = db.list()[0]
    assert session["output_paths"] == ["/out/a.srt", "/out/a.txt"]


def test_delete_removes_entry(db):
    job = make_done_job()
    db.save(job, duration_sec=0.0)
    db.delete(job.id)
    assert db.list() == []
