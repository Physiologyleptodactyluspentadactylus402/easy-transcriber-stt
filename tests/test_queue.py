import pytest
from pathlib import Path
from app.core.queue import JobQueue
from app.providers.base import Job, TranscribeOptions


def make_job(**kwargs) -> Job:
    defaults = dict(
        input_files=[Path("test.mp3")],
        opts=TranscribeOptions(),
        provider_name="openai",
        model_id="whisper-1",
        status="pending",
    )
    defaults.update(kwargs)
    return Job(**defaults)


def test_job_has_uuid_id():
    job = make_job()
    assert len(job.id) == 36  # UUID4 format


def test_two_jobs_have_different_ids():
    assert make_job().id != make_job().id


def test_queue_add_and_get():
    q = JobQueue()
    job = make_job()
    q.add(job)
    assert q.get(job.id) is job


def test_queue_get_missing_returns_none():
    q = JobQueue()
    assert q.get("nonexistent") is None


def test_queue_list_returns_all():
    q = JobQueue()
    j1, j2 = make_job(), make_job()
    q.add(j1)
    q.add(j2)
    assert len(q.list()) == 2


def test_queue_update_status():
    q = JobQueue()
    job = make_job()
    q.add(job)
    q.update(job.id, status="running", progress=0.5)
    updated = q.get(job.id)
    assert updated.status == "running"
    assert updated.progress == 0.5


def test_queue_cancel():
    q = JobQueue()
    job = make_job()
    q.add(job)
    q.cancel(job.id)
    assert q.get(job.id).status == "cancelled"
