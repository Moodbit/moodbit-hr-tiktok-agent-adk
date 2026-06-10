"""
Video generation job store + runner.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from google.cloud import firestore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoJobStore:
    def __init__(self, collection: Optional[str] = None) -> None:
        name = collection or os.getenv("ADK_VIDEO_JOBS_COLLECTION", "video_jobs")
        self._client = firestore.Client()
        self._collection = self._client.collection(name)

    def create_job(self, payload: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        doc = {
            "job_id": job_id,
            "status": "queued",
            "progress": [],
            "result": None,
            "error": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            **payload,
        }
        self._collection.document(job_id).set(doc)
        return job_id

    def update_job(self, job_id: str, fields: Dict[str, Any]) -> None:
        fields = {**fields, "updated_at": _utc_now()}
        self._collection.document(job_id).set(fields, merge=True)

    def append_progress(self, job_id: str, message: str) -> None:
        self._collection.document(job_id).update(
            {
                "progress": firestore.ArrayUnion([message]),
                "updated_at": _utc_now(),
            }
        )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        snap = self._collection.document(job_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        data["job_id"] = job_id
        return data


def run_in_background(target: Callable[[], None]) -> None:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
