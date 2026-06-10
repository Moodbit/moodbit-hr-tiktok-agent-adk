"""
Media tools for HR TikTok pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from app.services.image_generator_service import ImageGeneratorService, BASE_IMAGE_PATH
from app.services.veo3_generator_service import VideoGeneratorService
from app.services.video_job_service import VideoJobStore, run_in_background


def ensure_images(count: int) -> List[str]:
    image_service = ImageGeneratorService(
        sa_path="service_a.json",
        location="us-central1",
        model_id="gemini-2.5-flash-image",
        overwrite_base=True,
    )
    return image_service.ensure_images(count)


def generate_video(script_parts: List[str], image_paths: List[str]) -> str:
    project_id = os.getenv("GCP_PROJECT_ID", "")
    video_service = VideoGeneratorService(
        project_id=project_id,
        service_account_json="service_a.json",
        model_id="veo-3.1-fast-generate-001",
        location="us-central1",
    )
    anchor = None
    if image_paths:
        anchor = None if os.path.abspath(image_paths[0]) == os.path.abspath(BASE_IMAGE_PATH) else image_paths[0]
    images = [anchor] * len(script_parts)
    return video_service.generate_series_and_merge(
        scripts=script_parts,
        images=images,
        output_dir="tmp_clips",
        merged_output_root="./video_raw.mp4",
        reuse_last_frame=False,
    )


def start_video_job(script_parts: List[str], image_paths: List[str]) -> Dict[str, Any]:
    """Start a background video generation job and return job metadata."""
    job_store = VideoJobStore()
    payload = {
        "script_parts": script_parts,
        "image_paths": image_paths,
    }
    job_id = job_store.create_job(payload)

    def _run_job() -> None:
        job_store.update_job(job_id, {"status": "running"})
        try:
            project_id = os.getenv("GCP_PROJECT_ID", "")
            video_service = VideoGeneratorService(
                project_id=project_id,
                service_account_json="service_a.json",
                model_id="veo-3.1-fast-generate-001",
                location="us-central1",
            )
            anchor = None
            if image_paths:
                anchor = None if os.path.abspath(image_paths[0]) == os.path.abspath(BASE_IMAGE_PATH) else image_paths[0]
            images = [anchor] * len(script_parts)

            def _progress(msg: str) -> None:
                job_store.append_progress(job_id, msg)

            output_path = video_service.generate_series_and_merge(
                scripts=script_parts,
                images=images,
                output_dir="tmp_clips",
                merged_output_root="./video_raw.mp4",
                reuse_last_frame=False,
                progress_callback=_progress,
            )
            job_store.update_job(
                job_id,
                {
                    "status": "completed",
                    "result": {"video_path": output_path},
                },
            )
        except Exception as exc:
            job_store.update_job(job_id, {"status": "failed", "error": str(exc)})

    run_in_background(_run_job)
    return {"status": "pending", "job_id": job_id}


def get_video_job_status(job_id: str) -> Dict[str, Any]:
    """Check background video generation job status."""
    job_store = VideoJobStore()
    job = job_store.get_job(job_id)
    if not job:
        return {"status": "not_found", "job_id": job_id}

    response = {
        "status": job.get("status"),
        "job_id": job_id,
        "progress": job.get("progress", []),
        "error": job.get("error"),
    }
    if job.get("result"):
        response["result"] = job["result"]
    return response
