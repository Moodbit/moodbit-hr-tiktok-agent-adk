"""
Media tools for HR TikTok pipeline.
"""

from __future__ import annotations

import os
from typing import List

from app.services.image_generator_service import ImageGeneratorService, BASE_IMAGE_PATH
from app.services.veo3_generator_service import VideoGeneratorService


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
