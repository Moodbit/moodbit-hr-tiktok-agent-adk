"""
Publish tools for HR TikTok pipeline.
"""

from __future__ import annotations

import json
import os
import asyncio
from typing import Optional

import requests

from app.services.azure_blob_storage_service import AzureBlobStorageService
from app.services.tiktok_service import TikTokService


def upload_video(video_path: str) -> str:
    az_conn = os.getenv("AZ_TIKTOK_STORAGE_CONNECTION_STRING")
    blob_storage = AzureBlobStorageService(conn_string=az_conn, container="tiktok")
    return blob_storage.upload_blob_from_disk(video_path)


async def _is_url_ready(url: str, retries: int = 10) -> bool:
    for _ in range(retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(5)
    return False


def upload_video_and_wait(video_path: str) -> str:
    url = upload_video(video_path)
    ready = asyncio.run(_is_url_ready(url))
    if not ready:
        raise RuntimeError(f"TikTok video URL is not accessible: {url}")
    return url


def publish_to_tiktok(caption: str, video_url: str) -> str:
    tiktok_auth_json = os.getenv("TIKTOK_AUTH_JSON")
    service = TikTokService(json.loads(tiktok_auth_json or "{}"))
    return service.post_video_content(caption, video_url)
