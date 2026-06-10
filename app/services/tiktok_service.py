"""
services/tiktok_service.py
──────────────────────────
TikTok Business API client.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import requests


class TikTokService:
    def __init__(self, auth_info: Dict) -> None:
        self.auth_info = auth_info
        self.pending_posts: Dict[str, dict] = {}

    def _refresh_token(self) -> str:
        headers = {"Content-Type": "application/json"}
        body = {
            "client_id": os.environ.get("TIKTOK_CLIENT_KEY"),
            "client_secret": os.environ.get("TIKTOK_CLIENT_SECRET"),
            "grant_type": "refresh_token",
            "refresh_token": self.auth_info["refresh_token"],
        }
        response = requests.post(
            "https://business-api.tiktok.com/open_api/v1.3/tt_user/oauth2/refresh_token/",
            headers=headers,
            json=body,
            timeout=60,
        )
        raw = response.json()
        if "data" not in raw:
            raise RuntimeError(f"[TikTok] Token refresh failed: {raw}")
        data = raw["data"]
        self.auth_info["access_token"] = data["access_token"]
        self.auth_info["access_token_expiration"] = (
            datetime.now() + timedelta(seconds=data["expires_in"])
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return data["access_token"]

    def post_video_content(self, caption: str, video_url: str) -> str:
        access_token = self.auth_info["access_token"]
        token_expiration = datetime.strptime(
            self.auth_info["access_token_expiration"], "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)
        if token_expiration < datetime.now(tz=timezone.utc):
            access_token = self._refresh_token()

        headers = {
            "Access-Token": access_token,
            "Content-Type": "application/json",
        }
        body = {
            "business_id": self.auth_info["business_id"],
            "video_url": video_url,
            "post_info": {
                "caption": caption,
                "disable_comment": False,
                "is_ai_generated": True,
                "is_brand_organic": True,
                "is_branded_content": False,
            },
        }

        resp = requests.post(
            "https://business-api.tiktok.com/open_api/v1.3/business/video/publish/",
            headers=headers,
            json=body,
            timeout=60,
        )

        try:
            response = resp.json()
        except Exception:
            resp.raise_for_status()
            raise

        if resp.status_code != 200:
            raise RuntimeError(f"TikTok HTTP {resp.status_code}: {response}")

        error = response.get("error") or {}
        if error and str(error.get("code", "")).lower() not in ("ok", "0", ""):
            log_id = error.get("log_id") or response.get("request_id")
            raise RuntimeError(
                f"TikTok API error: {error.get('code')} — {error.get('message')} (log_id={log_id})"
            )

        data = response.get("data") or {}
        share_id = data.get("share_id")
        if not share_id:
            raise RuntimeError(f"TikTok response missing share_id: {response}")

        self.pending_posts[share_id] = {}
        return "Post pending to verification"

    def check_pending_posts_status(self):
        # Placeholder for background poller if needed later.
        return "ok"
