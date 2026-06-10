"""
services/veo3_generator_service.py
──────────────────────────────────
Google Veo 3 video generation via Vertex AI REST API.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple

import requests
import google.auth
from google.auth.transport.requests import Request as GARequest
from google.oauth2 import service_account

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_OUTFIT_CATALOG = {
    "business_casual_blue": "light blue Oxford shirt, neat dark trousers",
    "smart_navy_blazer": "navy blazer over white shirt",
    "charcoal_sweater": "charcoal fine-knit crewneck sweater over dark trousers",
    "light_grey_suit": "light grey suit jacket over white shirt",
    "olive_shirt_khaki": "olive button-down shirt, khaki trousers",
}

_BACKGROUND_CATALOG = {
    "modern_office": "bright modern open-plan office background, blurred",
    "conference_room": "clean conference room with a glass wall, neutral tones",
    "white_studio": "neutral white studio background, subtle gradient",
    "home_office_bookshelf": "home office with a neat bookshelf, warm lighting",
    "coworking_space": "busy coworking space background, blurred, warm tones",
}


def _get_bearer(sa_json_path: Optional[str]) -> str:
    if sa_json_path and os.path.exists(sa_json_path):
        creds = service_account.Credentials.from_service_account_file(
            sa_json_path, scopes=[CLOUD_SCOPE]
        )
    else:
        creds, _ = google.auth.default(scopes=[CLOUD_SCOPE])
    creds.refresh(GARequest())
    if not creds.token:
        raise RuntimeError("[Auth] Could not obtain OAuth2 token.")
    return creds.token


def _find_image_path(path_or_base: str) -> str:
    for p in [path_or_base, f"{path_or_base}.png", f"{path_or_base}.jpg", f"{path_or_base}.jpeg"]:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"[Image] Not found: {path_or_base}")


def _sanitize_hr_text(s: str) -> str:
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\b(\d{1,3}\.){3}\d{1,3}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 500:
        s = s[:500]
    return s


_VIDEO_PERSONA_PROMPT = """\
[Cinematography] Medium close-up portrait, framing from mid-chest upward, subject centered, static camera, eye-level, 9:16 portrait, shallow depth of field, warm ring-light.
[Subject] A warm, professional HR specialist in their 30s-40s — use the provided reference image EXACTLY for face, hair, and glasses. Outfit: {outfit}. Keep this outfit UNCHANGED for the entire clip.
[Action] The character is actively speaking to camera from the very first frame. Natural head movement, genuine gestures, direct eye contact. Confident, conversational, friendly tone.
[Context] {background}. Professional indoor atmosphere.
[Style] Cinematic, photorealistic, static camera — no movement, no zoom, no pan or drift throughout the entire clip.
The character says, "{script}"
SFX: Clear natural room tone. No music. No captions or text overlays."""

_VIDEO_CONTINUATION_PROMPT = """\
SEAMLESS CONTINUATION — this clip continues the same shot. The provided start-frame image is the SINGLE SOURCE OF TRUTH for all visual details.
[Cinematography] IDENTICAL framing to the start frame: mid-chest upward, subject centered, static camera, eye-level, same focal length, same depth of field, same lighting. Do NOT zoom or reframe.
[Subject] The EXACT same person as in the start frame: same face, same hair, same glasses. CRITICAL: wearing {outfit} — do NOT change the outfit, color, or any clothing detail at any point during this clip. The outfit must remain identical from first to last frame.
[Action] Already speaking mid-conversation from frame zero. Smooth continuation of gesture and speech. No pause, no re-entrance, no transition. Direct eye contact.
[Context] IDENTICAL background to the start frame: {background}. Same lighting, same color grade, same depth of field. Do NOT change the background or environment at any point.
[Style] Cinematic, photorealistic, static camera — no movement, no zoom, no pull-back, no drift throughout the entire clip.
The character says, "{script}"
SFX: Clear natural room tone, continuous. No music. No captions."""


class VideoGeneratorService:
    def __init__(
        self,
        project_id: str,
        service_account_json: Optional[str] = "service_a.json",
        model_id: str = "veo-3.0-fast-generate-preview",
        location: str = "us-central1",
        duration_seconds: int = 8,
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
        generate_audio: bool = True,
        person_generation: str = "allow_all",
    ):
        self.project_id = project_id
        self.sa_json = service_account_json
        self.model_id = model_id
        self.location = location
        self.duration_seconds = duration_seconds
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
        self.generate_audio = generate_audio
        self.person_generation = person_generation

        self._base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/"
            f"publishers/google/models/{model_id}"
        )

    def _headers(self) -> Dict[str, str]:
        token = _get_bearer(self.sa_json)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), mime

    def _submit_job(
        self,
        prompt_text: str,
        image_path: Optional[str] = None,
        reference_image_paths: Optional[List[str]] = None,
        outfit: Optional[str] = None,
        background: Optional[str] = None,
        is_continuation: bool = False,
    ) -> str:
        _bg = background or random.choice(list(_BACKGROUND_CATALOG.values()))
        _outfit = outfit or random.choice(list(_OUTFIT_CATALOG.values()))
        template = _VIDEO_CONTINUATION_PROMPT if is_continuation else _VIDEO_PERSONA_PROMPT
        final_prompt = template.format(
            script=_sanitize_hr_text(prompt_text), outfit=_outfit, background=_bg
        )

        negative_prompt = (
            "freeze frame, fade in from still image, dissolve transition, wipe transition, "
            "static pose at start, camera zoom, camera push in, camera pull back, "
            "camera pan, camera drift, rotation sequence, multiple angles, reference sheet, "
            "clothing change, outfit change, costume change, wardrobe change, different clothes, "
            "different outfit, shirt change, color change on clothing"
        )

        payload: Dict[str, Any] = {
            "instances": [{"prompt": final_prompt}],
            "parameters": {
                "durationSeconds": self.duration_seconds,
                "aspectRatio": self.aspect_ratio,
                "resolution": self.resolution,
                "generateAudio": self.generate_audio,
                "personGeneration": self.person_generation,
                "negativePrompt": negative_prompt,
            },
        }

        if image_path:
            real_path = _find_image_path(image_path)
            b64, mime = self._encode_image(real_path)
            payload["instances"][0]["image"] = {
                "bytesBase64Encoded": b64,
                "mimeType": mime,
            }

        if reference_image_paths:
            ref_imgs = []
            for rp in reference_image_paths:
                real_rp = _find_image_path(rp)
                b64r, mimer = self._encode_image(real_rp)
                ref_imgs.append({
                    "image": {"bytesBase64Encoded": b64r, "mimeType": mimer},
                    "referenceType": "asset",
                })
            payload["instances"][0]["referenceImages"] = ref_imgs

        url = f"{self._base_url}:predictLongRunning"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)

        if not resp.ok:
            raise RuntimeError(f"[Veo3] Submit failed {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        op_name = data.get("name") or data.get("operationName")
        if not op_name:
            raise RuntimeError(f"[Veo3] No operation name in response: {data}")
        return op_name

    def _poll_job(self, op_name: str, timeout_secs: int = 600, poll_interval: int = 10) -> List[bytes]:
        deadline = time.time() + timeout_secs
        fetch_url = f"{self._base_url}:fetchPredictOperation"
        last_heartbeat = 0.0

        while time.time() < deadline:
            resp = requests.post(
                fetch_url,
                headers=self._headers(),
                json={"operationName": op_name},
                timeout=30,
            )
            if not resp.ok:
                raise RuntimeError(f"[Veo3] Poll failed {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            if not data.get("done"):
                now = time.time()
                if now - last_heartbeat >= 5:
                    print(f"[Veo3] Job in progress... ({op_name[-40:]})")
                    sys.stdout.flush()
                    last_heartbeat = now
                time.sleep(poll_interval)
                continue

            if "error" in data:
                raise RuntimeError(f"[Veo3] Job error: {data['error']}")

            videos = []
            response = data.get("response", {})
            for vid in response.get("videos", []):
                b64 = vid.get("bytesBase64Encoded")
                if b64:
                    videos.append(base64.b64decode(b64))
            if not videos:
                for pred in response.get("predictions", []):
                    b64 = pred.get("bytesBase64Encoded") or pred.get("video", {}).get("bytesBase64Encoded")
                    if b64:
                        videos.append(base64.b64decode(b64))
            if not videos:
                raise RuntimeError(f"[Veo3] No video bytes in response: {list(response.keys())}")
            return videos

        raise TimeoutError(f"[Veo3] Timed out after {timeout_secs}s for {op_name}")

    def _get_duration(self, clip_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            clip_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 8.0
        try:
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    return float(stream.get("duration", 8))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return 8.0

    def _extract_last_frame(self, clip_path: str, output_path: str) -> str:
        cmd = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.5",
            "-i",
            clip_path,
            "-vframes",
            "1",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"[ffmpeg] Last-frame extraction failed: {result.stderr[:200]}")
        return output_path

    def _generate_clip(
        self,
        script: str,
        image_path: Optional[str],
        output_path: str,
        max_retries: int = 3,
        outfit: Optional[str] = None,
        background: Optional[str] = None,
        is_continuation: bool = False,
        reference_image_paths: Optional[List[str]] = None,
    ) -> str:
        safe_script = _sanitize_hr_text(script)
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, max_retries + 1):
            try:
                op_name = self._submit_job(
                    safe_script,
                    image_path,
                    reference_image_paths=reference_image_paths,
                    outfit=outfit,
                    background=background,
                    is_continuation=is_continuation,
                )
                print(f"[Veo3] Job submitted (attempt {attempt}): {op_name[-40:]}")
                videos = self._poll_job(op_name)
                with open(output_path, "wb") as f:
                    f.write(videos[0])
                print(f"[Veo3] Clip saved: {output_path}")
                return output_path
            except RuntimeError as exc:
                last_exc = exc
                print(f"[Veo3] Attempt {attempt} failed: {exc}. {'Retrying...' if attempt < max_retries else 'Giving up.'}")
                time.sleep(5 * attempt)
        raise last_exc

    def generate_series_and_merge(
        self,
        scripts: List[str],
        images: List[str],
        output_dir: str = "tmp_clips",
        merged_output_root: str = "video_raw",
        reuse_last_frame: bool = False,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        clip_paths: List[str] = []
        video_outfit = random.choice(list(_OUTFIT_CATALOG.values()))
        video_background = random.choice(list(_BACKGROUND_CATALOG.values()))
        print(f"[Veo3] Series style — outfit: '{video_outfit[:40]}...', bg: '{video_background[:40]}...'")

        persona_cache = images[0] if images else None
        reference_images = [persona_cache] if persona_cache else None
        anchor_image: Optional[str] = None

        for i, (script, _img) in enumerate(zip_longest(scripts, images, fillvalue=None)):
            if script is None:
                break
            clip_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")

            if i == 0:
                print("[Veo3] Clip 0 — referenceImages mode (no freeze, outfit from text+reference).")
                self._generate_clip(
                    script,
                    image_path=None,
                    output_path=clip_path,
                    outfit=video_outfit,
                    background=video_background,
                    is_continuation=False,
                    reference_image_paths=reference_images,
                )
            else:
                print(f"[Veo3] Clip {i} — last-frame start (outfit/bg/framing continuity from previous clip).")
                self._generate_clip(
                    script,
                    image_path=anchor_image,
                    output_path=clip_path,
                    outfit=video_outfit,
                    background=video_background,
                    is_continuation=True,
                    reference_image_paths=None,
                )
            clip_paths.append(clip_path)

            if i < len(scripts) - 1:
                last_frame_path = os.path.join(output_dir, f"last_frame_{i:03d}.png")
                try:
                    anchor_image = self._extract_last_frame(clip_path, last_frame_path)
                    print(f"[Veo3] Last frame extracted → {last_frame_path}")
                except Exception as exc:
                    print(f"[Veo3] Warning: could not extract last frame: {exc}. Next clip uses no start frame.")
                    anchor_image = None

        if not clip_paths:
            raise RuntimeError("[Veo3] No clips were generated.")

        if len(clip_paths) == 1:
            merged = merged_output_root if merged_output_root.endswith(".mp4") else merged_output_root + ".mp4"
            shutil.copy(clip_paths[0], merged)
            print(f"[Veo3] Single clip copied to: {merged}")
            return merged

        merged = merged_output_root if merged_output_root.endswith(".mp4") else merged_output_root + ".mp4"
        n = len(clip_paths)

        durations = [self._get_duration(p) for p in clip_paths]
        trim_tail = 0.5

        input_args: List[str] = []
        for p in clip_paths:
            input_args += ["-i", p]

        filter_parts: List[str] = []

        for i in range(n):
            dur = durations[i]
            end_t = max(1.0, dur - trim_tail)
            filter_parts.append(f"[{i}:v]trim=end={end_t:.3f},setpts=PTS-STARTPTS[v{i}t]")

        for i in range(n):
            dur = durations[i]
            end_t = max(1.0, dur - trim_tail)
            filter_parts.append(
                f"[{i}:a]atrim=end={end_t:.3f},asetpts=PTS-STARTPTS,"
                f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[a{i}t]"
            )

        v_inputs = "".join([f"[v{i}t]" for i in range(n)])
        a_inputs = "".join([f"[a{i}t]" for i in range(n)])
        filter_parts.append(f"{v_inputs}concat=n={n}:v=1:a=0[vout]")
        filter_parts.append(f"{a_inputs}concat=n={n}:v=0:a=1[aout]")

        filter_complex = ";".join(filter_parts)
        cmd = [
            "ffmpeg",
            "-y",
            *input_args,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            merged,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"[ffmpeg] Merge failed: {result.stderr[:300]}")

        print(f"[Veo3] Merged video saved: {merged}")
        return merged
