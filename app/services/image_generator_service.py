"""
services/image_generator_service.py
────────────────────────────────────
Generates AI persona images using Gemini.
"""

from __future__ import annotations

import mimetypes
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import google.genai as genai
from google.genai import types as genai_types
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GARequest

_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

IMA_DIR = "app/ima"
BASE_IMAGE_PATH = os.path.join(IMA_DIR, "Imagebase.png")
PERSONA_CACHE_PATH = os.path.join(IMA_DIR, "persona_cache.png")
ASPECT_RATIO = "1:1"

OUTFIT_CATALOG: Dict[str, str] = {
    "business_casual_blue": "Light blue Oxford shirt (no tie), neat dark trousers; no logos; professional.",
    "smart_navy_blazer": "Navy blazer over white shirt; no tie; business professional; matte.",
    "charcoal_sweater": "Charcoal fine-knit crewneck sweater over dark trousers; no logos; clean.",
    "light_grey_suit": "Light grey suit jacket over white shirt; no tie; modern professional.",
    "olive_shirt_khaki": "Olive button-down shirt (open collar), khaki trousers; business casual.",
}

BACKGROUND_CATALOG: Dict[str, str] = {
    "modern_office": "Bright modern open-plan office background, blurred; plants visible; professional.",
    "conference_room": "Clean conference room with a glass wall behind; neutral tones; professional.",
    "white_studio": "Neutral white/light grey studio background; subtle gradient; professional.",
    "home_office_bookshelf": "Home office with a neat bookshelf behind; warm lighting; cosy professional.",
    "coworking_space": "Busy coworking space background, blurred; warm tones; energetic.",
}

ACTION_POSTURE_CATALOG: Dict[str, Dict[str, str]] = {
    "speaking_directly": {
        "action": "Speaking directly to camera with a warm smile and open posture.",
        "gaze": "Direct eye contact with camera.",
    },
    "pointing_up": {
        "action": "Right hand pointing upward with index finger, making a key point.",
        "gaze": "Direct eye contact with camera.",
    },
    "arms_open": {
        "action": "Arms slightly open and relaxed, welcoming posture.",
        "gaze": "Direct eye contact with camera.",
    },
    "nodding": {
        "action": "Slight nod with a knowing smile, as if agreeing with the audience.",
        "gaze": "Direct eye contact with camera.",
    },
    "thinking_pose": {
        "action": "Hand lightly on chin, thoughtful expression but still engaging.",
        "gaze": "Slight tilt then back to camera.",
    },
}

IDENTITY_LOCK = """
IDENTITY (HARD CONSTRAINTS):
- Keep the exact same person: facial structure, beard/stubble, glasses (if any), skin tone, hair details.
- No beautification, no age smoothing, no face shape changes.
- Maintain eye contact with the camera unless a variant explicitly says otherwise.
- CRITICAL: Generate ONLY ONE single photograph of the person in a natural scene.
- DO NOT generate reference sheets, character sheets, or multiple views of the person.
- DO NOT show the person from multiple angles in the same image.
- This must be a single natural photograph, not a design reference.
"""

QUALITY_FRAMING = """
FRAMING & QUALITY:
- Close-up portrait: face and upper shoulders only (head-and-shoulders shot), eye-level, strictly FRONTAL angle.
- DO NOT show the full body. DO NOT show the torso below the shoulders.
- Photorealistic, high detail, warm professional colour grade. No cartoon/painterly/CGI.
- Single natural photograph of one person in a professional setting.
- Natural lighting, realistic shadows, professional photography.
"""


def _log(msg: str) -> None:
    print(f"[ImageGen] {msg}")


def _next_sequential_name(ima_dir: str) -> str:
    pat = re.compile(r"ima(\d{2})\.png$", re.IGNORECASE)
    max_idx = 0
    if os.path.isdir(ima_dir):
        for f in os.listdir(ima_dir):
            m = pat.match(f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return f"ima{max_idx + 1:02d}"


def _image_inline_part(path: str) -> Dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        raw = f.read()
    return {"_bytes": raw, "_mime": mime}


def _extract_inline_images(resp) -> List[bytes]:
    imgs: List[bytes] = []
    for cand in (resp.candidates or []):
        for part in (cand.content.parts or []):
            if part.inline_data and part.inline_data.data:
                imgs.append(part.inline_data.data)
    return imgs


def _maybe_first_text(resp) -> str:
    for cand in (resp.candidates or []):
        for part in (cand.content.parts or []):
            if part.text:
                return part.text
    return ""


class ImageGeneratorService:
    def __init__(
        self,
        *,
        sa_path: str = "service_a.json",
        location: str = "us-central1",
        model_id: str = "gemini-2.5-flash-image",
        overwrite_base: bool = True,
        random_seed: Optional[int] = None,
    ):
        if random_seed is not None:
            random.seed(random_seed)

        os.makedirs(IMA_DIR, exist_ok=True)
        if not os.path.exists(BASE_IMAGE_PATH):
            raise FileNotFoundError(
                f"Place your HR professional reference photo at: {BASE_IMAGE_PATH}"
            )

        self.model_id = model_id
        self.overwrite_base = overwrite_base

        if sa_path and os.path.exists(sa_path):
            project_id = os.getenv("GCP_PROJECT_ID", "")
            if not project_id:
                raise RuntimeError("GCP_PROJECT_ID required for Vertex AI authentication")
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=[_CLOUD_SCOPE]
            )
            creds.refresh(GARequest())
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=creds,
            )
            _log(f"Ready. model={self.model_id} (Vertex AI, project={project_id})")
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_TTS_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                _log(f"Ready. model={self.model_id} (Gemini API key)")
            else:
                raise RuntimeError(
                    "Image generation requires either service_a.json + GCP_PROJECT_ID or GEMINI_API_KEY"
                )

    def _save_image(self, img_bytes: bytes, numbered_name: str) -> str:
        path = os.path.join(IMA_DIR, f"{numbered_name}.png")
        with open(path, "wb") as f:
            f.write(img_bytes)
        return path

    def _call_and_save(self, contents: List[Dict[str, Any]], force_name: Optional[str] = None) -> str:
        sdk_parts = []
        for msg in contents:
            for part in msg.get("parts", []):
                if "_bytes" in part:
                    sdk_parts.append(genai_types.Part.from_bytes(data=part["_bytes"], mime_type=part["_mime"]))
                elif "text" in part:
                    sdk_parts.append(part["text"])

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=sdk_parts,
            config=genai_types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
        name = force_name or _next_sequential_name(IMA_DIR)

        imgs = _extract_inline_images(resp)
        if not imgs:
            snippet = _maybe_first_text(resp)
            _log("No IMAGE returned. Text snippet:")
            print(snippet[:350])
            raise RuntimeError(f"No image generated for {name}")

        saved = self._save_image(imgs[0], name)
        _log(f"Saved → {saved}")
        return saved

    def _generate_base(self) -> Tuple[str, str, str]:
        outfit_key = random.choice(list(OUTFIT_CATALOG.keys()))
        background_key = random.choice(list(BACKGROUND_CATALOG.keys()))
        outfit_desc = OUTFIT_CATALOG[outfit_key]
        background_desc = BACKGROUND_CATALOG[background_key]

        base_img = _image_inline_part(BASE_IMAGE_PATH)
        prompt_text = f"""
{IDENTITY_LOCK}
{QUALITY_FRAMING}

OUTFIT: {outfit_desc}
BACKGROUND: {background_desc}
POSE: Neutral, warm professional smile, direct eye contact with camera.

CRITICAL FRAMING RULES:
- Generate ONLY ONE single close-up portrait photograph (head and shoulders).
- Camera is at eye level, directly in front of the person. NO side angles. NO rotation.
- Do NOT show the full body. Cut the frame at the shoulders.
- This is a portrait photograph, NOT a reference sheet or character design.
- ONE person, ONE angle, ONE frame.

Aspect ratio: {ASPECT_RATIO}.
Return exactly one photorealistic close-up portrait of the person's face and shoulders.
""".strip()

        _log(f"Generating base image (outfit={outfit_key}, bg={background_key})...")
        contents = [{"parts": [base_img, {"text": prompt_text}]}]
        path = self._call_and_save(contents, force_name="ima00")
        shutil.copy2(path, PERSONA_CACHE_PATH)
        _log(f"Persona cache updated → {PERSONA_CACHE_PATH}")
        return path, outfit_key, background_key

    def _generate_complement(
        self,
        base_path: str,
        outfit_key: str,
        background_key: str,
        action_key: str,
    ) -> str:
        new_outfit_key = random.choice([k for k in OUTFIT_CATALOG if k != outfit_key] or list(OUTFIT_CATALOG.keys()))
        new_background_key = random.choice([k for k in BACKGROUND_CATALOG if k != background_key] or list(BACKGROUND_CATALOG.keys()))

        outfit_desc = OUTFIT_CATALOG[new_outfit_key]
        background_desc = BACKGROUND_CATALOG[new_background_key]
        action = ACTION_POSTURE_CATALOG[action_key]

        base_img = _image_inline_part(base_path)
        prompt_text = f"""
{IDENTITY_LOCK}
{QUALITY_FRAMING}

OUTFIT: {outfit_desc}
BACKGROUND: {background_desc}
ACTION: {action['action']}
GAZE: {action['gaze']}

CRITICAL FRAMING RULES:
- Generate ONLY ONE single close-up portrait photograph (head and shoulders).
- Camera is at eye level, directly in front of the person. NO side angles. NO rotation.
- Do NOT show the full body. Cut the frame at the shoulders.
- This is a portrait photograph, NOT a reference sheet or character design.
- ONE person, ONE angle, ONE frame.

Aspect ratio: {ASPECT_RATIO}.
Return exactly one photorealistic close-up portrait of the person's face and shoulders.
""".strip()

        _log(f"Generating complement (action={action_key}, outfit={new_outfit_key}, bg={new_background_key})...")
        contents = [{"parts": [base_img, {"text": prompt_text}]}]
        return self._call_and_save(contents)

    def ensure_images(self, count: int) -> List[str]:
        if os.getenv("SKIP_IMAGE_GEN", "0").strip() == "1":
            _log("SKIP_IMAGE_GEN=1 — using Imagebase.png for all clips.")
            return [BASE_IMAGE_PATH] * count

        existing = sorted([
            os.path.join(IMA_DIR, f)
            for f in os.listdir(IMA_DIR)
            if re.match(r"ima\d{2}\.png$", f, re.IGNORECASE)
        ])

        if len(existing) >= count and not self.overwrite_base:
            return existing[:count]

        if os.path.exists(PERSONA_CACHE_PATH):
            _log("Reusing persona_cache.png to keep character consistent.")
            shutil.copy2(PERSONA_CACHE_PATH, os.path.join(IMA_DIR, "ima00.png"))
            return [os.path.join(IMA_DIR, "ima00.png")] * count

        try:
            base_path, outfit_key, background_key = self._generate_base()
        except Exception as exc:
            if os.path.exists(PERSONA_CACHE_PATH):
                _log(f"Base image generation failed: {exc}. Falling back to persona_cache.png.")
                return [PERSONA_CACHE_PATH] * count
            _log(f"Base image generation failed: {exc}. Falling back to Imagebase.png.")
            return [BASE_IMAGE_PATH] * count

        paths = [base_path]
        action_keys = random.sample(list(ACTION_POSTURE_CATALOG.keys()), min(len(ACTION_POSTURE_CATALOG), count - 1))

        for i in range(count - 1):
            action_key = action_keys[i % len(action_keys)]
            try:
                p = self._generate_complement(base_path, outfit_key, background_key, action_key)
                paths.append(p)
            except Exception as exc:
                _log(f"Complement {i + 1} failed: {exc}. Reusing base.")
                paths.append(base_path)

        return paths[:count]

    def cleanup_generated_images(self, keep_only_base: bool = True, remove_json: bool = True):
        if not os.path.isdir(IMA_DIR):
            return
        for f in os.listdir(IMA_DIR):
            if keep_only_base and re.match(r"ima\d{2}\.png$", f, re.IGNORECASE):
                try:
                    os.remove(os.path.join(IMA_DIR, f))
                    _log(f"Removed {f}")
                except Exception as exc:
                    _log(f"Could not remove {f}: {exc}")
            if remove_json and f.endswith(".response.json"):
                try:
                    os.remove(os.path.join(IMA_DIR, f))
                except Exception:
                    pass
