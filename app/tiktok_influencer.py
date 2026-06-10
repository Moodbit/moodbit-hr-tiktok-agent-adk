"""
Standalone TikTok HR content pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
from typing import List

import requests

from app.controllers.hr_content_controller import (
    generate_hr_activity_scripts,
    generate_hr_news_scripts,
    generate_hr_phrase_scripts,
    generate_hr_tip_scripts,
)
from app.controllers.models import execute_llm
from app.services.azure_blob_storage_service import AzureBlobStorageService
from app.services.image_generator_service import ImageGeneratorService, BASE_IMAGE_PATH
from app.services.speech_generator_service import SpeechGeneratorService
from app.services.tiktok_service import TikTokService
from app.services.veo3_generator_service import VideoGeneratorService

TTS_PARTS = int(os.getenv("TTS_PARTS", "3"))
TTS_MAX_WORDS = int(os.getenv("TTS_MAX_WORDS", "35"))


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        s = re.sub(r"^[a-zA-Z0-9_+-]+\n", "", s, count=1)
    return s.strip()


def _enforce_max_words(s: str, max_words: int) -> str:
    words = s.split()
    return " ".join(words[:max_words])


_PERSONA = """
You are Alex, an HR professional in your late 30s with 12 years of real-world experience in talent
management, culture, and people ops. You speak like a real person — direct, a little opinionated,
and genuinely passionate about making workplaces better. You use everyday language: contractions,
short punchy sentences, the occasional "honestly" or "here's the thing". You never sound like a
corporate brochure or a formal report.
"""

_FORMAT_STYLES = [
    {
        "name": "monologue",
        "instruction": """
Write a ONE flowing personal monologue. Start with a relatable hook or observation, dive into the
insight, and end with a punchy takeaway. No numbered sections — just natural spoken flow.
Example openings: "You know what nobody tells you about...", "I see this mistake all the time...",
"Honestly, this changed how I think about..."
""",
    },
    {
        "name": "hot_take",
        "instruction": """
Start with a bold, slightly contrarian opinion, back it up with 1-2 real reasons, and close with a
challenge or question for the viewer. Confident and punchy.
Example openings: "Unpopular opinion:", "Hot take:", "Nobody wants to say this, but..."
""",
    },
    {
        "name": "quick_list",
        "instruction": """
Tease the number of items upfront ("Three things most HR teams get wrong about..."), deliver each
item as a quick punchy sentence, then close with a one-liner call to action. Brisk and energetic —
like you're talking to a friend, not presenting a slide deck.
""",
    },
]

_NPARTS_PROMPT = """{persona}

<CONTENT>
{content}
</CONTENT>

{format_instruction}

Write EXACTLY {n} short spoken lines in English (max {max_words} words EACH), one per line.
These will be read aloud in a TikTok video — no markdown, no emojis, no URLs.
Each line must flow naturally when spoken out loud.

STRICT FORMAT (numbered, one line each):
1) <line one, max {max_words} words>
2) <line two, max {max_words} words>
...
{n}) <line {n}, max {max_words} words>
"""


_HUMANIZE_PROMPT = """{persona}

Rewrite these spoken TikTok lines to sound like a real person, not AI-generated text.

Original lines:
{lines}

Rules:
- Use contractions (you're, it's, they've, don't, can't, here's)
- Add natural speech markers where they fit: "honestly", "here's the thing", "look", "real talk"
- Vary rhythm — mix short punchy sentences with longer flowing ones
- Keep the same core information and stay within {max_words} words per line
- No emojis, no markdown, no URLs
- Return ONLY the rewritten lines in the same numbered format:
1) <rewritten line>
2) <rewritten line>
...
"""


_CAPTION_PROMPT = """{persona}

Content: {content}
Content type: {content_type}

Write a TikTok caption for this post.
- First sentence: a short scroll-stopping hook (punchy, curiosity-driven)
- 2-3 sentences total, conversational tone
- End with 3-5 relevant hashtags
- Max 150 characters before the hashtags

Return ONLY the caption text + hashtags, nothing else.
"""


def _humanize_scripts(scripts: List[str], model: str = "gemini-flash-latest", max_words: int = 35) -> List[str]:
    numbered = "\n".join(f"{i + 1}) {s}" for i, s in enumerate(scripts))
    prompt = _HUMANIZE_PROMPT.format(
        persona=_PERSONA,
        lines=numbered,
        max_words=max_words,
    )
    try:
        raw = execute_llm(model, [{"role": "user", "content": prompt}], {})
        raw = _strip_code_fences(raw)
        out = {}
        for line in raw.splitlines():
            line = line.strip()
            m = re.match(r"^\s*(\d+)\)\s*(.*)$", line)
            if m:
                idx = int(m.group(1))
                text = m.group(2).strip(" :\t")
                if text:
                    out[idx] = text
        result = [out.get(i + 1, scripts[i]) for i in range(len(scripts))]
        return [_enforce_max_words(s.strip().strip('"'), max_words) for s in result]
    except Exception as exc:
        print(f"[Humanize] Warning: {exc} — returning original scripts")
        return scripts


def build_hr_scripts(content: str, n_parts: int = 3, model: str = "gemini-flash-latest", max_words_per_part: int = 35) -> List[str]:
    import random as _random

    fmt = _random.choice(_FORMAT_STYLES)
    print(f"[Scripts] Format style: {fmt['name']}")
    prompt = _NPARTS_PROMPT.format(
        persona=_PERSONA,
        content=content,
        format_instruction=fmt["instruction"],
        n=n_parts,
        max_words=max_words_per_part,
    )
    raw = execute_llm(model, [{"role": "user", "content": prompt}], {})
    raw = _strip_code_fences(raw)
    print("[Scripts] LLM response:\n", raw)

    numbered = {}
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"^\s*(\d+)\)\s*(.*)$", line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip(" :\t")
            if text:
                numbered[idx] = text

    scripts: List[str] = [numbered[i] for i in range(1, n_parts + 1) if i in numbered]

    if len(scripts) < n_parts:
        plain = [l.strip() for l in raw.splitlines() if l.strip() and not re.match(r"^\s*\d+\)", l)]
        for l in plain:
            if len(scripts) >= n_parts:
                break
            scripts.append(l)

    while len(scripts) < n_parts:
        if len(scripts) == 0:
            scripts.append("Hello everyone! Quick HR tip coming your way today.")
        elif len(scripts) == n_parts - 1:
            scripts.append("Thanks for watching — stay curious and keep growing!")
        else:
            scripts.append("Applying this in your team can make a real difference.")

    scripts = [_enforce_max_words(s.strip().strip('"'), max_words_per_part) for s in scripts[:n_parts]]
    scripts = _humanize_scripts(scripts, model=model, max_words=max_words_per_part)
    print("[Scripts] Final scripts:", scripts)
    return scripts


def _generate_caption(content: str, content_type: str, model: str = "gemini-flash-latest") -> str:
    prompt = _CAPTION_PROMPT.format(
        persona=_PERSONA,
        content=content[:400],
        content_type=content_type,
    )
    try:
        caption = execute_llm(model, [{"role": "user", "content": prompt}], {})
        caption = _strip_code_fences(caption)
        print("[Caption] Generated:", caption)
        return caption[:2200]
    except Exception as exc:
        print(f"[Caption] Warning: {exc} — using fallback caption")
        return content[:150]


class TikTokInfluencer:
    def __init__(self):
        tiktok_auth_json = os.getenv("TIKTOK_AUTH_JSON")
        self.tiktok_service = TikTokService(json.loads(tiktok_auth_json or "{}"))

        az_conn = os.getenv("AZ_TIKTOK_STORAGE_CONNECTION_STRING")
        self.blob_storage = AzureBlobStorageService(conn_string=az_conn, container="tiktok")

        gcp_project_id = os.getenv("GCP_PROJECT_ID", "")

        gemini_tts_api_key = os.getenv("GEMINI_TTS_API_KEY", "").strip().strip('"').strip("'")
        self.tts_service = SpeechGeneratorService(
            api_key=gemini_tts_api_key,
            project_id=gcp_project_id,
            sa_json="service_a.json",
            location="us-central1",
        )

        self.video_service = VideoGeneratorService(
            project_id=gcp_project_id,
            service_account_json="service_a.json",
            model_id="veo-3.1-fast-generate-preview",
            location="us-central1",
        )

        self.image_service = ImageGeneratorService(
            sa_path="service_a.json",
            location="us-central1",
            model_id="gemini-2.5-flash-image",
            overwrite_base=True,
        )

    def _generate_video_from_scripts(self, scripts: List[str], output_path: str) -> str:
        if scripts:
            scripts = list(scripts)
            scripts[0] = "Hey everyone! " + scripts[0]
            scripts[-1] = scripts[-1] + " Thanks for watching!"

        print("[Video] Ensuring images in 'app/ima/'...")
        base_images = self.image_service.ensure_images(1)
        anchor = None if os.path.abspath(base_images[0]) == os.path.abspath(BASE_IMAGE_PATH) else base_images[0]
        ensured = [anchor] * len(scripts)
        print("[Video] Ensured images:", ensured)

        final_path = self.video_service.generate_series_and_merge(
            scripts=scripts,
            images=ensured,
            output_dir="tmp_clips",
            merged_output_root=output_path,
            reuse_last_frame=False,
        )
        print("[Video] Final video:", final_path)
        return final_path

    async def _is_url_ready(self, url: str, retries: int = 10) -> bool:
        for attempt in range(retries):
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code == 200:
                    return True
            except Exception as exc:
                print(f"[URLCheck] Attempt {attempt + 1}: {exc}")
            await asyncio.sleep(5)
        return False

    def _upload_and_verify(self, video_path: str) -> str:
        url = self.blob_storage.upload_blob_from_disk(video_path)
        print("[Upload] Video URL:", url)
        is_ready = asyncio.run(self._is_url_ready(url))
        if not is_ready:
            raise RuntimeError(f"TikTok video URL is not accessible: {url}")
        return url

    def _cleanup_images(self):
        try:
            self.image_service.cleanup_generated_images(keep_only_base=True, remove_json=True)
        except Exception as exc:
            print(f"[Cleanup] Warning: {exc}")

    def _post_to_tiktok(self, caption: str, video_url: str):
        result = self.tiktok_service.post_video_content(caption, video_url)
        print("[TikTok] Post result:", result)
        return result

    def post_video_content(self):
        content_methods = [
            self.post_hr_news_content,
            self.post_hr_phrase_content,
            self.post_hr_activity_content,
            self.post_hr_tip_content,
        ]
        import random

        chosen = random.choice(content_methods)
        print(f"[TikTok] Selected content type: {chosen.__name__}")
        chosen()

    def post_hr_news_content(self):
        try:
            print("[HR News] Generating content...")
            content, _title, link = generate_hr_news_scripts()

            scripts = build_hr_scripts(
                content=content,
                n_parts=TTS_PARTS,
                max_words_per_part=TTS_MAX_WORDS,
            )

            video_path = self._generate_video_from_scripts(scripts, "./video_raw.mp4")
            video_url = self._upload_and_verify(video_path)
            self._cleanup_images()

            caption = _generate_caption(content, "hr_news") + f"\n{link}"
            self._post_to_tiktok(caption, video_url)
            print("[HR News] Done.")

        except Exception as exc:
            print("[HR News] ERROR:", str(exc))
            traceback.print_exc()

    def post_hr_phrase_content(self):
        try:
            print("[HR Phrase] Generating content...")
            content = generate_hr_phrase_scripts()

            scripts = build_hr_scripts(
                content=content,
                n_parts=TTS_PARTS,
                max_words_per_part=TTS_MAX_WORDS,
            )

            video_path = self._generate_video_from_scripts(scripts, "./video_raw.mp4")
            video_url = self._upload_and_verify(video_path)
            self._cleanup_images()

            caption = _generate_caption(content, "hr_phrase")
            self._post_to_tiktok(caption, video_url)
            print("[HR Phrase] Done.")

        except Exception as exc:
            print("[HR Phrase] ERROR:", str(exc))
            traceback.print_exc()

    def post_hr_activity_content(self):
        try:
            print("[HR Activity] Generating content...")
            content = generate_hr_activity_scripts()

            scripts = build_hr_scripts(
                content=content,
                n_parts=TTS_PARTS,
                max_words_per_part=TTS_MAX_WORDS,
            )

            video_path = self._generate_video_from_scripts(scripts, "./video_raw.mp4")
            video_url = self._upload_and_verify(video_path)
            self._cleanup_images()

            caption = _generate_caption(content, "hr_activity")
            self._post_to_tiktok(caption, video_url)
            print("[HR Activity] Done.")

        except Exception as exc:
            print("[HR Activity] ERROR:", str(exc))
            traceback.print_exc()

    def post_hr_tip_content(self):
        try:
            print("[HR Tip] Generating content...")
            content = generate_hr_tip_scripts()

            scripts = build_hr_scripts(
                content=content,
                n_parts=TTS_PARTS,
                max_words_per_part=TTS_MAX_WORDS,
            )

            video_path = self._generate_video_from_scripts(scripts, "./video_raw.mp4")
            video_url = self._upload_and_verify(video_path)
            self._cleanup_images()

            caption = _generate_caption(content, "hr_tip")
            self._post_to_tiktok(caption, video_url)
            print("[HR Tip] Done.")

        except Exception as exc:
            print("[HR Tip] ERROR:", str(exc))
            traceback.print_exc()
