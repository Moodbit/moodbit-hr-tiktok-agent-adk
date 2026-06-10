"""
Content tools for HR TikTok pipeline.
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple

from app.controllers.hr_content_controller import (
    generate_hr_activity_scripts,
    generate_hr_news_scripts,
    generate_hr_phrase_scripts,
    generate_hr_tip_scripts,
)
from app.controllers.models import execute_llm

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


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        s = re.sub(r"^[a-zA-Z0-9_+-]+\n", "", s, count=1)
    return s.strip()


def _enforce_max_words(s: str, max_words: int) -> str:
    words = s.split()
    return " ".join(words[:max_words])


def generate_content(content_type: str) -> str:
    normalized = (content_type or "hr_tip").strip().lower()
    if normalized == "hr_news":
        content, _title, _link = generate_hr_news_scripts()
        return content
    if normalized == "hr_phrase":
        return generate_hr_phrase_scripts()
    if normalized == "hr_activity":
        return generate_hr_activity_scripts()
    return generate_hr_tip_scripts()


def build_scripts(raw_content: str, n_parts: int = 3, model: str = "gemini-flash-latest", max_words_per_part: int = 35) -> List[str]:
    fmt = random.choice(_FORMAT_STYLES)
    prompt = _NPARTS_PROMPT.format(
        persona=_PERSONA,
        content=raw_content,
        format_instruction=fmt["instruction"],
        n=n_parts,
        max_words=max_words_per_part,
    )
    raw = execute_llm(model, [{"role": "user", "content": prompt}], {})
    raw = _strip_code_fences(raw)

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
    return _humanize_scripts(scripts, model=model, max_words=max_words_per_part)


def _humanize_scripts(scripts: List[str], model: str = "gemini-flash-latest", max_words: int = 35) -> List[str]:
    numbered = "\n".join(f"{i + 1}) {s}" for i, s in enumerate(scripts))
    prompt = _HUMANIZE_PROMPT.format(
        persona=_PERSONA,
        lines=numbered,
        max_words=max_words,
    )
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


def generate_caption(raw_content: str, content_type: str, model: str = "gemini-flash-latest") -> str:
    prompt = _CAPTION_PROMPT.format(
        persona=_PERSONA,
        content=raw_content[:400],
        content_type=content_type,
    )
    caption = execute_llm(model, [{"role": "user", "content": prompt}], {})
    return _strip_code_fences(caption)[:2200]
