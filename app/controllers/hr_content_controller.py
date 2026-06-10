"""
controllers/hr_content_controller.py
────────────────────────────────────
Generates raw HR content strings for each TikTok content type.
Returns plain text that the influencer pipeline converts into voice scripts.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .models import execute_llm

# ─── Bing Search helper ───────────────────────────────────────────────────────

BING_KEY = os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY", "")
BING_ENDPOINT = os.environ.get("BING_SEARCH_V7_ENDPOINT", "https://api.bing.microsoft.com") + "/v7.0/search"

HR_NEWS_SOURCES = (
    "site:shrm.org OR site:hbr.org OR site:forbes.com/hr "
    "OR site:hrexecutive.com OR site:hrdive.com OR site:ere.net"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    )
}


def _bing_search_hr_news(num_results: int = 5) -> list:
    """Search Bing for recent HR news articles. Returns list of {title, url, snippet}."""
    topic = random.choice([
        "HR trends workforce",
        "employee engagement 2025",
        "talent acquisition strategy",
        "workplace culture innovation",
        "HR technology future of work",
        "performance management best practices",
        "diversity inclusion workplace",
        "employee wellbeing mental health work",
        "remote hybrid work HR",
    ])
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    params = {
        "q": f"{HR_NEWS_SOURCES} {topic}",
        "textDecorations": True,
        "textFormat": "HTML",
        "count": num_results,
        "responseFilter": "webPages",
        "freshness": "Week",
    }
    response = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({
            "title": item.get("name", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


def _fetch_article_text(url: str) -> str:
    """Download and extract plain text from an article URL."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text() for p in paragraphs).strip()
    except Exception as exc:
        print(f"[Fetch] Error fetching {url}: {exc}")
        return ""


# ─── LLM prompts ─────────────────────────────────────────────────────────────

_NEWS_SUMMARY_PROMPT = """
You are an HR professional reacting to a news article — write a 150-word take on it like a real
person, not a formal analyst.

Article:
<article>
{article_text}
</article>

Share what you actually think about it: what it means for HR teams right now, what strikes you about
it, and why it matters in the real world of work. Use first-person voice ("what strikes me here",
"here's what this means for your team", "I think"). Be direct and a little opinionated.
Plain English only. No URLs, no jargon, no article metadata.
Return ONLY the reaction paragraph.
"""

_PHRASE_PROMPT = """
You are an HR professional sharing a quote that genuinely moved you. Pick a real, attributed quote
from a business leader, philosopher, or HR thinker about work, people, or leadership.

Introduce it briefly as if sharing with a friend — 1 personal sentence before the quote that
explains why it resonates with you. Then the quote and attribution.

Format:
[1 personal sentence]\n"[quote]" — [Name]

Under 80 words total. Reflective, warm, authentic tone. No hashtags, no URLs.
Return ONLY the intro sentence + quote + attribution, nothing else.
"""

_ACTIVITY_PROMPT = """
You are an energetic HR facilitator who genuinely loves team-building. Design a simple, creative
activity that actually works in real teams.

Give it a fun, catchy name (3-5 words). Then describe it in 80-100 words with enthusiasm —
what you do, why it works, and what the team walks away with. Include 2-3 actionable steps.
Works remote or in-person. Write like you're pitching it to a friend, excited about the idea —
not writing a manual. High energy, conversational.
No hashtags, no URLs.
Return ONLY the activity name and description.
"""

_TIP_PROMPT = """
You are a senior HR pro who gives blunt, practical advice — no fluff.
Share ONE specific, immediately actionable tip about: {area}

Be direct: tell them exactly what to do and why it works. Address the reader as "you".
Use first-person authority: "I've seen this work", "in my experience", "the teams I've worked with".
Include a "try this today" micro-action they can actually do in the next 24 hours.
80-100 words. No hashtags, no URLs, no markdown.
Return ONLY the tip text.
"""

_TIP_AREAS = [
    "onboarding new employees",
    "conducting effective 1-on-1 meetings",
    "giving constructive feedback",
    "reducing employee burnout",
    "improving company culture",
    "managing remote teams",
    "boosting employee recognition",
    "talent retention strategies",
    "improving the hiring process",
    "building psychological safety at work",
]


@retry(reraise=True, after=lambda err: print(err), wait=wait_random_exponential(min=10, max=40), stop=stop_after_attempt(4))
def _llm(prompt: str, model: str = "gemini-flash-latest") -> str:
    return execute_llm(model, [{"role": "user", "content": prompt}], {})


# ─── Public API ───────────────────────────────────────────────────────────────


def generate_hr_news_scripts() -> Tuple[str, str, str]:
    """
    Fetch a recent HR news article and summarise it.
    Returns: (summary_text, article_title, article_url)
    """
    articles = _bing_search_hr_news(num_results=7)
    chosen = random.choice(articles)
    title = chosen["title"]
    url = chosen["url"]

    print(f"[HR News] Article: {title}")
    print(f"[HR News] URL: {url}")

    full_text = _fetch_article_text(url)
    if not full_text:
        full_text = chosen["snippet"]

    summary = _llm(_NEWS_SUMMARY_PROMPT.format(article_text=full_text[:4000]))
    return summary, title, url


def generate_hr_phrase_scripts() -> str:
    """Generate a motivational HR phrase/quote. Returns plain text."""
    return _llm(_PHRASE_PROMPT)


def generate_hr_activity_scripts() -> str:
    """Generate a team-building activity description. Returns plain text."""
    return _llm(_ACTIVITY_PROMPT)


def generate_hr_tip_scripts() -> str:
    """Generate a practical HR management tip. Returns plain text."""
    area = random.choice(_TIP_AREAS)
    return _llm(_TIP_PROMPT.format(area=area))
