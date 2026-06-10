"""
controllers/models.py
──────────────────────
LLM utilities for the standalone ADK app.
Uses Gemini via google-genai (Vertex AI).
"""

from __future__ import annotations

from typing import Any, Dict, List

from google import genai
from google.genai import types


class GeminiLLM:
    def __init__(self, *, model: str = "gemini-flash-latest") -> None:
        self.model = model
        # Vertex AI auth is controlled via env vars in agent.py
        self.client = genai.Client(vertexai=True)

    def chat(self, messages: List[Dict[str, str]], options: Dict[str, Any] | None = None) -> str:
        options = options or {}
        parts: List[str] = []
        for msg in messages:
            text = msg.get("content", "")
            if text:
                parts.append(text)

        config = types.GenerateContentConfig(
            temperature=options.get("temperature", 0.7),
            top_p=options.get("top_p", 0.95),
            max_output_tokens=options.get("max_output_tokens", 1024),
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=config,
        )
        return response.text or ""


_default_llm = GeminiLLM()


def execute_llm(model: str, messages: list, options: dict) -> str:
    """
    Wrapper to match old signature. Returns text content.
    """
    llm = GeminiLLM(model=model)
    return llm.chat(messages, options)
