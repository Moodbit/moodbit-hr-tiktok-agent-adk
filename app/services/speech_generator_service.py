"""
services/speech_generator_service.py
────────────────────────────────────
Generates WAV audio from text using Gemini TTS.
"""

from __future__ import annotations

import os
import wave

from google import genai
from google.genai import types


class SpeechGeneratorService:
    def __init__(
        self,
        api_key: str = "",
        default_voice: str = "kore",
        project_id: str = "",
        sa_json: str = "",
        location: str = "us-central1",
    ):
        self.default_voice = default_voice.lower()
        if project_id and sa_json:
            sa_abs = os.path.abspath(sa_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_abs
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            self._model = "gemini-2.5-flash-preview-tts"
        else:
            self.client = genai.Client(api_key=api_key)
            self._model = "gemini-2.5-flash-preview-tts"

    def generate_speech(
        self,
        text: str,
        style_instruction: str = "",
        output_path: str = "output.wav",
    ) -> str:
        content = f"{style_instruction}: {text}" if style_instruction else text

        response = self.client.models.generate_content(
            model=self._model,
            contents=content,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.default_voice
                        )
                    )
                ),
            ),
        )

        pcm_data = response.candidates[0].content.parts[0].inline_data.data
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)

        return output_path
