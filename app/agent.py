# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from pathlib import Path

import google.auth
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.genai import types

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if SRC_ROOT.exists():
    sys.path.append(str(SRC_ROOT))

from src.tools.content_tools import build_scripts, generate_caption, generate_content
from src.tools.media_tools import ensure_images, generate_video
from src.tools.publish_tools import publish_to_tiktok, upload_video

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

def generate_content_tool(content_type: str) -> str:
    """Generate raw HR content text for the requested content type. The accepted types are hr_tip, hr_phrase, hr_activity."""
    return generate_content(content_type)


def build_scripts_tool(raw_content: str, n_parts: int = 3) -> list[str]:
    """Split raw content into short spoken script parts."""
    return build_scripts(raw_content, n_parts=n_parts)


def generate_caption_tool(raw_content: str, content_type: str) -> str:
    """Generate a TikTok caption for the content."""
    return generate_caption(raw_content, content_type)


def ensure_images_tool(count: int) -> list[str]:
    """Ensure persona images exist and return their paths."""
    return ensure_images(count)


def generate_video_tool(script_parts: list[str], image_paths: list[str]) -> str:
    """Generate and merge video clips into a final MP4."""
    return generate_video(script_parts, image_paths)


def upload_video_tool(video_path: str) -> str:
    """Upload a local MP4 to Azure Blob and return the URL."""
    return upload_video(video_path)


def publish_to_tiktok_tool(caption: str, video_url: str) -> str:
    """Publish the video URL to TikTok and return the share id."""
    return publish_to_tiktok(caption, video_url)


content_agent = LlmAgent(
    name="hr_content_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You generate HR content, scripts, and captions. "
        "Call generate_content_tool, then build_scripts_tool, then generate_caption_tool. "
        "Return the result you got including raw_content, script_parts, caption and ask the user for their approval."
    ),
    tools=[
        FunctionTool(func=generate_content_tool),
        FunctionTool(func=build_scripts_tool),
        FunctionTool(func=generate_caption_tool),
    ],
)

media_agent = LlmAgent(
    name="media_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You generate images and the final video and save it in storage. "
        "Call ensure_images_tool, then generate_video_tool, then upload_video_tool. "
        # "Return a JSON object with keys: image_paths, video_path."
        "Return the result you got including the blob_video_url as hyperlink and ask for the user approval. "
    ),
    tools=[
        FunctionTool(func=ensure_images_tool),
        FunctionTool(func=generate_video_tool),
        FunctionTool(func=upload_video_tool),

    ],
)

publish_agent = LlmAgent(
    name="publish_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You publish the video to TikTok. "
        "Call publish_to_tiktok_tool. "
        "Return the result you got including tiktok_share_id. "
        "If post is pending, mention that it usually takes a few minutes to process and user will be able to see it on their TikTok soon."

    ),
    tools=[
        FunctionTool(func=publish_to_tiktok_tool),
    ],
)

root_agent = LlmAgent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an HR TikTok pipeline coordinator. Your goal is to run the full pipeline for the user end-to-end, from content creation, to media generation, to publishing. "
        "Communicate with the user and find out what type of content they want (tips, phrase, activity). Then route requests to the appropriate sub-agent(s) in this order: "
        "1. hr_content_agent for content/scripts/caption "
        "2. media_agent for images/video "
        "3. publish_agent for upload/publish "
        "The output of each agent step serves as input to the next. "
        "Use sub-agents in sequence. "
        "Default content_type to hr_tip if unspecified."
    ),
    sub_agents=[content_agent, media_agent, publish_agent]
)

app = App(
    root_agent=root_agent,
    name="app",
)
