# hr-tiktok-adk

Standalone HR TikTok automation agent built with Google ADK.
Recreates the original pipeline without importing from the legacy repo.

## Project Structure

```
hr-tiktok-adk/
├── app/         # Core agent code
│   ├── agent.py               # ADK agent and tool wiring
│   ├── tiktok_influencer.py   # Orchestrator pipeline
│   ├── scheduler.py           # Schedule helpers
│   ├── controllers/           # Content generation
│   ├── services/              # External services (TikTok, Azure, Veo, TTS, image gen)
│   ├── ima/                    # Persona reference image
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Gemini CLI](https://github.com/google-gemini/gemini-cli) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground

## Core Capabilities

- Generate and post HR TikTok videos across 4 content types (news, phrase, activity, tip)
- Generate persona images (Gemini), voiceover (Gemini TTS), and video clips (Veo 3)
- Upload to Azure Blob Storage and publish via TikTok Business API
- Manual pipeline runs only (no scheduler)

## Environment Variables

| Variable | Description |
| --- | --- |
| `BING_SEARCH_V7_SUBSCRIPTION_KEY` | Bing Search API key |
| `BING_SEARCH_V7_ENDPOINT` | Bing Search endpoint URL |
| `GCP_PROJECT_ID` | GCP project ID for Vertex AI |
| `GEMINI_API_KEY` | Google Gemini API key (image gen fallback) |
| `GEMINI_TTS_API_KEY` | Google Gemini API key (TTS fallback) |
| `AZ_TIKTOK_STORAGE_CONNECTION_STRING` | Azure Storage connection string for TikTok videos |
| `AZURE_STORAGE_KEY` | Azure Storage account key for schedule JSON |
| `AZURE_FILE_POSTS_DATE` | Azure Blob URL to schedule JSON |
| `TIKTOK_AUTH_JSON` | JSON string containing TikTok auth tokens |
| `TIKTOK_CLIENT_KEY` | TikTok app client key |
| `TIKTOK_CLIENT_SECRET` | TikTok app client secret |
| `TTS_PARTS` | Number of script parts per video (default `3`) |
| `TTS_MAX_WORDS` | Max words per script part (default `35`) |
| `SKIP_IMAGE_GEN` | If `1`, skip image generation and use base image |

## Persona Image

Place your persona reference photo at:

```
app/ima/Imagebase.png
```
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.
