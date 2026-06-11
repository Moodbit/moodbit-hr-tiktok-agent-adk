# Moodbit HR TikTok Agent

Standalone HR TikTok automation agent built with Google ADK.
Generates HR scripts, images, multi-clip videos, uploads assets to Azure Blob Storage,
and publishes to TikTok via the Business API. Video generation runs as a long-running
job with progress streaming and Firestore-backed persistence for Cloud Run autoscaling.

## Project Structure

```
hr-tiktok-adk/
├── app/                       # Core agent code
│   ├── agent.py               # ADK agent and tool wiring
│   ├── fast_api_app.py        # FastAPI entrypoint (ADK web)
│   ├── services_registry.py   # Firestore session service registration
│   ├── tiktok_influencer.py   # Orchestrator pipeline
│   ├── controllers/           # Content generation
│   ├── services/              # External services (TikTok, Azure, Veo, TTS, image gen)
│   ├── ima/                    # Persona reference image
│   └── app_utils/             # App utilities and helpers
├── src/                       # Tool implementations
├── tests/                     # Unit and integration tests
├── AGENTS.md                  # Agents CLI workflow guide
└── pyproject.toml             # Project dependencies
```

## Architecture Diagram

```mermaid
flowchart TB
	U[User] -->|API requests| FA[FastAPI ADK Web App]
	FA --> ADK[ADK App / Root Agent]

	ADK --> CA[Content Agent]
	ADK --> MA[Media Agent]
	ADK --> PA[Publish Agent]

	CA --> CT[Content Tools]
	CT --> HC[HR Content Controllers]
	HC --> LLM[Gemini (google-genai)]

	MA --> MT[Media Tools]
	MT --> IMG[Image Generator Service]
	MT --> VID[Video Generator Service (Veo)]
	MT --> JOBS[Video Job Store]
	JOBS --> FS[Firestore]

	MA --> AZU[Azure Blob Storage]

	PA --> PT[Publish Tools]
	PT --> TT[TikTok Business API]
	PT --> AZU

	FA --> LOGS[Cloud Logging / OTEL]
	FA --> FS
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
```

## Core Capabilities

- Generate and post HR TikTok videos across 3 content types (phrase, activity, tip)
- Generate persona images (Gemini) and video clips with audio (Veo 3)
- Upload clips and final video to Azure Blob Storage, then publish via TikTok Business API
- Long-running video generation with Firestore-backed job tracking + progress streaming
- Manual pipeline runs only (no scheduler)

## Environment Variables

| Variable | Description |
| --- | --- |
| `GCP_PROJECT_ID` | GCP project ID for Vertex AI |
| `GEMINI_API_KEY` | Google Gemini API key (image gen fallback) |
| `AZ_TIKTOK_STORAGE_CONNECTION_STRING` | Azure Storage connection string for TikTok videos |
| `TIKTOK_AUTH_JSON` | JSON string containing TikTok auth tokens |
| `TIKTOK_CLIENT_KEY` | TikTok app client key |
| `TIKTOK_CLIENT_SECRET` | TikTok app client secret |
| `SKIP_IMAGE_GEN` | If `1`, skip image generation and use base image |
| `ADK_SESSION_SERVICE_URI` | Session backend URI (use `firestore://<collection>` for Cloud Run) |
| `ADK_FIRESTORE_ROOT_COLLECTION` | Root collection for ADK sessions (used by Firestore backend) |
| `ADK_VIDEO_JOBS_COLLECTION` | Firestore collection for video jobs (default `video_jobs`) |
| `ALLOW_ORIGINS` | Comma-separated CORS origins for the API |
| `LOGS_BUCKET_NAME` | GCS bucket name for ADK artifacts/logs |

## Persona Image

Place your persona reference photo at:

```
app/ima/Imagebase.png
```

## Firestore Session Storage (Cloud Run)

Set the session backend and collection when deploying:

- `ADK_SESSION_SERVICE_URI=firestore://sessions`
- `ADK_FIRESTORE_ROOT_COLLECTION=sessions`

The Firestore registry is loaded via `app/services_registry.py`.

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
