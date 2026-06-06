---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOURUSERNAME/voxify.git && cd voxify

# 2. Add API keys
cp .env.example .env   # then set GROQ_API_KEY inside

# 3. Run
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000)

---

## Key Engineering Decisions

**No ffmpeg** — audio is resampled to 16kHz mono PCM WAV entirely in Python
using `wave` and `numpy`. Eliminates a heavy system dependency and keeps
the Docker image lean.

**Dual API fallback** — if Groq Whisper fails, the app automatically retries
via HuggingFace Inference API. Handled in `transcriber.py` with zero UI impact.

**File validation on both sides** — the React frontend disables the submit
button and shows an inline warning for files >25MB. The FastAPI backend
independently returns HTTP 413. Groq's API hard limit never gets hit.

**Multi-stage Docker builds** — builder stage installs deps, runtime stage
copies only the output. Backend image: ~250MB. Frontend: Nginx alpine +
static `dist/` only.

---

## API Reference

Full interactive docs at `/docs` (Swagger) and `/redoc`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check → `{"status":"ok","version":"1.0.0"}` |
| `POST` | `/transcribe` | Upload audio → returns transcript + timestamps |
| `POST` | `/summarize` | Transcript → summary + action items + speakers |
| `POST` | `/export/pdf` | Returns styled PDF report as bytes |
| `POST` | `/export/txt` | Returns plain text report |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) — free tier |
| `HF_API_KEY` | No | HuggingFace fallback STT |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |
| `VITE_API_URL` | Render only | Backend URL baked into React build |
| `HOST_PORT` | No | Host port for Compose (default: `3000`) |

---

## Tests

```bash
# Backend — 21 tests
pip install -r requirements-dev.txt
pytest tests/ -v

# Frontend — 8 tests
cd frontend && npm ci && npm run test
```

All external API calls (Groq, HuggingFace) are mocked — test suite
runs fully offline with no credentials required.

---

## Deployment

### Render (live)
1. Push repo to GitHub
2. Render Dashboard → **New → Blueprint** → connect repo
3. Render reads `render.yaml` — creates both services automatically
4. Set `GROQ_API_KEY` in Render dashboard environment variables

### Self-hosted
```bash
docker compose up --build -d
```

### CI/CD
Every push to `main` triggers: lint → pytest → vitest → Docker build
(amd64 + arm64) → push to GHCR. Git tags (`v*.*.*`) trigger auto-deploy.

---

## Contributing

```bash
git checkout -b feature/your-feature
# make changes + add tests
pytest tests/ -v && cd frontend && npm test
git commit -m "feat: your feature"
git push origin feature/your-feature
# open a Pull Request
```

---

## License

MIT