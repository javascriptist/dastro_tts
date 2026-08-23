# Dastro STT

Standalone speech-to-text service built around [`OvozifyLabs/whisper-small-uz-v1`](https://huggingface.co/OvozifyLabs/whisper-small-uz-v1).

The model supports Uzbek, Russian, and English. The service accepts WAV directly and accepts MP3, M4A, and WebM when FFmpeg is installed. It also serves a browser playground at `/`.

## Run locally with Python

Requirements: Python 3.10+ and, for non-WAV uploads, FFmpeg.

```bash
cd stt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python download_model.py
uvicorn app:app --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090` and use the recorder or file picker. The model is copied to `models/whisper-small-uz-v1/` and is ignored by Git because the files are large.

For the first local test, the browser recorder creates a 16 kHz-compatible WAV, so FFmpeg is not required. Install FFmpeg to upload compressed audio:

```bash
brew install ffmpeg
```

## Run with Docker

Docker installs FFmpeg and persists the model/cache directories on the host.

```bash
cd stt
cp .env.example .env
docker compose up --build
```

The first boot downloads the model. Later boots reuse `./models` and `./huggingface-cache`.

## Railway deployment

Railway can deploy this folder directly because it contains a `Dockerfile` and `railway.toml`.

1. Create a Railway service from this repository.
2. Set the service Root Directory to `/stt` because this is a monorepo.
3. Add a Railway Volume mounted at `/data`. This prevents the model from being downloaded again after a redeploy.
4. Add these variables:

```text
MODEL_ID=OvozifyLabs/whisper-small-uz-v1
MODEL_PATH=/data/models/whisper-small-uz-v1
HF_HOME=/data/huggingface
DEVICE=cpu
STT_API_KEY=generate-a-long-random-value
CORS_ORIGINS=https://your-frontend.example
MAX_AUDIO_MB=50
MAX_AUDIO_SECONDS=300
```

Do not set `PORT` manually. Railway provides it, and the container uses it automatically. Leave `CORS_ORIGINS` as `*` only for a private test; use the exact frontend origin in production.

The first deployment downloads the model before Uvicorn starts. Wait for `Model download complete` in the deploy logs. Railway checks `/health` after the server binds. The service uses one process by design because each additional worker loads another copy of the model into memory. Start with at least 2 GB RAM and increase the service size if CPU inference runs out of memory; CPU inference works but is slower than a GPU.

After deployment:

```bash
curl https://YOUR-RAILWAY-DOMAIN/health
curl -X POST https://YOUR-RAILWAY-DOMAIN/transcribe \
  -H "X-API-Key: YOUR_STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "language=uz"
```

The playground is available at `https://YOUR-RAILWAY-DOMAIN/`. If `STT_API_KEY` is set, enter the same key in the playground's optional API key field.

## API

### `GET /health`

Returns liveness and whether the model has been loaded:

```json
{
  "status": "ok",
  "model": "OvozifyLabs/whisper-small-uz-v1",
  "model_loaded": false,
  "device": "unloaded"
}
```

### `GET /ready`

Loads the model and returns `503` until it is ready. This endpoint is protected by `STT_API_KEY` when a key is configured.

### `POST /transcribe`

Multipart fields:

- `file`: audio file
- `language`: `auto`, `uz`, `ru`, or `en`; defaults to `auto`

Response:

```json
{
  "text": "...",
  "language": "uz",
  "duration_seconds": 4.32,
  "model": "/data/models/whisper-small-uz-v1"
}
```

### `POST /v1/audio/transcriptions`

OpenAI-compatible upload shape for clients that only need `{ "text": "..." }`. It uses the same `file` and `language` fields and the same API key header.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `OvozifyLabs/whisper-small-uz-v1` | Hugging Face repository |
| `MODEL_PATH` | `./models/whisper-small-uz-v1` | Local copied model directory |
| `HF_HOME` | `./models/cache` | Hugging Face cache directory |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `STT_API_KEY` | empty | Optional `X-API-Key` protection |
| `CORS_ORIGINS` | `*` | Comma-separated browser origins |
| `MAX_AUDIO_MB` | `50` | Upload size limit |
| `MAX_AUDIO_SECONDS` | `300` | Audio duration limit |
| `PORT` | `8090` | HTTP port |
