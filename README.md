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
uvicorn app:app --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090` and use the recorder or file picker. The service starts the HTTP server immediately and loads the model in the background. The model is downloaded into the Hugging Face cache on first startup. To explicitly copy it into `models/whisper-small-uz-v1/` instead, run `python download_model.py` before starting the service; that directory is ignored by Git because the files are large.

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

The first boot downloads the model in the background while the HTTP server is already available. Later boots reuse `./models` and `./huggingface-cache`.

## Railway deployment

Railway can deploy this folder directly because it contains a `Dockerfile` and `railway.toml`.

1. Create a Railway service from `https://github.com/javascriptist/dastro_tts`.
2. Leave the service Root Directory empty (or set it to `/`). This repository already contains the `Dockerfile` at its root. Only use `/stt` when deploying from the original Dastro monorepo.
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
MAX_NEW_TOKENS=256
TORCH_NUM_THREADS=4
CPU_QUANTIZE=true
TORCH_QUANTIZED_ENGINE=fbgemm
PRELOAD_MODEL=true
LOG_LEVEL=INFO
```

Do not set `PORT` manually. Railway provides it, and the container uses it automatically. Leave `CORS_ORIGINS` as `*` only for a private test; use the exact frontend origin in production.

The first deployment starts Uvicorn immediately, then downloads and loads the model in the background. Railway checks `/health` while this happens. Watch the logs for `model_load_started`, `model_weights_loaded`, `cpu_dynamic_quantization_enabled`, and `model_ready`. Check `/health` until `model_state` is `ready` before sending audio. The service uses one process by design because each additional worker loads another copy of the model into memory. Start with at least 2 GB RAM and increase the service size if CPU inference runs out of memory; CPU inference works but is slower than a GPU. Dynamic int8 quantization is enabled by default on CPU to reduce inference latency and memory use.

After deployment:

```bash
curl https://YOUR-RAILWAY-DOMAIN/health
curl https://YOUR-RAILWAY-DOMAIN/ready -H "X-API-Key: YOUR_STT_API_KEY"
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
  "model_state": "loading",
  "device": "cpu",
  "model_error": null,
  "model_load_seconds": null,
  "uptime_seconds": 12.345
}
```

`model_state` is `unloaded`, `loading`, `ready`, or `error`. `/health` is a fast liveness endpoint and does not wait for model loading. `/ready` waits for the model and returns `503` with the loading error until the model is usable.

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
| `PRELOAD_MODEL` | `true` | Load the model in a background thread at startup |
| `MAX_NEW_TOKENS` | `256` | Maximum generated transcript tokens; values below 256 are raised to avoid clipping normal recordings |
| `TORCH_NUM_THREADS` | `4` | CPU thread limit; tune to the replica's vCPU count |
| `CPU_QUANTIZE` | `true` | Apply dynamic int8 quantization on CPU |
| `TORCH_QUANTIZED_ENGINE` | `fbgemm` | Preferred CPU quantization backend; falls back to an installed backend |
| `LOG_LEVEL` | `INFO` | Application log level |
| `STT_API_KEY` | empty | Optional `X-API-Key` protection |
| `CORS_ORIGINS` | `*` | Comma-separated browser origins |
| `MAX_AUDIO_MB` | `50` | Upload size limit |
| `MAX_AUDIO_SECONDS` | `300` | Audio duration limit |
| `PORT` | `8090` | HTTP port |
