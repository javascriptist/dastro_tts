# Dastro Voice

Standalone speech-to-text and text-to-speech services for Uzbek.

Speech-to-text is built around [`OvozifyLabs/whisper-small-uz-v1`](https://huggingface.co/OvozifyLabs/whisper-small-uz-v1) and supports Uzbek, Russian, and English. The service accepts WAV directly and accepts MP3, M4A, and WebM when FFmpeg is installed.

Text-to-speech is built around [Navoiy TTS](https://aisha.group/en/blog/navoiy-tts-open-source-uzbek-text-to-speech) (`aisha-org/navoiy-tts` on Hugging Face), a CosyVoice2-0.5B fine-tune for Uzbek. It's off by default (`TTS_ENABLED=false`) — see [Text-to-speech (Navoiy TTS)](#text-to-speech-navoiy-tts) before enabling it, since it needs a CUDA GPU.

Both live in the same `app.py` and the same browser playground at `/`, so they're meant to run **together as one service** — that's what makes the playground's transcription bench and text-to-speech bench work side by side against a single origin. There are two deployment images:

- **`Dockerfile` / `railway.toml`** — CPU-only, speech-to-text alone, cheapest. TTS stays disabled.
- **`Dockerfile.voice` / `railway.voice.toml`** — GPU, speech-to-text **and** text-to-speech together. Use this if you want the whole playground working.

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

This is the CPU-only, STT-alone deployment. If you want text-to-speech in the same service, skip ahead to [Text-to-speech (Navoiy TTS)](#text-to-speech-navoiy-tts) and deploy `Dockerfile.voice` instead — everything below still applies to it (variables, `/health` polling), it's just a different Dockerfile/toml and a GPU plan.

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

## Text-to-speech (Navoiy TTS)

[Navoiy TTS](https://aisha.group/en/blog/navoiy-tts-open-source-uzbek-text-to-speech) turns Uzbek text into speech with a set of emotion presets (`calm`, `happy`, `sad`, `angry`, `nervous`, `surprised`, `whisper`, `warm`, `tired`, `sarcastic`) and includes a normalizer for numbers, dates, times, and Uzbek Cyrillic. It's a CosyVoice2-0.5B fine-tune, distributed as a checkpoint plus the upstream [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) engine — it needs a CUDA GPU, which the CPU-only STT deployment above doesn't have. `app.py` and the playground already treat STT and TTS as one service; `Dockerfile.voice` is the image that actually turns both on together, on a GPU.

> **Heads up:** the upstream `inference.py` only documents a CLI invocation (`--cosyvoice-dir`, `--base-model-dir`, `--checkpoint`, `--reference`, `--text`, `--emotion`), not a stable Python API, and this integration was written without network access to the actual `aisha-org/navoiy-tts` and `FunAudioLLM/CosyVoice` repositories to confirm exact filenames or flags. `app.py` shells out to the documented CLI and discovers the produced clip by scanning its (fresh, per-request) working directory for the newest `.wav` file, so it should survive minor differences in the upstream output filename — but confirm the flags still match, and that a `reference.wav` ships in the checkpoint's files, once you've actually run `setup_tts_model.py`. Each request currently reloads the model from disk (no persistent in-process model yet), so latency per call will be higher than a typical TTS API; see the docstring on `NavoiyTTSRuntime` in `app.py` if you want to change that.

### Setup

Requirements: an NVIDIA GPU with CUDA, `git`, and several GB of free disk (CosyVoice2-0.5B plus the Navoiy checkpoint).

```bash
cd stt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-voice.txt
python download_model.py           # Whisper weights, same as the CPU-only setup
python setup_tts_model.py          # clones CosyVoice, downloads the base model + checkpoint
pip install -r models/CosyVoice/requirements.txt
cp .env.example .env                # then set TTS_ENABLED=true
uvicorn app:app --host 0.0.0.0 --port 8090
```

`setup_tts_model.py` is idempotent — re-run it after a failed step and it skips what's already downloaded. If the checkpoint's Hugging Face repo doesn't include a `reference.wav`, either place one at `TTS_REFERENCE_PATH` yourself or always pass a `reference` file with each `/synthesize` request (the playground's recorder can capture one).

### Run with Docker

```bash
cd stt
cp .env.example .env
docker compose up --build voice
```

Requires the NVIDIA Container Toolkit on the host. This builds `Dockerfile.voice`, which downloads the Whisper weights, clones CosyVoice, and downloads its base model during the build, so the first build is slow and the resulting image is large. It serves both STT and TTS from one port.

### Railway deployment

Deploy one service — the same repo, pointed at the GPU image instead of the CPU one:

1. Create a Railway service from `https://github.com/javascriptist/dastro_tts`, root directory empty. If you already have the CPU-only STT service running, either switch its config-as-code path (next step) or create a fresh service and retire the old one — either way you end up with one service serving both.
2. Set the service's config-as-code path to `railway.voice.toml` (Railway's service settings let you point at a non-default toml) so it builds `Dockerfile.voice` instead of `Dockerfile`.
3. Choose a GPU-backed Railway plan/region — the CPU plan used for STT-only won't run this.
4. Add a Volume mounted at `/data` so the Whisper weights, cloned engine, and downloaded TTS weights survive redeploys.
5. Set `STT_API_KEY` and `CORS_ORIGINS` as in the STT-only setup above. `TTS_ENABLED`, `MODEL_ID`, `MODEL_PATH`, `TTS_MODEL_PATH`, `TTS_COSYVOICE_DIR`, `TTS_BASE_MODEL_DIR`, and `HF_HOME` are already set in `Dockerfile.voice`.

After deployment, the playground at `https://YOUR-RAILWAY-DOMAIN/` has both benches working, and:

```bash
curl https://YOUR-RAILWAY-DOMAIN/health
curl -X POST https://YOUR-RAILWAY-DOMAIN/transcribe \
  -H "X-API-Key: YOUR_STT_API_KEY" \
  -F "file=@sample.wav" \
  -F "language=uz"
curl -X POST https://YOUR-RAILWAY-DOMAIN/synthesize \
  -H "X-API-Key: YOUR_STT_API_KEY" \
  -F "text=Assalomu alaykum, bu ovoz sinovi." \
  -F "emotion=warm" \
  --output speech.wav
```

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
  "uptime_seconds": 12.345,
  "tts_enabled": false,
  "tts_model": "aisha-org/navoiy-tts",
  "tts_model_state": "disabled",
  "tts_model_error": null,
  "tts_reference_configured": false
}
```

`model_state` (and `tts_model_state`) is `unloaded`/`disabled`, `loading`, `ready`, or `error`. `/health` is a fast liveness endpoint and does not wait for model loading. `/ready` waits for the STT model and returns `503` with the loading error until it is usable; TTS has no equivalent `/ready` endpoint because it doesn't keep a model loaded between requests — check `tts_model_state` on `/health` instead.

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

### `POST /synthesize`

Requires `TTS_ENABLED=true`. Multipart fields:

- `text`: text to speak, up to `TTS_MAX_TEXT_CHARS` characters
- `emotion`: one of `calm`, `happy`, `sad`, `angry`, `nervous`, `surprised`, `whisper`, `warm`, `tired`, `sarcastic`; defaults to `TTS_DEFAULT_EMOTION`
- `reference` (optional): a voice clip to clone instead of the configured default at `TTS_REFERENCE_PATH`

Returns `audio/wav` bytes directly, or a JSON error body with `503` while assets aren't ready and `422`/`400` for invalid input.

### `POST /v1/audio/speech`

OpenAI-compatible JSON shape: `{ "input": "...", "voice": "warm" }` (voice maps to the emotion preset). Returns `audio/wav`. Can't take an uploaded reference clip — use `/synthesize` for that.

## VoiceLab pipeline bench

`GET /voicelab` serves a second page that runs a complete voice loop against the
VoiceLab API: **live speech to text, then an OpenAI reservation agent, then text
to speech**. It is a bench for the VoiceLab models, not part of the Whisper
service. The agent is Dastro's phone host, working from the twenty sample
restaurants and the availability rules in `realtime-prompt.md`.

Both credentials stay on this server. The browser only ever calls this service,
so no key or realtime ticket reaches the page.

```bash
cp .env.example .env
# then fill in VOICELAB_API_KEY and OPENAI_API_KEY
uvicorn app:app --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090/voicelab`. Press **Start call** and speak: the
microphone listens continuously, and a pause of about a second commits your turn
to the pipeline. You can also type a guest line, click one of the sample lines,
or click a restaurant to ask about it. Every stage shows its own state and
latency, and the transcript keeps the audio for each host turn.

Set `PRELOAD_MODEL=false` when you only want this bench: it does not use the
local Whisper model unless you pick `Local Whisper` as the speech-to-text engine.

### Speech-to-text engines

| Engine | Path | Notes |
| --- | --- | --- |
| VoiceLab STT | `POST /v1/stt` | Default. One utterance per pause, polled to completion |
| VoiceLab realtime | `POST /v1/ticket` then `wss://.../v1/stt/stream` | Only when realtime is enabled for the key; falls back to the upload path automatically |
| Local Whisper | `POST /transcribe` | The `OvozifyLabs/whisper-small-uz-v1` model in this service |

The page disables the realtime option when `POST /v1/ticket` returns `404`, which
is what VoiceLab returns while realtime is off for an API key.

### Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/voicelab` | The bench page |
| `GET` | `/voicelab/config` | Key presence, realtime availability, restaurants, sample lines |
| `GET` | `/voicelab/voices?language={code}` | VoiceLab voice catalogue for the language |
| `POST` | `/voicelab/ticket?service=stt` | Mints a short-lived realtime ticket |
| `POST` | `/voicelab/stt` | One utterance to VoiceLab STT, waited out to a transcript |
| `POST` | `/voicelab/agent` | One reservation-host turn from OpenAI |
| `POST` | `/voicelab/tts` | The reply spoken by VoiceLab, returned as one WAV |

`POST /voicelab/agent` accepts an `api_key` field so the page can override the
server's OpenAI key for a session. `POST /voicelab/tts` splits replies longer
than VoiceLab's 1,000-byte limit on sentence boundaries and joins the audio into
a single WAV.

### VoiceLab and OpenAI settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOICELAB_API_KEY` | empty | VoiceLab developer key, `vlk_...` |
| `VOICELAB_BASE_URL` | `https://api.voicelab.uz` | VoiceLab API root |
| `VOICELAB_TIMEOUT_SECONDS` | `90` | Upstream request timeout |
| `VOICELAB_STT_POLL_SECONDS` | `0.7` | Gap between transcription polls |
| `VOICELAB_STT_POLL_TIMEOUT_SECONDS` | `90` | How long to wait for a transcription |
| `OPENAI_API_KEY` | empty | Key for the reservation agent |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API root |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default agent model |
| `OPENAI_TIMEOUT_SECONDS` | `45` | Agent request timeout |

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
| `TTS_ENABLED` | `false` | Turn on the `/synthesize` and `/v1/audio/speech` endpoints |
| `TTS_MODEL_ID` | `aisha-org/navoiy-tts` | Hugging Face repository for the checkpoint |
| `TTS_MODEL_PATH` | `./models/navoiy-tts` | Local directory for the downloaded checkpoint/inference script |
| `TTS_COSYVOICE_DIR` | `./models/CosyVoice` | Local clone of the CosyVoice engine |
| `TTS_COSYVOICE_REPO` | `https://github.com/FunAudioLLM/CosyVoice.git` | Git URL cloned by `setup_tts_model.py` |
| `TTS_BASE_MODEL_ID` | `FunAudioLLM/CosyVoice2-0.5B` | Hugging Face repository for the base model |
| `TTS_BASE_MODEL_DIR` | `./models/CosyVoice/pretrained_models/CosyVoice2-0.5B` | Local directory for the base model |
| `TTS_CHECKPOINT_PATH` | `./models/navoiy-tts/emotion_600h_joint.pt` | Navoiy TTS checkpoint file |
| `TTS_REFERENCE_PATH` | `./models/navoiy-tts/reference.wav` | Default voice reference clip; required unless every request uploads one |
| `TTS_DEFAULT_EMOTION` | `calm` | Emotion preset used when a request doesn't specify one |
| `TTS_MAX_TEXT_CHARS` | `500` | Maximum characters accepted per synthesis request |
| `TTS_INFERENCE_TIMEOUT_SECONDS` | `180` | Per-request subprocess timeout |
