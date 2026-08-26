from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_ID = os.getenv("MODEL_ID", "OvozifyLabs/whisper-small-uz-v1")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "whisper-small-uz-v1")))
HF_HOME = Path(os.getenv("HF_HOME", str(BASE_DIR / "models" / "cache")))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "50")) * 1024 * 1024
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "300"))
MAX_NEW_TOKENS = max(int(os.getenv("MAX_NEW_TOKENS", "256")), 256)
SHORT_AUDIO_SECONDS = 30.0
ALLOWED_LANGUAGES = {"auto", "uz", "ru", "en"}
SAMPLE_RATE = 16_000
STARTED_AT = time.monotonic()

# Navoiy TTS (https://aisha.group/en/blog/navoiy-tts-open-source-uzbek-text-to-speech) is a
# CosyVoice2-0.5B fine-tune for Uzbek from aisha-org/navoiy-tts on Hugging Face. Unlike the
# Whisper STT model above, it needs CUDA and the upstream CosyVoice engine cloned from GitHub,
# so it cannot share the CPU-only STT deployment. Run setup_tts_model.py (see README) and set
# TTS_ENABLED=true to turn it on; it stays off by default.
TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() not in {"0", "false", "no", "off"}
TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "aisha-org/navoiy-tts")
TTS_MODEL_PATH = Path(os.getenv("TTS_MODEL_PATH", str(BASE_DIR / "models" / "navoiy-tts")))
TTS_COSYVOICE_DIR = Path(os.getenv("TTS_COSYVOICE_DIR", str(BASE_DIR / "models" / "CosyVoice")))
TTS_BASE_MODEL_DIR = Path(
    os.getenv("TTS_BASE_MODEL_DIR", str(TTS_COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"))
)
TTS_CHECKPOINT_PATH = Path(os.getenv("TTS_CHECKPOINT_PATH", str(TTS_MODEL_PATH / "emotion_600h_joint.pt")))
TTS_REFERENCE_PATH = Path(os.getenv("TTS_REFERENCE_PATH", str(TTS_MODEL_PATH / "reference.wav")))
TTS_INFERENCE_SCRIPT = Path(os.getenv("TTS_INFERENCE_SCRIPT", str(TTS_MODEL_PATH / "inference.py")))
TTS_PYTHON_BIN = os.getenv("TTS_PYTHON_BIN", sys.executable)
TTS_DEFAULT_EMOTION = os.getenv("TTS_DEFAULT_EMOTION", "calm")
TTS_MAX_TEXT_CHARS = int(os.getenv("TTS_MAX_TEXT_CHARS", "500"))
TTS_INFERENCE_TIMEOUT_SECONDS = float(os.getenv("TTS_INFERENCE_TIMEOUT_SECONDS", "180"))
TTS_SAMPLE_RATE = 24_000
ALLOWED_EMOTIONS = {
    "calm",
    "happy",
    "sad",
    "angry",
    "nervous",
    "surprised",
    "whisper",
    "warm",
    "tired",
    "sarcastic",
}

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s dastro-stt %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dastro-stt")


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _model_source() -> str:
    has_weights = any(MODEL_PATH.glob("model*.safetensors")) or any(MODEL_PATH.glob("pytorch_model*.bin"))
    if (MODEL_PATH / "config.json").is_file() and has_weights:
        return str(MODEL_PATH)
    return MODEL_ID


def _requested_device() -> str:
    return os.getenv("DEVICE", "auto").lower()


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE or audio.size == 0:
        return audio
    target_size = max(1, round(audio.size * SAMPLE_RATE / source_rate))
    source_positions = np.arange(audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_size, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _decode_wav(payload: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            source_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("The WAV file could not be decoded") from error

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648
    else:
        raise ValueError("Only 8-bit, 16-bit, and 32-bit WAV files are supported")

    if channels > 1:
        usable_size = audio.size - (audio.size % channels)
        audio = audio[:usable_size].reshape(-1, channels).mean(axis=1)
    return _resample(audio, source_rate)


def _decode_with_ffmpeg(payload: bytes) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise ValueError("This audio format needs FFmpeg. Upload WAV or install FFmpeg on the server.")

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip() or "FFmpeg could not decode the audio"
        raise ValueError(detail)
    return np.frombuffer(result.stdout, dtype=np.float32)


def decode_audio(payload: bytes) -> np.ndarray:
    if payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        audio = _decode_wav(payload)
    else:
        audio = _decode_with_ffmpeg(payload)
    if audio.size == 0:
        raise ValueError("The audio file is empty")
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio)))
    if peak > 1:
        audio = audio / peak
    if audio.size / SAMPLE_RATE > MAX_AUDIO_SECONDS:
        raise ValueError(f"Audio must be {MAX_AUDIO_SECONDS:g} seconds or shorter")
    return audio


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(sample_rate)
        sink.writeframes(pcm.tobytes())


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    model: str


class WhisperRuntime:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._pipeline: Any | None = None
        self._pipeline_device: int | Any = -1
        self._device = "unloaded"
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._state = "unloaded"
        self._last_error: str | None = None
        self._load_started_at: float | None = None
        self._load_duration_seconds: float | None = None
        self._cpu_quantized = False

    @property
    def loaded(self) -> bool:
        return self._state == "ready" and self._model is not None and self._processor is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def state(self) -> str:
        return self._state

    def status(self) -> dict[str, Any]:
        return {
            "model_loaded": self.loaded,
            "model_state": self._state,
            "device": self._device,
            "cpu_quantized": self._cpu_quantized,
            "model_error": self._last_error,
            "model_load_seconds": self._load_duration_seconds,
        }

    def _resolve_device(self, torch: Any) -> tuple[str, Any, int | Any]:
        requested = _requested_device()
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("DEVICE=mps was requested, but Apple Metal is not available")
        if requested not in {"cpu", "cuda", "mps"}:
            raise RuntimeError("DEVICE must be auto, cpu, cuda, or mps")
        dtype = torch.float16 if requested == "cuda" else torch.float32
        pipeline_device: int | Any = 0 if requested == "cuda" else -1
        if requested == "mps":
            pipeline_device = torch.device("mps")
        return requested, dtype, pipeline_device

    def load(self, wait: bool = True) -> None:
        if self._model is not None and self._processor is not None:
            return
        if not self._load_lock.acquire(blocking=wait):
            raise RuntimeError("Model is still loading; retry shortly")
        self._load_started_at = time.monotonic()
        self._load_duration_seconds = None
        self._state = "loading"
        self._last_error = None
        source = _model_source()
        logger.info(
            "model_load_started model_id=%s source=%s device_request=%s",
            MODEL_ID,
            source,
            _requested_device(),
        )
        try:
            if self._model is not None and self._processor is not None:
                return
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                device, dtype, pipeline_device = self._resolve_device(torch)
                configured_threads = os.getenv("TORCH_NUM_THREADS", "4").strip()
                if configured_threads:
                    torch.set_num_threads(int(configured_threads))
                logger.info("torch_threads_configured threads=%d", torch.get_num_threads())
                processor = WhisperProcessor.from_pretrained(source, cache_dir=str(HF_HOME))
                logger.info("processor_loaded source=%s", source)
                model = WhisperForConditionalGeneration.from_pretrained(
                    source,
                    cache_dir=str(HF_HOME),
                    dtype=dtype,
                )
                logger.info("model_weights_loaded source=%s device=%s", source, device)
                model.to(device)
                if device == "cpu" and os.getenv("CPU_QUANTIZE", "true").lower() not in {"0", "false", "no", "off"}:
                    try:
                        supported_engines = [engine for engine in torch.backends.quantized.supported_engines if engine != "none"]
                        preferred_engine = os.getenv("TORCH_QUANTIZED_ENGINE", "fbgemm")
                        quantized_engine = preferred_engine if preferred_engine in supported_engines else (supported_engines[0] if supported_engines else None)
                        if quantized_engine is None:
                            raise RuntimeError("PyTorch has no supported quantization engine")
                        torch.backends.quantized.engine = quantized_engine
                        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                        self._cpu_quantized = True
                        logger.info("cpu_dynamic_quantization_enabled engine=%s", quantized_engine)
                    except Exception as error:
                        self._cpu_quantized = False
                        logger.exception("cpu_dynamic_quantization_failed; continuing with float32 model")
                model.eval()
                self._model = model
                self._processor = processor
                self._pipeline_device = pipeline_device
                self._device = device
                self._state = "ready"
                self._load_duration_seconds = round(time.monotonic() - self._load_started_at, 3)
                logger.info(
                    "model_ready model_id=%s device=%s load_seconds=%.3f",
                    MODEL_ID,
                    device,
                    self._load_duration_seconds,
                )
            except Exception as error:
                self._model = None
                self._processor = None
                self._pipeline = None
                self._cpu_quantized = False
                self._state = "error"
                self._last_error = f"{type(error).__name__}: {error}"[:500]
                logger.exception("model_load_failed model_id=%s source=%s", MODEL_ID, source)
                raise
        finally:
            if self._load_started_at is not None and self._load_duration_seconds is None:
                self._load_duration_seconds = round(time.monotonic() - self._load_started_at, 3)
            self._load_lock.release()

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.load(wait=False)
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Another transcription is already running; retry shortly")
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
        }
        if language != "auto":
            generate_kwargs["language"] = language
        inference_started_at = time.perf_counter()
        logger.info(
            "inference_started duration_seconds=%.3f language=%s max_new_tokens=%d quantized=%s",
            audio.size / SAMPLE_RATE,
            language,
            MAX_NEW_TOKENS,
            self._cpu_quantized,
        )
        try:
            if audio.size <= SAMPLE_RATE * SHORT_AUDIO_SECONDS:
                import torch

                feature_started_at = time.perf_counter()
                inputs = self._processor(
                    audio,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs.input_features.to(self._device)
                attention_mask = getattr(inputs, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._device)
                logger.info("features_ready duration_ms=%.1f", (time.perf_counter() - feature_started_at) * 1000)
                with torch.inference_mode():
                    if attention_mask is None:
                        predicted_ids = self._model.generate(input_features, **generate_kwargs)
                    else:
                        predicted_ids = self._model.generate(input_features, attention_mask=attention_mask, **generate_kwargs)
                text = str(self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]).strip()
                logger.info(
                    "inference_complete duration_seconds=%.3f generation_seconds=%.3f characters=%d",
                    audio.size / SAMPLE_RATE,
                    time.perf_counter() - inference_started_at,
                    len(text),
                )
                return text

            if self._pipeline is None:
                from transformers import pipeline

                logger.info("long_audio_pipeline_started duration_seconds=%.3f", audio.size / SAMPLE_RATE)
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self._model,
                    tokenizer=self._processor.tokenizer,
                    feature_extractor=self._processor.feature_extractor,
                    chunk_length_s=30,
                    stride_length_s=5,
                    device=self._pipeline_device,
                )
            result = self._pipeline(audio, generate_kwargs=generate_kwargs)
            text = str(result["text"]).strip()
            logger.info(
                "inference_complete duration_seconds=%.3f generation_seconds=%.3f characters=%d",
                audio.size / SAMPLE_RATE,
                time.perf_counter() - inference_started_at,
                len(text),
            )
            return text
        finally:
            self._inference_lock.release()

    def start_preload(self) -> None:
        if os.getenv("PRELOAD_MODEL", "true").lower() in {"0", "false", "no", "off"}:
            logger.info("model_preload_disabled")
            return

        def preload() -> None:
            try:
                self.load()
            except Exception:
                logger.error("model_preload_failed; inspect /health and service logs")

        threading.Thread(target=preload, name="whisper-model-loader", daemon=True).start()
        logger.info("model_preload_scheduled model_id=%s", MODEL_ID)


class NavoiyTTSRuntime:
    """Wraps the Navoiy TTS CLI (upstream `inference.py`) as a subprocess per request.

    The upstream project only documents a CLI invocation, not a stable importable Python API,
    so this shells out to it rather than guessing internal module/class names. Because the
    checkpoint (~0.5B params) is reloaded onto the GPU on every call, each request is slow;
    switch this class to holding an in-process model instead once the upstream Python API is
    confirmed against the version actually installed.
    """

    def __init__(self) -> None:
        self._state = "disabled" if not TTS_ENABLED else "unloaded"
        self._last_error: str | None = None
        self._check_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def status(self) -> dict[str, Any]:
        return {
            "tts_enabled": TTS_ENABLED,
            "tts_model": TTS_MODEL_ID,
            "tts_model_state": self._state,
            "tts_model_error": self._last_error,
            "tts_reference_configured": TTS_REFERENCE_PATH.is_file(),
        }

    def _verify_assets(self) -> None:
        required = {
            "TTS_COSYVOICE_DIR": TTS_COSYVOICE_DIR,
            "TTS_BASE_MODEL_DIR": TTS_BASE_MODEL_DIR,
            "TTS_CHECKPOINT_PATH": TTS_CHECKPOINT_PATH,
            "TTS_INFERENCE_SCRIPT": TTS_INFERENCE_SCRIPT,
        }
        missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise RuntimeError(
                "Missing Navoiy TTS assets: "
                + ", ".join(missing)
                + ". Run setup_tts_model.py and install CosyVoice's requirements, or see README."
            )
        if not TTS_REFERENCE_PATH.is_file():
            logger.warning(
                "tts_reference_missing path=%s; every /synthesize call must upload a reference clip",
                TTS_REFERENCE_PATH,
            )

    def load(self) -> None:
        if not TTS_ENABLED:
            raise RuntimeError("TTS is disabled on this deployment; set TTS_ENABLED=true after setup")
        if self._state == "ready":
            return
        if not self._check_lock.acquire(blocking=False):
            return
        try:
            self._state = "loading"
            self._last_error = None
            self._verify_assets()
            self._state = "ready"
            logger.info("tts_assets_verified model=%s checkpoint=%s", TTS_MODEL_ID, TTS_CHECKPOINT_PATH)
        except Exception as error:
            self._state = "error"
            self._last_error = f"{type(error).__name__}: {error}"[:500]
            logger.exception("tts_assets_check_failed")
            raise
        finally:
            self._check_lock.release()

    def start_preload(self) -> None:
        if not TTS_ENABLED:
            logger.info("tts_disabled")
            return

        def preload() -> None:
            try:
                self.load()
            except Exception:
                logger.error("tts_asset_check_failed; inspect /health and service logs")

        threading.Thread(target=preload, name="tts-asset-check", daemon=True).start()
        logger.info("tts_asset_check_scheduled model=%s", TTS_MODEL_ID)

    def synthesize(self, text: str, emotion: str, reference_path: Path) -> bytes:
        self.load()
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Another synthesis request is already running; retry shortly")
        try:
            with tempfile.TemporaryDirectory(prefix="navoiy-tts-out-") as tmp_dir:
                command = [
                    TTS_PYTHON_BIN,
                    str(TTS_INFERENCE_SCRIPT),
                    "--cosyvoice-dir",
                    str(TTS_COSYVOICE_DIR),
                    "--base-model-dir",
                    str(TTS_BASE_MODEL_DIR),
                    "--checkpoint",
                    str(TTS_CHECKPOINT_PATH),
                    "--reference",
                    str(reference_path),
                    "--text",
                    text,
                    "--emotion",
                    emotion,
                ]
                logger.info("tts_inference_started emotion=%s characters=%d", emotion, len(text))
                started_at = time.perf_counter()
                try:
                    result = subprocess.run(
                        command,
                        cwd=tmp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=TTS_INFERENCE_TIMEOUT_SECONDS,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(
                        f"Navoiy TTS did not finish within {TTS_INFERENCE_TIMEOUT_SECONDS:g}s"
                    ) from error

                # The documented CLI doesn't take an --output flag, so the produced clip is
                # discovered by scanning the (empty, per-request) working directory afterward.
                produced = sorted(
                    Path(tmp_dir).rglob("*.wav"), key=lambda item: item.stat().st_mtime, reverse=True
                )
                if result.returncode != 0 or not produced:
                    detail = result.stderr.decode("utf-8", errors="replace").strip()[-2000:]
                    raise RuntimeError(detail or "Navoiy TTS inference failed")
                audio_bytes = produced[0].read_bytes()
                logger.info(
                    "tts_inference_complete emotion=%s characters=%d duration_seconds=%.3f bytes=%d",
                    emotion,
                    len(text),
                    time.perf_counter() - started_at,
                    len(audio_bytes),
                )
                return audio_bytes
        finally:
            self._inference_lock.release()


runtime = WhisperRuntime()
tts_runtime = NavoiyTTSRuntime()
app = FastAPI(title="Dastro Uzbek Voice", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.on_event("startup")
async def preload_model() -> None:
    logger.info(
        "service_started model_id=%s model_path=%s hf_home=%s",
        MODEL_ID,
        MODEL_PATH,
        HF_HOME,
    )
    runtime.start_preload()
    tts_runtime.start_preload()


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return response


def _check_api_key(request: Request) -> None:
    expected = os.getenv("STT_API_KEY", "")
    if expected and request.headers.get("x-api-key") != expected:
        raise HTTPException(status_code=401, detail="Invalid STT API key")


async def _read_audio(file: UploadFile) -> bytes:
    payload = await file.read(MAX_AUDIO_BYTES + 1)
    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    if not payload:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    return payload


async def _transcribe_upload(request: Request, file: UploadFile, language: str) -> TranscriptionResponse:
    _check_api_key(request)
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=422, detail="language must be auto, uz, ru, or en")
    payload = await _read_audio(file)
    logger.info(
        "transcription_started filename=%s content_type=%s bytes=%d language=%s",
        file.filename or "unknown",
        file.content_type or "unknown",
        len(payload),
        language,
    )
    try:
        audio = decode_audio(payload)
    except ValueError as error:
        logger.warning("audio_decode_failed filename=%s error=%s", file.filename or "unknown", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        text = await run_in_threadpool(runtime.transcribe, audio, language)
    except RuntimeError as error:
        logger.warning("transcription_unavailable state=%s error=%s", runtime.state, error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("transcription_failed filename=%s", file.filename or "unknown")
        raise HTTPException(status_code=500, detail="Transcription failed") from error
    logger.info(
        "transcription_complete filename=%s duration_seconds=%.3f characters=%d",
        file.filename or "unknown",
        audio.size / SAMPLE_RATE,
        len(text),
    )
    return TranscriptionResponse(
        text=text,
        language=language,
        duration_seconds=round(audio.size / SAMPLE_RATE, 3),
        model=MODEL_ID if _model_source() == MODEL_ID else str(MODEL_PATH),
    )


@app.get("/")
def playground() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 3),
        **runtime.status(),
        **tts_runtime.status(),
    }


@app.get("/ready")
def ready(request: Request) -> dict[str, Any]:
    _check_api_key(request)
    try:
        runtime.load()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {error}") from error
    return {"status": "ready", "model": MODEL_ID, "device": runtime.device}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
) -> TranscriptionResponse:
    return await _transcribe_upload(request, file, language)


@app.post("/v1/audio/transcriptions")
async def openai_compatible_transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict[str, str]:
    result = await _transcribe_upload(request, file, language)
    return {"text": result.text}


async def _synthesize(request: Request, text: str, emotion: str, reference: UploadFile | None) -> bytes:
    _check_api_key(request)
    if not TTS_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="TTS is not enabled on this deployment. Run setup_tts_model.py and set TTS_ENABLED=true.",
        )
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > TTS_MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"text must be {TTS_MAX_TEXT_CHARS} characters or fewer")
    emotion = (emotion or TTS_DEFAULT_EMOTION).strip().lower()
    if emotion not in ALLOWED_EMOTIONS:
        raise HTTPException(
            status_code=422, detail=f"emotion must be one of: {', '.join(sorted(ALLOWED_EMOTIONS))}"
        )

    reference_path = TTS_REFERENCE_PATH
    tmp_reference_dir: tempfile.TemporaryDirectory[str] | None = None
    if reference is not None:
        payload = await reference.read(MAX_AUDIO_BYTES + 1)
        if len(payload) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Reference audio is too large")
        if not payload:
            raise HTTPException(status_code=400, detail="Reference audio is empty")
        try:
            reference_audio = decode_audio(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        tmp_reference_dir = tempfile.TemporaryDirectory(prefix="navoiy-tts-ref-")
        reference_path = Path(tmp_reference_dir.name) / "reference.wav"
        _write_wav(reference_path, reference_audio, SAMPLE_RATE)
    elif not TTS_REFERENCE_PATH.is_file():
        raise HTTPException(
            status_code=422,
            detail="No reference voice is configured. Upload a reference clip or set TTS_REFERENCE_PATH.",
        )

    try:
        logger.info("synthesis_started emotion=%s characters=%d", emotion, len(text))
        audio_bytes = await run_in_threadpool(tts_runtime.synthesize, text, emotion, reference_path)
    except RuntimeError as error:
        logger.warning("tts_unavailable state=%s error=%s", tts_runtime.state, error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("synthesis_failed")
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from error
    finally:
        if tmp_reference_dir is not None:
            tmp_reference_dir.cleanup()
    return audio_bytes


@app.post("/synthesize")
async def synthesize(
    request: Request,
    text: str = Form(...),
    emotion: str = Form(TTS_DEFAULT_EMOTION),
    reference: UploadFile | None = File(None),
) -> Response:
    audio_bytes = await _synthesize(request, text, emotion, reference)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


class SpeechRequest(BaseModel):
    input: str
    voice: str = TTS_DEFAULT_EMOTION


@app.post("/v1/audio/speech")
async def openai_compatible_speech(request: Request, body: SpeechRequest) -> Response:
    audio_bytes = await _synthesize(request, body.input, body.voice, None)
    return Response(content=audio_bytes, media_type="audio/wav")
