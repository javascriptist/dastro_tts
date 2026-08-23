from __future__ import annotations

import io
import os
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_ID = os.getenv("MODEL_ID", "OvozifyLabs/whisper-small-uz-v1")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "whisper-small-uz-v1")))
HF_HOME = Path(os.getenv("HF_HOME", str(BASE_DIR / "models" / "cache")))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "50")) * 1024 * 1024
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "300"))
ALLOWED_LANGUAGES = {"auto", "uz", "ru", "en"}
SAMPLE_RATE = 16_000


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _model_source() -> str:
    if (MODEL_PATH / "config.json").is_file():
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


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    model: str


class WhisperRuntime:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._device = "unloaded"
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str:
        return self._device

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

    def load(self) -> None:
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

                device, dtype, pipeline_device = self._resolve_device(torch)
                source = _model_source()
                processor = WhisperProcessor.from_pretrained(source, cache_dir=str(HF_HOME))
                model = WhisperForConditionalGeneration.from_pretrained(
                    source,
                    cache_dir=str(HF_HOME),
                    torch_dtype=dtype,
                )
                model.to(device)
                model.eval()
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    stride_length_s=5,
                    device=pipeline_device,
                )
                self._device = device
            except Exception:
                self._pipeline = None
                raise

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.load()
        generate_kwargs: dict[str, str] = {"task": "transcribe"}
        if language != "auto":
            generate_kwargs["language"] = language
        with self._inference_lock:
            result = self._pipeline(audio, generate_kwargs=generate_kwargs)
        return str(result["text"]).strip()


runtime = WhisperRuntime()
app = FastAPI(title="Dastro Uzbek STT", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


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
    try:
        audio = decode_audio(payload)
        text = await run_in_threadpool(runtime.transcribe, audio, language)
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Transcription failed") from error
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
        "model_loaded": runtime.loaded,
        "device": runtime.device,
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
