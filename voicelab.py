"""VoiceLab pipeline bench: live STT, an OpenAI agent turn, and VoiceLab TTS.

The VoiceLab and OpenAI keys stay on this server. The browser page talks only
to the routes in this module, so no credential is ever sent to the browser.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
import wave
from datetime import date
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field


logger = logging.getLogger("dastro-stt.voicelab")

VOICELAB_BASE_URL = os.getenv("VOICELAB_BASE_URL", "https://api.voicelab.uz").rstrip("/")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
VOICELAB_TIMEOUT_SECONDS = float(os.getenv("VOICELAB_TIMEOUT_SECONDS", "90"))
STT_POLL_SECONDS = float(os.getenv("VOICELAB_STT_POLL_SECONDS", "0.7"))
STT_POLL_TIMEOUT_SECONDS = float(os.getenv("VOICELAB_STT_POLL_TIMEOUT_SECONDS", "90"))
TTS_TEXT_BYTE_LIMIT = 1000
AGENT_LANGUAGES = {"uz", "ru", "en"}
TERMINAL_FAILURE_STATES = {"failed", "error", "cancelled", "canceled", "rejected"}
REALTIME_PROBE_TTL_SECONDS = 120.0


def voicelab_api_key() -> str:
    """The .env.example in this repo uses a lowercase name; accept both."""
    return (os.getenv("VOICELAB_API_KEY") or os.getenv("voicelab_api_key") or "").strip()


def openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


# The twenty restaurants Dastro works with, mirrored from realtime-prompt.md.
# They are the only ones the agent may book, and the page renders them as chips.
RESTAURANTS: list[dict[str, Any]] = [
    {"name": "Afsona", "district": "Shayxontohur", "cuisine": "Uzbek", "price": 3, "seats": 100, "note": "Classic Uzbek, big plov."},
    {"name": "Caravan", "district": "Yakkasaroy", "cuisine": "Uzbek", "price": 3, "seats": 120, "note": "Uzbek, popular with tourists."},
    {"name": "Besh Qozon", "district": "Yunusobod", "cuisine": "Uzbek", "price": 2, "seats": 220, "note": "Huge, famous for plov, great for large groups."},
    {"name": "Khiva Restaurant", "district": "Mirobod", "cuisine": "Uzbek", "price": 2, "seats": 130, "note": "Uzbek, relaxed."},
    {"name": "Khan Chapan", "district": "Olmazor", "cuisine": "Uzbek", "price": 3, "seats": 180, "note": "Uzbek, Asian and European mix."},
    {"name": "Lali", "district": "Yunusobod", "cuisine": "Uzbek", "price": 3, "seats": 72, "note": "Uzbek, intimate."},
    {"name": "Cafe 1991", "district": "Yunusobod", "cuisine": "Uzbek", "price": 3, "seats": 80, "note": "Uzbek and Lebanese."},
    {"name": "Ember & Embar", "district": "Shayxontohur", "cuisine": "Steak", "price": 4, "seats": 88, "note": "Steakhouse, upscale."},
    {"name": "Fillet Restaurant", "district": "Yakkasaroy", "cuisine": "Steak", "price": 4, "seats": 84, "note": "Steak and Latin American."},
    {"name": "Myasnoy Steak House", "district": "Yakkasaroy", "cuisine": "Steak", "price": 4, "seats": 90, "note": "Steakhouse, European."},
    {"name": "Syrovarnya", "district": "Yunusobod", "cuisine": "Italian", "price": 3, "seats": 110, "note": "Italian, fresh pasta and house-made cheese, garden setting."},
    {"name": "Sette Restaurant and Bar", "district": "Mirzo Ulugbek", "cuisine": "Italian", "price": 4, "seats": 96, "note": "Italian, smart."},
    {"name": "Actor Restaurant", "district": "Yakkasaroy", "cuisine": "European", "price": 3, "seats": 76, "note": "European, international."},
    {"name": "Yuzhanin", "district": "Mirobod", "cuisine": "European", "price": 3, "seats": 70, "note": "European and Russian."},
    {"name": "Bibigon Cafe", "district": "Mirobod", "cuisine": "Cafe", "price": 2, "seats": 65, "note": "Light European, good for lunch and coffee."},
    {"name": "Kosebasi", "district": "Shayxontohur", "cuisine": "Turkish", "price": 3, "seats": 110, "note": "Turkish and Mediterranean."},
    {"name": "Forn Lebnen", "district": "Chilonzor", "cuisine": "Lebanese", "price": 3, "seats": 65, "note": "Lebanese, Mediterranean."},
    {"name": "Kaarvan, the Indian Kitchen", "district": "Shayxontohur", "cuisine": "Indian", "price": 3, "seats": 90, "note": "Indian."},
    {"name": "Pro Khinkali", "district": "Yunusobod", "cuisine": "Georgian", "price": 2, "seats": 88, "note": "Georgian, khinkali and khachapuri."},
    {"name": "Pho Vietnam Noodle Bar", "district": "Mirobod", "cuisine": "Vietnamese", "price": 2, "seats": 58, "note": "Vietnamese, quick."},
]

SAMPLE_PROMPTS: list[dict[str, str]] = [
    {"language": "uz", "text": "Juma kuni kechqurun to'rt kishiga joy bor mi?"},
    {"language": "uz", "text": "Besh Qozonda o'n ikki kishiga stol kerak edi."},
    {"language": "ru", "text": "Здравствуйте, хочу столик на двоих в Сироварне завтра в восемь."},
    {"language": "en", "text": "Ember and Embar, eight o'clock tonight, table for two."},
    {"language": "en", "text": "Something Italian, not too pricey, tomorrow evening."},
    {"language": "en", "text": "Sette on Monday, party of six."},
]

OPENAI_MODEL_CHOICES = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]


def _restaurant_lines() -> str:
    return "\n".join(
        f"- {item['name']}, {item['district']}, price {item['price']}, {item['seats']} seats. {item['note']}"
        for item in RESTAURANTS
    )


def system_prompt(language: str) -> str:
    """The Dastro phone-host brief, adapted from realtime-prompt.md."""
    today = date.today()
    opening = {
        "uz": "Dastro, assalomu alaykum! Qaysi restoranga joy band qilmoqchisiz?",
        "ru": "Дастро, здравствуйте! В какой ресторан хотите забронировать столик?",
        "en": "Dastro, good evening! Which restaurant would you like to book?",
    }[language]
    return f"""# ROLE

You are the phone host for Dastro, a restaurant reservation service in Tashkent,
Uzbekistan. You answer calls from guests who want to book a table. You are not a
general assistant: you only handle reservations, changes, cancellations, and
questions about the restaurants listed below.

Today is {today:%A, %d %B %Y}. You can book up to 14 days ahead.

# HOW TO TALK

Every character you produce is spoken out loud by a text-to-speech voice. Never
output markdown, bullet points, asterisks, digits-as-symbols, or emoji.

- Warm, quick, professional. A good host, not a call-centre script.
- Short sentences. One question at a time. Never stack two questions.
- Keep every turn under 40 words, even when reading back a confirmation.
- Never read out more than three restaurants or three times at once.
- Read phone numbers back digit by digit, grouped.
- Say seating times with these exact words, never as digits or as minutes:
  Uzbek: 12:00 "kunduzi soat o'n ikkida", 14:00 "kunduzi soat ikkida",
  18:00 "kechqurun soat oltida", 20:00 "kechqurun soat sakkizda",
  22:00 "kechqurun soat o'nda".
  Russian: "в двенадцать дня", "в два дня", "в шесть вечера",
  "в восемь вечера", "в десять вечера".
  English: "at noon", "at two in the afternoon", "at six in the evening",
  "at eight in the evening", "at ten in the evening".
- Never mention that you are an AI or a model. If asked directly, say you are
  Dastro's booking assistant and move on.
- Do not apologise more than once for the same thing.

## Language

The guest is speaking {language}. Reply in that same language and follow any
switch the guest makes, every turn, without commenting on it. Your opening line
is: "{opening}"

# WHAT TO COLLECT

Every booking needs all five before you confirm:

1. Restaurant, from the list below.
2. Date. Accept "tomorrow", "Friday", "the 29th" and convert to a real date.
3. Time, one of the seatings offered for that restaurant.
4. Party size, 1 to 10.
5. Guest name and phone number, Uzbek mobile format: +998 then nine digits.

Ask about occasion, seating preference, allergies, or a high chair only if the
guest raises it or the party is six or more. Never ask for email. Never ask for
payment or card details.

# FLOW

Greet and ask which restaurant. If the guest describes what they want instead of
naming one, offer two or three matches. Then ask the date, then the party size,
then offer open seatings. Collect the name and phone number. Read the booking
back once and wait for an explicit yes. On yes, invent a code as DAS plus four
digits, say it slowly, and close warmly.

# RESTAURANTS

These twenty are the only restaurants that exist. Never invent one. If the guest
asks for somewhere else, say Dastro does not work with that restaurant yet and
offer the closest match by cuisine or district. Price is 1 (cheap) to 4
(expensive).

{_restaurant_lines()}

# AVAILABILITY

Treat these rules as the live book. Answer instantly and confidently. Never say
you are checking a system and never use the word database.

Standard seatings, every restaurant: lunch at 12:00 and 14:00, dinner at 18:00,
20:00, and 22:00. Bibigon Cafe and Pho Vietnam Noodle Bar run 12:00, 14:00, and
18:00 only.

Unless a rule below says otherwise, the seating is open. Say yes without
hesitating.

- Friday and Saturday at 20:00, only these have space: Besh Qozon, Khan Chapan,
  Khiva Restaurant, Pro Khinkali. Everywhere else, offer 18:00 or 22:00.
- Ember & Embar is fully booked at 20:00 every night; 18:00 and 22:00 are open.
- Sette Restaurant and Bar is closed Mondays.
- Syrovarnya has two tables left at 20:00 on any day, so a party of five or more
  does not fit that seating.
- Fillet Restaurant and Myasnoy Steak House do not serve lunch Monday to Friday.
- Afsona is closed for a private event on Sunday 30 August.

Party size: 1 to 4 is always fine. 5 to 8 is fine except at Pho Vietnam Noodle
Bar and Bibigon Cafe. 9 or 10 only at Besh Qozon, Khan Chapan, Khiva Restaurant,
Caravan, and Kosebasi. More than 10 cannot be booked on the phone: it needs the
restaurant's events team, so offer a callback and take the name and number.

Today's 12:00 and 14:00 seatings have passed; do not offer them today.

When nothing fits, never leave the guest with a flat no. Offer, in order: a
different time at the same restaurant, the same time on a nearby day, then a
similar restaurant that is open.

# GUARDRAILS

Never invent a restaurant, a dish, a price in so'm, or a menu item. Never quote a
bill total or promise a table number. Never take card details. If the guest is
angry, let them finish, apologise once, and solve it. If asked something
off-topic, say you only handle Dastro bookings. If you cannot understand after
two tries, offer a callback and take the name and number."""


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    language: str = "uz"
    model: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)


class AgentResponse(BaseModel):
    reply: str
    model: str
    latency_ms: float
    usage: dict[str, Any] | None = None


class TtsRequest(BaseModel):
    text: str
    language: str = "uz"
    voice_id: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


router = APIRouter(prefix="/voicelab", tags=["voicelab"])
_client: httpx.AsyncClient | None = None
_realtime_probe: dict[str, Any] = {"checked_at": 0.0, "available": False, "detail": "not probed"}


def _http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(VOICELAB_TIMEOUT_SECONDS, connect=10.0))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _require_voicelab_key() -> str:
    key = voicelab_api_key()
    if not key:
        raise HTTPException(status_code=503, detail="VOICELAB_API_KEY is not configured on this server")
    return key


def _voicelab_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _upstream_detail(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback
    message = payload.get("message") or fallback
    code = (payload.get("error") or {}).get("code")
    return f"{message} ({code})" if code else message


def _raise_upstream(response: httpx.Response, fallback: str) -> None:
    detail = _upstream_detail(response, fallback)
    logger.warning("voicelab_upstream_error status=%s detail=%s", response.status_code, detail)
    # 4xx caused by this server's key should not read as the browser's fault.
    status = response.status_code if response.status_code in {400, 402, 413, 415, 422, 429} else 502
    raise HTTPException(status_code=status, detail=detail)


async def _probe_realtime(key: str) -> dict[str, Any]:
    now = time.monotonic()
    if now - float(_realtime_probe["checked_at"]) < REALTIME_PROBE_TTL_SECONDS:
        return _realtime_probe
    try:
        response = await _http_client().post(
            f"{VOICELAB_BASE_URL}/v1/ticket",
            headers={**_voicelab_headers(key), "Content-Type": "application/json"},
            json={"transport": "websocket", "service": "stt"},
            timeout=15.0,
        )
    except httpx.HTTPError as error:
        _realtime_probe.update(checked_at=now, available=False, detail=f"probe failed: {error}")
        return _realtime_probe
    if response.status_code == 201:
        _realtime_probe.update(checked_at=now, available=True, detail="realtime STT is enabled")
    else:
        _realtime_probe.update(
            checked_at=now,
            available=False,
            detail=_upstream_detail(response, f"ticket endpoint returned {response.status_code}"),
        )
    return _realtime_probe


@router.get("/config")
async def config() -> dict[str, Any]:
    """Everything the page needs to render without holding a credential itself."""
    key = voicelab_api_key()
    payload: dict[str, Any] = {
        "voicelab_key_present": bool(key),
        "openai_key_present": bool(openai_api_key()),
        "openai_model": OPENAI_MODEL,
        "openai_models": OPENAI_MODEL_CHOICES,
        "languages": [
            {"code": "uz", "name": "Uzbek"},
            {"code": "ru", "name": "Russian"},
            {"code": "en", "name": "English"},
        ],
        "restaurants": RESTAURANTS,
        "sample_prompts": SAMPLE_PROMPTS,
        "realtime": {"available": False, "detail": "VOICELAB_API_KEY is not configured"},
    }
    if key:
        probe = await _probe_realtime(key)
        payload["realtime"] = {"available": probe["available"], "detail": probe["detail"]}
    return payload


@router.get("/voices")
async def voices(language: str = "uz") -> dict[str, Any]:
    key = _require_voicelab_key()
    try:
        response = await _http_client().get(
            f"{VOICELAB_BASE_URL}/v1/voices",
            headers=_voicelab_headers(key),
            params={"language": language},
            timeout=20.0,
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"VoiceLab voices request failed: {error}") from error
    if response.status_code != 200:
        _raise_upstream(response, "VoiceLab could not list voices")
    return {"data": response.json().get("data", [])}


@router.post("/ticket")
async def ticket(service: str = "stt") -> dict[str, Any]:
    """Mint a short-lived realtime ticket. The API key never leaves this server."""
    if service not in {"stt", "tts"}:
        raise HTTPException(status_code=422, detail="service must be stt or tts")
    key = _require_voicelab_key()
    try:
        response = await _http_client().post(
            f"{VOICELAB_BASE_URL}/v1/ticket",
            headers={**_voicelab_headers(key), "Content-Type": "application/json"},
            json={"transport": "websocket", "service": service},
            timeout=20.0,
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"VoiceLab ticket request failed: {error}") from error
    if response.status_code == 404:
        raise HTTPException(status_code=503, detail="Realtime VoiceLab is not enabled for this API key")
    if response.status_code != 201:
        _raise_upstream(response, "VoiceLab could not mint a realtime ticket")
    payload = response.json()
    return {
        "ticket": payload["ticket"],
        "websocket_url": payload["websocket_url"],
        "expires_at": payload.get("expires_at"),
        "service": payload.get("service", service),
    }


async def _poll_transcription(key: str, transcription_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + STT_POLL_TIMEOUT_SECONDS
    while True:
        response = await _http_client().get(
            f"{VOICELAB_BASE_URL}/v1/stt/transcriptions/{transcription_id}",
            headers=_voicelab_headers(key),
            timeout=30.0,
        )
        if response.status_code != 200:
            _raise_upstream(response, "VoiceLab could not read the transcription")
        payload = response.json()
        status = payload.get("status", "completed")
        if status == "completed":
            return payload
        if status in TERMINAL_FAILURE_STATES:
            detail = payload.get("error") or payload.get("message") or f"VoiceLab transcription {status}"
            raise HTTPException(status_code=502, detail=str(detail))
        # Anything else (queued, running, processing, ...) is still in flight.
        if time.monotonic() > deadline:
            raise HTTPException(status_code=504, detail="VoiceLab transcription did not finish in time")
        await asyncio.sleep(STT_POLL_SECONDS)


@router.post("/stt")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("uz"),
) -> dict[str, Any]:
    """Send one utterance to VoiceLab STT and wait for the finished transcript."""
    if language not in AGENT_LANGUAGES:
        raise HTTPException(status_code=422, detail="language must be uz, ru, or en")
    key = _require_voicelab_key()
    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The utterance is empty")
    started_at = time.perf_counter()
    try:
        response = await _http_client().post(
            f"{VOICELAB_BASE_URL}/v1/stt",
            headers={**_voicelab_headers(key), "Idempotency-Key": str(uuid.uuid4())},
            files={"audio": (audio.filename or "utterance.wav", payload, audio.content_type or "audio/wav")},
            data={"language": language, "include_speakers": "false"},
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"VoiceLab STT request failed: {error}") from error

    if response.status_code == 202:
        result = await _poll_transcription(key, response.json()["id"])
    elif response.status_code == 200:
        result = response.json()
    else:
        _raise_upstream(response, "VoiceLab could not transcribe the audio")
        return {}

    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    text = str(result.get("transcript") or result.get("text") or "").strip()
    logger.info(
        "voicelab_stt_complete language=%s bytes=%d latency_ms=%.1f characters=%d",
        language,
        len(payload),
        latency_ms,
        len(text),
    )
    return {
        "text": text,
        "language": result.get("language", language),
        "duration_ms": result.get("duration_ms"),
        "latency_ms": latency_ms,
        "engine": "voicelab",
    }


@router.post("/agent", response_model=AgentResponse)
async def agent(payload: AgentRequest) -> AgentResponse:
    """One reservation-host turn from OpenAI, on top of the Dastro system prompt."""
    if payload.language not in AGENT_LANGUAGES:
        raise HTTPException(status_code=422, detail="language must be uz, ru, or en")
    if not payload.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    key = (payload.api_key or "").strip() or openai_api_key()
    if not key:
        raise HTTPException(status_code=503, detail="No OpenAI key. Set OPENAI_API_KEY or paste a key on the page.")

    model = (payload.model or OPENAI_MODEL).strip()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(payload.language)},
            *[{"role": message.role, "content": message.content} for message in payload.messages],
        ],
        "temperature": payload.temperature,
        "max_tokens": 220,
    }
    started_at = time.perf_counter()
    try:
        response = await _http_client().post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {error}") from error

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", "OpenAI rejected the request")
        except ValueError:
            detail = "OpenAI rejected the request"
        logger.warning("openai_error status=%s detail=%s", response.status_code, detail)
        status = response.status_code if response.status_code in {400, 401, 403, 429} else 502
        raise HTTPException(status_code=status, detail=detail)

    result = response.json()
    reply = str(result["choices"][0]["message"].get("content") or "").strip()
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info("openai_turn_complete model=%s latency_ms=%.1f characters=%d", model, latency_ms, len(reply))
    return AgentResponse(
        reply=reply,
        model=result.get("model", model),
        latency_ms=latency_ms,
        usage=result.get("usage"),
    )


def _split_for_tts(text: str) -> list[str]:
    """VoiceLab caps one request at 1000 UTF-8 bytes, so split on sentences."""
    if len(text.encode("utf-8")) <= TTS_TEXT_BYTE_LIMIT:
        return [text]
    chunks: list[str] = []
    current = ""
    for piece in text.replace("!", "!\n").replace("?", "?\n").replace(".", ".\n").split("\n"):
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current} {piece}".strip()
        if len(candidate.encode("utf-8")) > TTS_TEXT_BYTE_LIMIT and current:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    # A single sentence can still be too long; cut it on a byte boundary.
    bounded: list[str] = []
    for chunk in chunks:
        encoded = chunk.encode("utf-8")
        while len(encoded) > TTS_TEXT_BYTE_LIMIT:
            bounded.append(encoded[:TTS_TEXT_BYTE_LIMIT].decode("utf-8", errors="ignore"))
            encoded = encoded[TTS_TEXT_BYTE_LIMIT:]
        remainder = encoded.decode("utf-8", errors="ignore").strip()
        if remainder:
            bounded.append(remainder)
    return bounded


def _concatenate_wav(parts: list[bytes]) -> bytes:
    if len(parts) == 1:
        return parts[0]
    frames: list[bytes] = []
    params = None
    for part in parts:
        with wave.open(io.BytesIO(part), "rb") as source:
            params = source.getparams()
            frames.append(source.readframes(source.getnframes()))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(params.nchannels)
        target.setsampwidth(params.sampwidth)
        target.setframerate(params.framerate)
        target.writeframes(b"".join(frames))
    return buffer.getvalue()


@router.post("/tts")
async def synthesize(payload: TtsRequest) -> Response:
    """Speak the agent's reply with VoiceLab and return one WAV to the browser."""
    if payload.language not in AGENT_LANGUAGES:
        raise HTTPException(status_code=422, detail="language must be uz, ru, or en")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    key = _require_voicelab_key()

    started_at = time.perf_counter()
    audio_parts: list[bytes] = []
    characters_used = 0
    for index, chunk in enumerate(_split_for_tts(text)):
        try:
            response = await _http_client().post(
                f"{VOICELAB_BASE_URL}/v1/tts",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Idempotency-Key": f"dastro-{uuid.uuid4().hex}-{index}",
                },
                json={
                    "text": chunk,
                    "language": payload.language,
                    "voice_id": payload.voice_id,
                    "speed": payload.speed,
                },
            )
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=f"VoiceLab TTS request failed: {error}") from error
        if response.status_code != 200:
            _raise_upstream(response, "VoiceLab could not generate speech")
        audio_parts.append(response.content)
        characters_used += int(response.headers.get("x-voicelab-characters-used", 0) or 0)

    audio = _concatenate_wav(audio_parts)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info(
        "voicelab_tts_complete language=%s voice=%s chunks=%d latency_ms=%.1f bytes=%d",
        payload.language,
        payload.voice_id,
        len(audio_parts),
        latency_ms,
        len(audio),
    )
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Pipeline-Latency-Ms": str(latency_ms),
            "X-VoiceLab-Characters-Used": str(characters_used),
            "Cache-Control": "no-store",
        },
    )
