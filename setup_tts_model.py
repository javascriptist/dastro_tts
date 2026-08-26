from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "aisha-org/navoiy-tts")
TTS_MODEL_PATH = Path(os.getenv("TTS_MODEL_PATH", str(BASE_DIR / "models" / "navoiy-tts")))
TTS_COSYVOICE_DIR = Path(os.getenv("TTS_COSYVOICE_DIR", str(BASE_DIR / "models" / "CosyVoice")))
TTS_COSYVOICE_REPO = os.getenv("TTS_COSYVOICE_REPO", "https://github.com/FunAudioLLM/CosyVoice.git")
TTS_BASE_MODEL_ID = os.getenv("TTS_BASE_MODEL_ID", "FunAudioLLM/CosyVoice2-0.5B")
TTS_BASE_MODEL_DIR = Path(
    os.getenv("TTS_BASE_MODEL_DIR", str(TTS_COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"))
)
REVISION = os.getenv("TTS_MODEL_REVISION", "main")


def _clone_cosyvoice() -> None:
    if (TTS_COSYVOICE_DIR / ".git").is_dir():
        print(f"CosyVoice already checked out at {TTS_COSYVOICE_DIR}")
        return
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone the CosyVoice engine. Install git and re-run.")
    TTS_COSYVOICE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {TTS_COSYVOICE_REPO} into {TTS_COSYVOICE_DIR} ...")
    subprocess.run(
        ["git", "clone", "--recursive", TTS_COSYVOICE_REPO, str(TTS_COSYVOICE_DIR)],
        check=True,
    )


def _download_base_model() -> None:
    if TTS_BASE_MODEL_DIR.is_dir() and any(TTS_BASE_MODEL_DIR.iterdir()):
        print(f"{TTS_BASE_MODEL_ID} already exists at {TTS_BASE_MODEL_DIR}")
        return
    TTS_BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TTS_BASE_MODEL_ID} to {TTS_BASE_MODEL_DIR} ...")
    snapshot_download(
        repo_id=TTS_BASE_MODEL_ID,
        local_dir=str(TTS_BASE_MODEL_DIR),
        token=os.getenv("HF_TOKEN") or None,
    )


def _download_navoiy_tts() -> None:
    if (TTS_MODEL_PATH / "inference.py").is_file():
        print(f"Navoiy TTS assets already exist at {TTS_MODEL_PATH}")
        return
    TTS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {TTS_MODEL_ID} to {TTS_MODEL_PATH} ...")
    snapshot_download(
        repo_id=TTS_MODEL_ID,
        revision=REVISION,
        local_dir=str(TTS_MODEL_PATH),
        token=os.getenv("HF_TOKEN") or None,
    )


def main() -> None:
    _clone_cosyvoice()
    _download_base_model()
    _download_navoiy_tts()
    print(
        "Navoiy TTS setup complete. Install CosyVoice's own dependencies "
        f"(pip install -r {TTS_COSYVOICE_DIR}/requirements.txt), then set TTS_ENABLED=true.\n"
        "This script cannot verify the upstream inference.py CLI flags from this environment; "
        "confirm them against the files just downloaded (see README's Text-to-speech section) "
        "before relying on /synthesize in production."
    )


if __name__ == "__main__":
    main()
