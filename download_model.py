from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_ID = os.getenv("MODEL_ID", "OvozifyLabs/whisper-small-uz-v1")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "whisper-small-uz-v1")))
REVISION = os.getenv("MODEL_REVISION", "main")


def main() -> None:
    if (MODEL_PATH / "config.json").is_file():
        print(f"Model already exists at {MODEL_PATH}")
        return
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {MODEL_PATH} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(MODEL_PATH),
        token=os.getenv("HF_TOKEN") or None,
    )
    print("Model download complete")


if __name__ == "__main__":
    main()
