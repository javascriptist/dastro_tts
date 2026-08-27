FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/data/models/whisper-small-uz-v1 \
    HF_HOME=/data/huggingface \
    PORT=8090

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py voicelab.py download_model.py .env.example ./
COPY static ./static
RUN mkdir -p /data/models/whisper-small-uz-v1 /data/huggingface

EXPOSE 8090
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8090} --log-level ${LOG_LEVEL:-info}"]
