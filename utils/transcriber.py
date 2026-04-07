"""
transcriber.py — Speech-to-Text via Groq & HuggingFace (no OpenAI)

Groq:        Uses whisper-large-v3-turbo via Groq's ultra-fast inference API
HuggingFace: Uses openai/whisper-large-v3 via HF Inference API
"""

import os
import json
import requests
from pathlib import Path


# ── Groq Whisper ──────────────────────────────────────────────────────────────
def transcribe_with_groq(
    file_path: str,
    api_key: str,
    model: str = "whisper-large-v3-turbo",
    response_format: str = "verbose_json"
) -> tuple[str, str, float, list[dict]]:
    """
    Transcribe audio using Groq's Whisper API.

    Models available on Groq:
      - whisper-large-v3-turbo  (fastest, recommended)
      - whisper-large-v3        (most accurate)

    Returns:
        (transcript, language, duration, segments)
    """
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    file_size = Path(file_path).stat().st_size
    if file_size > 25 * 1024 * 1024:
        raise ValueError("File exceeds Groq's 25 MB limit. Please compress the audio.")

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (Path(file_path).name, audio_file, _get_mime(file_path)),
        }
        data = {
            "model": model,
            "response_format": response_format,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    result    = resp.json()
    text      = result.get("text", "").strip()
    language  = result.get("language", "unknown").capitalize()
    duration  = result.get("duration", 0.0)
    segments  = [
        {
            "start": round(s.get("start", 0), 2),
            "end":   round(s.get("end", 0), 2),
            "text":  s.get("text", "").strip()
        }
        for s in result.get("segments", [])
    ]

    return text, language, duration, segments


# ── HuggingFace Whisper ───────────────────────────────────────────────────────
def transcribe_with_huggingface(
    file_path: str,
    api_key: str,
    model: str = "openai/whisper-large-v3"
) -> tuple[str, str, float, list[dict]]:
    """
    Transcribe using HuggingFace Inference API — openai/whisper-large-v3.

    Note: HF Inference API returns plain text; no timestamps or language
    detection in the free tier. Duration is estimated from file size.

    Returns:
        (transcript, language, duration, segments)
    """
    url     = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        audio_data = f.read()

    # HF Inference API — send raw bytes
    resp = requests.post(
        url,
        headers={**headers, "Content-Type": "audio/wav"},
        data=audio_data,
        timeout=180
    )

    if resp.status_code == 503:
        raise RuntimeError("HuggingFace model is loading. Wait ~30s and retry.")
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text}")

    result = resp.json()

    # HF returns {"text": "..."} for Whisper
    if isinstance(result, dict):
        text = result.get("text", "").strip()
    elif isinstance(result, list) and result:
        text = result[0].get("generated_text", result[0].get("text", "")).strip()
    else:
        text = str(result).strip()

    # Estimate duration from file size (rough: ~1 MB ≈ 60s for WAV)
    file_mb  = Path(file_path).stat().st_size / (1024 * 1024)
    duration = round(file_mb * 60, 1)

    return text, "Detected", duration, []


# ── Utility ───────────────────────────────────────────────────────────────────
def _get_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".mp3":  "audio/mpeg",
        ".mp4":  "audio/mp4",
        ".wav":  "audio/wav",
        ".m4a":  "audio/mp4",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(ext, "audio/wav")
