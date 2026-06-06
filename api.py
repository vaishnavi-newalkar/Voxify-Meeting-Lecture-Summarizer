"""
api.py — FastAPI backend for Voxify
Wraps existing transcriber, summarizer, exporter utils into REST endpoints.
"""

import os
import io
import wave
import tempfile
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from utils.transcriber import transcribe_with_groq, transcribe_with_huggingface
from utils.summarizer import summarize_text, extract_action_items, identify_speakers
from utils.exporter import export_to_txt, export_to_pdf

app = FastAPI(
    title="Voxify API",
    description="FastAPI backend for Voxify. Handles audio transcription, meeting summarization, and exporting reports.",
    version="1.0.0",
    contact={
        "name": "Voxify Engineering",
        "url": "https://github.com/OWNER/REPO",
    }
)

# CORS — origins from env var (comma-separated), fallback to wildcard for dev
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in _cors_origins.split(",")] if _cors_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Audio Preprocessor ────────────────────────────────────────────────────────
def preprocess_wav_bytes(raw_bytes: bytes, target_rate: int = 16000) -> bytes:
    """Reads raw WAV bytes, converts to mono 16kHz 16-bit PCM WAV."""
    with wave.open(io.BytesIO(raw_bytes)) as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames   = wf.getnframes()
        raw_pcm    = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(samp_width, np.int16)
    samples = np.frombuffer(raw_pcm, dtype=dtype).astype(np.float32)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if frame_rate != target_rate:
        duration   = len(samples) / frame_rate
        new_length = int(duration * target_rate)
        samples    = np.interp(
            np.linspace(0, len(samples) - 1, new_length),
            np.arange(len(samples)),
            samples,
        )

    max_val = np.max(np.abs(samples)) or 1.0
    samples = (samples / max_val * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(target_rate)
        wf_out.writeframes(samples.tobytes())
    return buf.getvalue()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health Check")
@app.get("/api/health", summary="API Health Check Alias")
def health():
    """
    Check the health of the API server.
    
    Returns a simple JSON response indicating the service is up and running.
    """
    return {"status": "ok", "version": "1.0.0"}


# ── Transcribe ────────────────────────────────────────────────────────────────
@app.post("/api/transcribe", summary="Transcribe Audio")
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Form("groq"),
    api_key: str = Form(...),
):
    """
    Transcribe an uploaded audio file using Groq or HuggingFace APIs.
    
    - **file**: The audio file to transcribe (max 25MB).
    - **engine**: "groq" or "huggingface" (defaults to "groq").
    - **api_key**: The API key for the selected engine.
    """
    raw_bytes = await file.read()

    # ── File size guard (Groq limit = 25 MB) ─────────────────────────────
    MAX_SIZE = 25 * 1024 * 1024  # 25 MB
    if len(raw_bytes) > MAX_SIZE:
        size_mb = len(raw_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Groq's Whisper API only accepts files under 25 MB. Please upload a smaller file.",
        )

    # Determine file extension
    ext = Path(file.filename or "audio.wav").suffix.lower()

    # If WAV from recording, preprocess
    if ext == ".wav":
        try:
            raw_bytes = preprocess_wav_bytes(raw_bytes)
        except Exception:
            pass  # Use original bytes if preprocessing fails

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        if engine == "groq":
            text, language, duration, segments = transcribe_with_groq(tmp_path, api_key)
        else:
            text, language, duration, segments = transcribe_with_huggingface(tmp_path, api_key)

        if not text:
            raise HTTPException(status_code=400, detail="Transcription returned empty.")

        return {
            "transcript": text,
            "language": language,
            "duration": duration,
            "segments": segments,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Summarize ─────────────────────────────────────────────────────────────────
@app.post("/api/summarize", summary="Summarize Transcript")
async def summarize(
    transcript: str = Form(...),
    length_option: str = Form("Standard (5–8 points)"),
    model: str = Form("llama-3.3-70b-versatile"),
    api_key: str = Form(...),
):
    """
    Generate a summary and extract action items from a transcript.
    
    - **transcript**: The full text transcript to process.
    - **length_option**: Desired summary format/length.
    - **model**: The LLM model to use (e.g., llama-3.3-70b-versatile).
    - **api_key**: The Groq API key to use for LLM inference.
    """
    try:
        summary = summarize_text(transcript, length_option, model, api_key)
        action_items = extract_action_items(transcript, model, api_key)
        return {
            "summary": summary,
            "action_items": action_items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Speaker Diarization ──────────────────────────────────────────────────────
@app.post("/api/speakers", summary="Identify Speakers")
async def speakers(
    transcript: str = Form(...),
    model: str = Form("llama-3.3-70b-versatile"),
    api_key: str = Form(...),
):
    """
    Analyze a transcript and identify different speakers.
    
    - **transcript**: The full text transcript.
    - **model**: The LLM model to use.
    - **api_key**: The Groq API key to use for LLM inference.
    """
    try:
        result = identify_speakers(transcript, model, api_key)
        return {"speakers": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Export TXT ────────────────────────────────────────────────────────────────
@app.post("/api/export/txt", summary="Export TXT Report")
async def export_txt(
    transcript: str = Form(""),
    summary: str = Form(""),
    action_items: str = Form("[]"),
):
    """
    Generate a downloadable TXT report.
    
    - **transcript**: The full meeting transcript.
    - **summary**: The generated meeting summary.
    - **action_items**: JSON array string of action items.
    """
    items = json.loads(action_items) if action_items else []
    data = export_to_txt(transcript, summary, items)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=meeting_report.txt"},
    )


# ── Export PDF ────────────────────────────────────────────────────────────────
@app.post("/api/export/pdf", summary="Export PDF Report")
async def export_pdf(
    transcript: str = Form(""),
    summary: str = Form(""),
    action_items: str = Form("[]"),
):
    """
    Generate a downloadable PDF report.
    
    - **transcript**: The full meeting transcript.
    - **summary**: The generated meeting summary.
    - **action_items**: JSON array string of action items.
    """
    items = json.loads(action_items) if action_items else []
    try:
        data = export_to_pdf(transcript, summary, items)
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=meeting_report.pdf"},
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="fpdf2 not installed. Run: pip install fpdf2")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
