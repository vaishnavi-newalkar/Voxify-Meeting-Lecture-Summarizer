"""
conftest.py — Shared pytest fixtures for Voxify backend tests.

Provides:
  - FastAPI TestClient via httpx
  - Mock GROQ_API_KEY environment variable
  - Helper to generate minimal valid WAV bytes
"""

import io
import os
import wave
import struct
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_groq_env(monkeypatch):
    """Ensure GROQ_API_KEY is always set so CORS/env logic doesn't break."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_1234567890")
    monkeypatch.setenv("HF_API_KEY", "hf_test_fake_key_1234567890")


@pytest.fixture
def client():
    """Create a FastAPI TestClient for integration tests."""
    from api import app
    return TestClient(app)


@pytest.fixture
def small_wav_bytes():
    """Generate a minimal valid 16-bit mono WAV file (0.1 seconds of silence).

    This is used to test endpoints that expect audio file uploads without
    needing a real audio file on disk.
    """
    sample_rate = 16000
    duration = 0.1  # seconds
    n_samples = int(sample_rate * duration)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


@pytest.fixture
def oversized_bytes():
    """Generate bytes that exceed the 25 MB Groq upload limit.

    Returns 26 MB of zeros — enough to trigger the 413 guard in
    the /api/transcribe endpoint.
    """
    return b"\x00" * (26 * 1024 * 1024)
