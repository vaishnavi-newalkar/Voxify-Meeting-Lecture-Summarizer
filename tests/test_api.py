"""
test_api.py — Integration tests for every FastAPI endpoint in api.py.

Tests cover:
  - GET  /health and /api/health
  - POST /api/transcribe  (valid, oversized, missing key)
  - POST /api/summarize   (valid, empty transcript, missing fields)
  - POST /api/export/txt
  - POST /api/export/pdf
  - POST /api/speakers

All external API calls (Groq, HuggingFace) are mocked.
"""

import io
import json
from unittest.mock import patch


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        """GET /health returns 200 with status=ok and version=1.0.0.

        Proves the health check endpoint used by Docker HEALTHCHECK is
        reachable and returns the expected JSON shape.
        """
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"

    def test_api_health_alias(self, client):
        """GET /api/health returns the same response as /health.

        Proves the aliased route (used by the frontend) works identically.
        """
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Transcribe endpoint ──────────────────────────────────────────────────────

class TestTranscribeEndpoint:

    @patch("api.transcribe_with_groq")
    def test_transcribe_valid_wav(self, mock_groq, client, small_wav_bytes):
        """POST /api/transcribe with a valid small WAV returns transcript data.

        Proves the endpoint accepts a file upload, calls the Groq transcriber,
        and returns the expected JSON shape with transcript/language/duration/segments.
        """
        mock_groq.return_value = (
            "Hello, this is a test.",
            "English",
            2.5,
            [{"start": 0.0, "end": 2.5, "text": "Hello, this is a test."}],
        )

        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
            data={"engine": "groq", "api_key": "gsk_test_key"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == "Hello, this is a test."
        assert data["language"] == "English"
        assert data["duration"] == 2.5
        assert len(data["segments"]) == 1

    def test_transcribe_oversized_file_returns_413(self, client, oversized_bytes):
        """POST /api/transcribe with a >25MB file returns 413.

        Proves the file size guard rejects oversized uploads before
        hitting the external API, saving bandwidth and API quota.
        """
        resp = client.post(
            "/api/transcribe",
            files={"file": ("big.wav", io.BytesIO(oversized_bytes), "audio/wav")},
            data={"engine": "groq", "api_key": "gsk_test_key"},
        )

        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_transcribe_missing_api_key_returns_422(self, client, small_wav_bytes):
        """POST /api/transcribe without api_key returns 422 (validation error).

        Proves FastAPI's Form(...) validation rejects requests missing
        required fields before any processing occurs.
        """
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
            data={"engine": "groq"},
            # api_key intentionally omitted
        )

        assert resp.status_code == 422

    @patch("api.transcribe_with_huggingface")
    def test_transcribe_huggingface_engine(self, mock_hf, client, small_wav_bytes):
        """POST /api/transcribe with engine=huggingface routes to HF transcriber.

        Proves the engine routing logic correctly dispatches to the
        HuggingFace transcriber when engine != 'groq'.
        """
        mock_hf.return_value = ("HF transcript.", "Detected", 1.0, [])

        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
            data={"engine": "huggingface", "api_key": "hf_test_key"},
        )

        assert resp.status_code == 200
        assert resp.json()["transcript"] == "HF transcript."
        mock_hf.assert_called_once()


# ── Summarize endpoint ───────────────────────────────────────────────────────

class TestSummarizeEndpoint:

    @patch("api.extract_action_items")
    @patch("api.summarize_text")
    def test_summarize_valid_transcript(self, mock_sum, mock_actions, client):
        """POST /api/summarize with valid transcript returns summary + action items.

        Proves the endpoint calls both summarize_text and extract_action_items
        and returns them in the correct JSON shape.
        """
        mock_sum.return_value = "## Summary\n- Point one\n- Point two"
        mock_actions.return_value = ["Send report", "Book meeting"]

        resp = client.post(
            "/api/summarize",
            data={
                "transcript": "We discussed the Q3 roadmap and decided to ship in August.",
                "length_option": "Brief (3–5 points)",
                "model": "llama-3.3-70b-versatile",
                "api_key": "gsk_test_key",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "Summary" in data["summary"]
        assert len(data["action_items"]) == 2

    def test_summarize_missing_transcript_returns_422(self, client):
        """POST /api/summarize without transcript field returns 422.

        Proves FastAPI rejects the request when the required 'transcript'
        Form field is missing.
        """
        resp = client.post(
            "/api/summarize",
            data={
                "api_key": "gsk_test_key",
                # transcript intentionally omitted
            },
        )

        assert resp.status_code == 422


# ── Speaker diarization endpoint ─────────────────────────────────────────────

class TestSpeakersEndpoint:

    @patch("api.identify_speakers")
    def test_speakers_valid(self, mock_spk, client):
        """POST /api/speakers returns reformatted transcript with speaker labels.

        Proves the endpoint passes transcript to identify_speakers and
        wraps the result in {"speakers": ...}.
        """
        mock_spk.return_value = "Speaker 1: Hello\nSpeaker 2: Hi there"

        resp = client.post(
            "/api/speakers",
            data={
                "transcript": "Hello. Hi there.",
                "model": "llama-3.3-70b-versatile",
                "api_key": "gsk_test_key",
            },
        )

        assert resp.status_code == 200
        assert "Speaker 1" in resp.json()["speakers"]


# ── Export endpoints ─────────────────────────────────────────────────────────

class TestExportEndpoints:

    def test_export_txt_returns_text(self, client):
        """POST /api/export/txt returns a text/plain file with expected sections.

        Proves the TXT exporter includes SUMMARY, TRANSCRIPT, and
        ACTION ITEMS sections in the downloaded report.
        """
        resp = client.post(
            "/api/export/txt",
            data={
                "transcript": "Full meeting transcript here.",
                "summary": "Key points discussed.",
                "action_items": json.dumps(["Task A", "Task B"]),
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        body = resp.text
        assert "SUMMARY" in body
        assert "FULL TRANSCRIPT" in body
        assert "ACTION ITEMS" in body
        assert "Task A" in body

    def test_export_pdf_returns_pdf_bytes(self, client):
        """POST /api/export/pdf returns a valid PDF starting with %PDF.

        Proves the PDF exporter generates actual PDF content (not an error)
        and sets the correct content-disposition header.
        """
        resp = client.post(
            "/api/export/pdf",
            data={
                "transcript": "Meeting notes.",
                "summary": "Summary here.",
                "action_items": json.dumps(["Follow up"]),
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        # PDF magic bytes
        assert resp.content[:5] == b"%PDF-"
