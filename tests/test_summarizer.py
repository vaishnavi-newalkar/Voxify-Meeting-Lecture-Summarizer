"""
test_summarizer.py — Unit tests for utils/summarizer.py.

All Groq API calls are mocked via unittest.mock.patch on requests.post.
No real HTTP requests are made.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestSummarizeText:

    @patch("utils.summarizer.requests.post")
    def test_summarize_returns_markdown(self, mock_post):
        """summarize_text returns the LLM response as-is for valid input.

        Proves the function sends the transcript to Groq's chat API and
        returns the content from choices[0].message.content.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "## Summary\n- Key point one"}}]
        }
        mock_post.return_value = mock_resp

        from utils.summarizer import summarize_text

        result = summarize_text(
            "We discussed shipping in Q3.",
            "Brief (3–5 points)",
            "llama-3.3-70b-versatile",
            "gsk_fake_key",
        )

        assert "Summary" in result
        assert "Key point" in result
        mock_post.assert_called_once()

    def test_summarize_empty_transcript(self):
        """summarize_text returns 'No transcript provided.' for empty input.

        Proves the early-return guard prevents unnecessary API calls
        when the transcript is blank or whitespace-only.
        """
        from utils.summarizer import summarize_text

        result = summarize_text("   ", "Brief (3–5 points)", "model", "key")
        assert result == "No transcript provided."

    @patch("utils.summarizer.requests.post")
    def test_summarize_api_error_raises(self, mock_post):
        """summarize_text raises RuntimeError when Groq returns non-200.

        Proves error propagation so callers (the API endpoint) can
        convert it to an HTTP 500 response.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API key"
        mock_post.return_value = mock_resp

        from utils.summarizer import summarize_text

        with pytest.raises(RuntimeError, match="Groq LLM error 401"):
            summarize_text("Some text.", "Standard (5–8 points)", "model", "bad_key")


class TestExtractActionItems:

    @patch("utils.summarizer.requests.post")
    def test_extract_returns_list(self, mock_post):
        """extract_action_items returns a list of strings from valid JSON response.

        Proves the function correctly parses a JSON array returned by
        the LLM and returns it as a Python list.
        """
        items = ["Send report by Friday", "Schedule follow-up"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(items)}}]
        }
        mock_post.return_value = mock_resp

        from utils.summarizer import extract_action_items

        result = extract_action_items("We need to send a report.", "model", "key")
        assert isinstance(result, list)
        assert len(result) == 2
        assert "report" in result[0].lower()

    def test_extract_empty_transcript(self):
        """extract_action_items returns empty list for empty transcript.

        Proves the guard clause avoids an API call when there's nothing
        to extract from.
        """
        from utils.summarizer import extract_action_items

        result = extract_action_items("", "model", "key")
        assert result == []

    @patch("utils.summarizer.requests.post")
    def test_extract_malformed_json_returns_empty(self, mock_post):
        """extract_action_items returns [] when LLM returns non-JSON text.

        Proves graceful degradation — the function catches JSON parse
        errors and returns an empty list instead of crashing.
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Here are some tasks: blah blah"}}]
        }
        mock_post.return_value = mock_resp

        from utils.summarizer import extract_action_items

        result = extract_action_items("Some meeting text.", "model", "key")
        assert result == []
