"""
test_exporter.py — Unit tests for utils/exporter.py.
"""

from utils.exporter import export_to_txt, export_to_pdf

class TestExporter:

    def test_export_txt_includes_sections(self):
        """export_to_txt returns bytes containing transcript, summary, and action items.

        Proves that the plain-text exporter formats the provided inputs into
        distinct sections correctly.
        """
        transcript = "This is the transcript."
        summary = "This is the summary."
        action_items = ["Item 1", "Item 2"]

        result = export_to_txt(transcript, summary, action_items)
        
        assert isinstance(result, bytes)
        decoded = result.decode("utf-8")
        assert "SUMMARY" in decoded
        assert "This is the summary." in decoded
        assert "ACTION ITEMS" in decoded
        assert "Item 1" in decoded
        assert "FULL TRANSCRIPT" in decoded
        assert "This is the transcript." in decoded

    def test_export_txt_handles_empty_inputs(self):
        """export_to_txt gracefully handles missing summary or action items.

        Proves that the exporter doesn't crash when optional data is missing,
        and outputs placeholder text instead.
        """
        result = export_to_txt("", "", [])
        
        assert isinstance(result, bytes)
        decoded = result.decode("utf-8")
        assert "No summary available." in decoded
        assert "No transcript available." in decoded
        assert "ACTION ITEMS" not in decoded # Should not be present if empty

    def test_export_pdf_returns_pdf_bytes(self):
        """export_to_pdf generates valid PDF bytes.

        Proves that the fpdf2 library correctly generates a PDF file
        starting with the expected magic bytes.
        """
        transcript = "Transcript text."
        summary = "Summary text."
        action_items = ["Task 1"]

        result = export_to_pdf(transcript, summary, action_items)

        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF-") # PDF magic bytes
        # Just check it's non-empty and has EOF marker
        assert len(result) > 100
        assert b"%%EOF" in result

    def test_export_pdf_handles_special_characters(self):
        """export_to_pdf safely encodes problematic characters.

        Proves that the latin-1 encoding fallback works for characters
        that might otherwise cause fpdf2 to crash.
        """
        transcript = "Café & naïve users: ✨ emoji test."
        
        # Should not raise UnicodeEncodeError
        result = export_to_pdf(transcript, "Summary", ["Action 1"])
        
        assert isinstance(result, bytes)
        assert result.startswith(b"%PDF-")
