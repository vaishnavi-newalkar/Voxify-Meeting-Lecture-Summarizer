import { FileText, Download, FileDown, CheckSquare, Users, Clock } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { exportTxt, exportPdf } from "../api";

export default function ResultsPanel({ results, settings }) {
  const { transcript, summary, actionItems, language, duration, segments, speakers } = results;
  const words = transcript ? transcript.split(/\s+/).length : 0;

  const handleExport = async (type) => {
    try {
      let blob;
      if (type === "txt") {
        blob = await exportTxt(transcript, summary, actionItems || []);
      } else {
        blob = await exportPdf(transcript, summary, actionItems || []);
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = type === "txt" ? "meeting_report.txt" : "meeting_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert("Export error: " + err.message);
    }
  };

  const downloadText = (text, filename) => {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="results-section" id="results">
      <div className="results-header">
        <h2>Analysis Results</h2>
      </div>

      {/* Metrics */}
      <div className="results-metrics">
        <div className="metric-chip">
          🌐 <strong>{language || "—"}</strong>
        </div>
        <div className="metric-chip">
          📝 <strong>{words.toLocaleString()}</strong> words
        </div>
        <div className="metric-chip">
          ⏱️ <strong>{duration ? `${duration.toFixed(1)}s` : "—"}</strong>
        </div>
        <div className="metric-chip">
          📊 <strong>{transcript ? transcript.length.toLocaleString() : 0}</strong> chars
        </div>
        {actionItems && actionItems.length > 0 && (
          <div className="metric-chip">
            ✅ <strong>{actionItems.length}</strong> action items
          </div>
        )}
      </div>

      {/* Transcript + Summary */}
      <div className="results-grid">
        <div className="result-card">
          <div className="result-card-header">
            <FileText size={14} /> Full Transcript
          </div>
          <div className="result-card-body">
            {settings.doTimestamps && segments && segments.length > 0 ? (
              segments.map((seg, i) => {
                const ms = Math.floor(seg.start / 60);
                const ss = Math.floor(seg.start % 60);
                const me = Math.floor(seg.end / 60);
                const se = Math.floor(seg.end % 60);
                const ts = `${String(ms).padStart(2, "0")}:${String(ss).padStart(2, "0")} → ${String(me).padStart(2, "0")}:${String(se).padStart(2, "0")}`;
                return (
                  <div key={i} className="timestamp-seg">
                    <span className="timestamp-tag">{ts}</span>
                    <span>{seg.text}</span>
                  </div>
                );
              })
            ) : (
              <p style={{ whiteSpace: "pre-wrap" }}>{transcript}</p>
            )}
          </div>
        </div>

        <div className="result-card">
          <div className="result-card-header">
            🤖 AI Summary
          </div>
          <div className="result-card-body">
            {summary ? <ReactMarkdown>{summary}</ReactMarkdown> : <p>No summary generated.</p>}
          </div>
        </div>
      </div>

      {/* Speakers */}
      {speakers && (
        <div className="result-card" style={{ marginBottom: "1.5rem" }}>
          <div className="result-card-header">
            <Users size={14} /> Speaker Diarization
          </div>
          <div className="result-card-body">
            {speakers.split("\n").filter(Boolean).map((line, i) => {
              if (line.trim().startsWith("Speaker")) {
                const parts = line.split(":", 2);
                return (
                  <div key={i} className="speaker-block">
                    <div className="speaker-label-tag">{parts[0]}</div>
                    {parts[1] || ""}
                  </div>
                );
              }
              return <div key={i} className="speaker-block">{line}</div>;
            })}
          </div>
        </div>
      )}

      {/* Action Items */}
      {actionItems && actionItems.length > 0 && (
        <div className="result-card action-items-section" style={{ marginBottom: "1.5rem" }}>
          <div className="result-card-header">
            <CheckSquare size={14} /> Action Items
          </div>
          <div className="result-card-body">
            {actionItems.map((item, i) => (
              <div key={i} className="action-item">
                <CheckSquare size={16} className="action-item-check" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export */}
      <div className="export-row">
        <button className="export-btn" onClick={() => handleExport("txt")}>
          <FileDown size={16} /> Download TXT
        </button>
        <button className="export-btn" onClick={() => handleExport("pdf")}>
          <Download size={16} /> Download PDF
        </button>
        <button className="export-btn" onClick={() => downloadText(transcript, "transcript.txt")}>
          <FileText size={16} /> Transcript Only
        </button>
        {actionItems && actionItems.length > 0 && (
          <button
            className="export-btn"
            onClick={() =>
              downloadText(actionItems.map((a, i) => `[ ] ${a}`).join("\n"), "action_items.txt")
            }
          >
            <CheckSquare size={16} /> Action Items
          </button>
        )}
      </div>
    </section>
  );
}
