import { X, CheckCircle } from "lucide-react";
import { LLM_OPTIONS, LENGTH_OPTIONS } from "../constants";

export default function SettingsDrawer({ open, onClose, settings, onChange }) {
  const set = (key, val) => onChange({ ...settings, [key]: val });

  return (
    <>
      <div
        className={`settings-overlay ${open ? "open" : ""}`}
        onClick={onClose}
      />
      <aside className={`settings-drawer ${open ? "open" : ""}`}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="settings-close" onClick={onClose} aria-label="Close settings">
            <X size={18} />
          </button>
        </div>

        {/* STT Engine */}
        <div className="settings-section">
          <div className="settings-label">🔊 Speech-to-Text Engine</div>
          <select
            className="settings-select"
            value={settings.engine}
            onChange={(e) => set("engine", e.target.value)}
          >
            <option value="groq">Groq (Whisper Large v3 Turbo) — Recommended</option>
            <option value="huggingface">HuggingFace (Whisper Large v3)</option>
          </select>
        </div>

        {/* API Key */}
        <div className="settings-section">
          <div className="settings-label">
            🔑 {settings.engine === "groq" ? "Groq" : "HuggingFace"} API Key
          </div>
          <input
            className="settings-input"
            type="password"
            placeholder={settings.engine === "groq" ? "gsk_..." : "hf_..."}
            value={settings.apiKey}
            onChange={(e) => set("apiKey", e.target.value)}
          />
          {settings.apiKey && (
            <div className="key-saved">
              <CheckCircle size={14} /> Key saved
            </div>
          )}
        </div>

        {/* LLM Model */}
        <div className="settings-section">
          <div className="settings-label">🤖 LLM for Summarization</div>
          <select
            className="settings-select"
            value={settings.llmModel}
            onChange={(e) => set("llmModel", e.target.value)}
          >
            {LLM_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Summary Length */}
        <div className="settings-section">
          <div className="settings-label">📝 Summary Depth</div>
          <select
            className="settings-select"
            value={settings.lengthOption}
            onChange={(e) => set("lengthOption", e.target.value)}
          >
            {LENGTH_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* Toggles */}
        <div className="settings-section">
          <div className="settings-label">🔬 Extra Analysis</div>

          <div className="settings-toggle">
            <span className="settings-toggle-label">Speaker Diarization</span>
            <button
              className={`toggle-switch ${settings.doSpeakers ? "active" : ""}`}
              onClick={() => set("doSpeakers", !settings.doSpeakers)}
              aria-label="Toggle speaker diarization"
            >
              <span className="toggle-knob" />
            </button>
          </div>

          <div className="settings-toggle">
            <span className="settings-toggle-label">Show Timestamps</span>
            <button
              className={`toggle-switch ${settings.doTimestamps ? "active" : ""}`}
              onClick={() => set("doTimestamps", !settings.doTimestamps)}
              aria-label="Toggle timestamps"
            >
              <span className="toggle-knob" />
            </button>
          </div>
        </div>

        <div style={{ marginTop: "auto", paddingTop: "1rem", borderTop: "1px solid var(--border)" }}>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Voxify v2.0 — Groq + HuggingFace<br />
            Open-source speech intelligence
          </p>
        </div>
      </aside>
    </>
  );
}
