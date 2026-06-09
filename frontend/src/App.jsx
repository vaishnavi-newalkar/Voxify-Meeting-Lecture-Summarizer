import { useState, useRef, useCallback } from "react";
import {
  Mic, Upload, Settings, ArrowRight, Sparkles,
  ListChecks, Users, FileText, Square, AlertTriangle
} from "lucide-react";
import SettingsDrawer from "./components/SettingsDrawer";
import ResultsPanel from "./components/ResultsPanel";
import { transcribeAudio, summarizeTranscript, identifySpeakers } from "./api";
import { SAMPLE_TRANSCRIPT } from "./constants";

const INITIAL_SETTINGS = {
  engine: "groq",
  apiKey: "",
  llmModel: "llama-3.3-70b-versatile",
  lengthOption: "Standard (5–8 points)",
  doSpeakers: false,
  doTimestamps: false,
};

const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB — Groq API limit

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState(INITIAL_SETTINGS);

  // Transcript input
  const [transcript, setTranscript] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  // Recording
  const [recording, setRecording] = useState(false);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);

  // Processing
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");

  // Results
  const [results, setResults] = useState(null);

  // ── Word count ──────────────────────────────────────────────────────
  const wordCount = transcript.trim() ? transcript.trim().split(/\s+/).length : 0;

  // ── File size check ─────────────────────────────────────────────────
  const fileTooLarge = audioFile && audioFile.size > MAX_FILE_SIZE;

  // ── File upload ─────────────────────────────────────────────────────
  const fileInputRef = useRef(null);
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setTranscript("");
  };

  // ── Recording ───────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      mediaRecorder.current.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: "audio/wav" });
        const file = new File([blob], "recording.wav", { type: "audio/wav" });
        setAudioFile(file);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorder.current.start();
      setRecording(true);
    } catch (error){
      console.error(error);
      alert("Microphone access denied. Please allow microphone permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && recording) {
      mediaRecorder.current.stop();
      setRecording(false);
    }
  };

  // ── Main processing pipeline ────────────────────────────────────────
  const handleSummarize = useCallback(async () => {
    // Determine the API key to use
    const apiKey = settings.apiKey;
    if (!apiKey) {
      alert("Please enter your API key in Settings first.");
      setSettingsOpen(true);
      return;
    }

    // If we have pasted transcript text (no audio file), go straight to summarize
    const hasAudio = !!audioFile;
    const hasTranscript = transcript.trim().length > 0;

    if (!hasAudio && !hasTranscript) {
      alert("Please paste a transcript, upload an audio file, or record audio first.");
      return;
    }

    setLoading(true);
    setResults(null);
    let currentTranscript = transcript;
    let transcriptionData = {};

    try {
      // Step 1: Transcribe if we have audio
      if (hasAudio) {
        setLoadingStep("Transcribing audio…");
        const data = await transcribeAudio(audioFile, settings.engine, apiKey);
        currentTranscript = data.transcript;
        transcriptionData = data;
        setTranscript(currentTranscript);
      }

      // Step 2: Summarize
      setLoadingStep("Generating summary & action items…");
      const summaryData = await summarizeTranscript(
        currentTranscript,
        settings.lengthOption,
        settings.llmModel,
        apiKey
      );

      // Step 3: Speaker diarization (optional)
      let speakersData = null;
      if (settings.doSpeakers) {
        setLoadingStep("Identifying speakers…");
        const spk = await identifySpeakers(currentTranscript, settings.llmModel, apiKey);
        speakersData = spk.speakers;
      }

      setResults({
        transcript: currentTranscript,
        summary: summaryData.summary,
        actionItems: summaryData.action_items,
        language: transcriptionData.language || "Pasted",
        duration: transcriptionData.duration || 0,
        segments: transcriptionData.segments || [],
        speakers: speakersData,
      });

      // Scroll to results
      setTimeout(() => {
        document.getElementById("results")?.scrollIntoView({ behavior: "smooth" });
      }, 200);
    } catch (error){
      console.error(error);
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  }, [audioFile, transcript, settings]);

  return (
    <>
      {/* ── Navbar ──────────────────────────────────────────────────── */}
      <nav className="navbar">
        <a className="navbar-brand" href="/">
          <div className="navbar-icon"><Sparkles size={18} /></div>
          <span className="navbar-title">Voxify</span>
        </a>
        <div className="navbar-links">
          <a href="#features">How it works</a>
          <a href="#features">Features</a>
          <button
            className="navbar-cta"
            onClick={() => setSettingsOpen(true)}
            aria-label="Open settings"
          >
            <Settings size={16} style={{ verticalAlign: "middle", marginRight: 4 }} />
            Settings
          </button>
        </div>
      </nav>

      {/* ── Settings Drawer ─────────────────────────────────────────── */}
      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
      />

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <header className="hero">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          New · Speaker diarization
        </div>
        <h1>
          Meeting notes that feel<br />
          <em>handwritten</em>, not generated.
        </h1>
        <p className="hero-sub">
          Paste a transcript or record live. Voxify returns calm, well-organised notes
          — key points, action items, decisions — in the time it takes to refill your coffee.
        </p>
      </header>

      {/* ── Workspace ───────────────────────────────────────────────── */}
      <main className="workspace">
        <div className="workspace-grid">
          {/* Left — Transcript */}
          <div className="transcript-panel">
            <div className="panel-label">
              <FileText size={14} /> Transcript
            </div>
            <textarea
              className="transcript-textarea"
              placeholder="Paste your meeting transcript here, or click 'Try a sample' below…"
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
            />

            {/* Audio preview */}
            {audioUrl && (
              <div className="audio-preview">
                <audio controls src={audioUrl} />
                <span className="audio-file-info">
                  {audioFile?.name || "Recording"} · {audioFile ? (audioFile.size / (1024 * 1024)).toFixed(1) + " MB" : ""}
                </span>
              </div>
            )}

            {/* File size warning */}
            {fileTooLarge && (
              <div className="file-size-warning">
                <AlertTriangle size={16} />
                <span>
                  File exceeds <strong>25 MB</strong> limit ({(audioFile.size / (1024 * 1024)).toFixed(1)} MB).
                  Groq's Whisper API only accepts files under 25 MB. Please upload a smaller file.
                </span>
              </div>
            )}

            <div className="transcript-footer">
              <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                <span className="word-count">{wordCount} words</span>
                <button className="try-sample" onClick={() => { setTranscript(SAMPLE_TRANSCRIPT); setAudioFile(null); setAudioUrl(null); }}>
                  Try a sample
                </button>
              </div>

              <div className="transcript-actions">
                <button className="upload-btn" onClick={() => fileInputRef.current?.click()}>
                  <Upload size={15} /> Upload .wav
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp3,.mp4,.wav,.m4a,.ogg,.flac,.webm"
                    onChange={handleFileUpload}
                  />
                </button>

                <button
                  className="summarize-btn"
                  onClick={handleSummarize}
                  disabled={loading || fileTooLarge || (!transcript.trim() && !audioFile)}
                >
                  {loading ? "Processing…" : fileTooLarge ? "File too large" : "Summarize"}
                  {!loading && !fileTooLarge && <ArrowRight size={16} className="arrow" />}
                </button>
              </div>
            </div>
          </div>

          {/* Right — Record */}
          <div className="record-panel">
            <span className="record-panel-label">Or capture live</span>
            <button
              className={`record-btn ${recording ? "recording" : ""}`}
              onClick={recording ? stopRecording : startRecording}
              aria-label={recording ? "Stop recording" : "Start recording"}
            >
              {recording ? <Square size={24} /> : <Mic size={28} />}
            </button>
            <span className="record-label">
              {recording ? "Recording… Click to stop" : "Start recording"}
            </span>
            <span className="record-sub">
              Transcribes in your browser. Audio never leaves the call.
            </span>
          </div>
        </div>
      </main>

      {/* ── Loading State ───────────────────────────────────────────── */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <span className="loading-text">Analyzing your meeting…</span>
          <span className="loading-step">{loadingStep}</span>
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────────── */}
      {results && <ResultsPanel results={results} settings={settings} />}

      {/* ── Features ────────────────────────────────────────────────── */}
      <section className="features" id="features">
        <div className="feature-card">
          <div className="feature-icon"><Mic size={20} /></div>
          <h3>Live or pasted</h3>
          <p>
            Record straight in the browser or drop a transcript from Zoom, Meet, or Otter.
          </p>
        </div>
        <div className="feature-card">
          <div className="feature-icon"><ListChecks size={20} /></div>
          <h3>Action-item aware</h3>
          <p>
            Owners, deadlines, and dependencies — extracted with the right verbs attached.
          </p>
        </div>
        <div className="feature-card">
          <div className="feature-icon"><Users size={20} /></div>
          <h3>Knows who said what</h3>
          <p>
            Speaker diarization keeps quotes and decisions attributed correctly.
          </p>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-brand">
          <Sparkles size={16} />
          <span>Voxify · quiet notes for loud meetings</span>
        </div>
        <div className="footer-links">
          <a href="#">Privacy</a>
          <a href="#">Changelog</a>
          <a href="#">Contact</a>
        </div>
      </footer>
    </>
  );
}
