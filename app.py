import streamlit as st
import os
import io
import wave
import struct
import tempfile
import numpy as np
from utils.transcriber import transcribe_with_groq, transcribe_with_huggingface
from utils.summarizer import summarize_text, extract_action_items, identify_speakers
from utils.exporter import export_to_txt, export_to_pdf

# ── Audio Preprocessor (no ffmpeg required) ───────────────────────────────────
def preprocess_wav_bytes(raw_bytes: bytes, target_rate: int = 16000) -> bytes:
    """
    Reads raw WAV bytes, converts to mono 16kHz 16-bit PCM WAV.
    Uses only stdlib (wave) + numpy — no ffmpeg/pydub needed.
    Returns processed WAV bytes.
    """
    with wave.open(io.BytesIO(raw_bytes)) as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()   # bytes per sample
        frame_rate = wf.getframerate()
        n_frames   = wf.getnframes()
        raw_pcm    = wf.readframes(n_frames)

    # Decode to numpy
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(samp_width, np.int16)
    samples = np.frombuffer(raw_pcm, dtype=dtype).astype(np.float32)

    # Interleaved → mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to target_rate using linear interpolation
    if frame_rate != target_rate:
        duration   = len(samples) / frame_rate
        new_length = int(duration * target_rate)
        samples    = np.interp(
            np.linspace(0, len(samples) - 1, new_length),
            np.arange(len(samples)),
            samples
        )

    # Normalise to int16
    max_val = np.max(np.abs(samples)) or 1.0
    samples = (samples / max_val * 32767).astype(np.int16)

    # Write back to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)           # 16-bit
        wf_out.setframerate(target_rate)
        wf_out.writeframes(samples.tobytes())
    return buf.getvalue()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voxify — AI Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Warm Earthy Aesthetic ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark warm background */
  .stApp {
    background: #0f0e0c;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #141310 !important;
    border-right: 1px solid #2a2720 !important;
  }

  /* Logo + title */
  .voxify-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
  }
  .voxify-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #d97706, #92400e);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .voxify-title {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f59e0b, #d97706);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  /* Hero headline */
  .hero-headline {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.15;
    color: #faf7f2;
    letter-spacing: -0.03em;
    margin-bottom: 0.6rem;
  }
  .hero-sub {
    font-size: 1.1rem;
    color: #78716c;
    margin-bottom: 2.5rem;
    max-width: 520px;
  }
  .accent-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(217,119,6,0.12);
    border: 1px solid rgba(217,119,6,0.25);
    color: #f59e0b;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1.2rem;
  }

  /* Cards */
  .glass-card {
    background: #1a1814;
    border: 1px solid #2a2720;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.2rem;
  }
  .result-card {
    background: #161412;
    border: 1px solid #2a2720;
    border-radius: 14px;
    padding: 1.5rem;
  }

  /* Metric pills */
  .metric-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem;
  }
  .metric-pill {
    background: #1e1c18;
    border: 1px solid #2e2b25;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 110px;
  }
  .metric-pill .val {
    font-size: 1.2rem; font-weight: 600; color: #f59e0b;
  }
  .metric-pill .lbl {
    font-size: 0.72rem; color: #57534e; margin-top: 2px;
  }

  /* Action items */
  .action-item {
    background: rgba(16,185,129,0.07);
    border-left: 3px solid #10b981;
    padding: 0.55rem 1rem;
    margin: 0.35rem 0;
    border-radius: 0 8px 8px 0;
    color: #d1fae5;
    font-size: 0.9rem;
  }

  /* Section labels */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #57534e;
    margin-bottom: 0.6rem;
  }

  /* Transcript area */
  .stTextArea textarea {
    background: #1a1814 !important;
    border: 1px solid #2a2720 !important;
    color: #d6d3cd !important;
    border-radius: 12px !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
  }
  [data-testid="stBaseButton-primary"] > button,
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #d97706, #b45309) !important;
    border: none !important;
    color: #fff !important;
  }
  [data-testid="stBaseButton-primary"] > button:hover,
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(217,119,6,0.3) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #1a1814;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #2a2720;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #78716c !important;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #292420 !important;
    color: #f59e0b !important;
  }

  /* Status/alerts */
  .success-bar {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #6ee7b7;
    font-weight: 500;
  }
  .warn-bar {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #fcd34d;
    font-weight: 500;
  }

  /* Divider */
  hr { border-color: #2a2720 !important; }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #1a1814 !important;
    border: 2px dashed #2e2b25 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #d97706 !important;
  }

  /* Select boxes & inputs in sidebar */
  .stSelectbox > div > div,
  .stTextInput > div > div > input,
  .stRadio > div {
    background: #1e1c18 !important;
    border-color: #2a2720 !important;
    color: #d6d3cd !important;
  }

  /* Speaker diarization styled block */
  .speaker-block {
    background: #1e1c18;
    border: 1px solid #2a2720;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
    color: #d6d3cd;
    line-height: 1.6;
  }
  .speaker-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #f59e0b;
    margin-bottom: 4px;
  }

  /* Recording pulse animation */
  @keyframes pulse-amber {
    0%, 100% { box-shadow: 0 0 0 0 rgba(217,119,6,0.4); }
    50%       { box-shadow: 0 0 0 12px rgba(217,119,6,0); }
  }
  .recording-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #ef4444;
    border-radius: 50%;
    animation: pulse-amber 1.2s ease-in-out infinite;
    margin-right: 8px;
    vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="voxify-logo">
      <div class="voxify-icon">🎙️</div>
      <span class="voxify-title">Voxify</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Configuration")
    st.divider()

    # STT Backend
    st.markdown('<div class="section-label">🔊 Speech-to-Text Engine</div>', unsafe_allow_html=True)
    stt_backend = st.selectbox(
        "Engine",
        ["Groq (Whisper Large v3 Turbo) — Recommended", "HuggingFace (Whisper Large v3)"],
        label_visibility="collapsed"
    )
    use_groq = "Groq" in stt_backend

    # API Key
    if use_groq:
        st.markdown('<div class="section-label">🔑 Groq API Key</div>', unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq Key",
            type="password",
            placeholder="gsk_...",
            help="Free at console.groq.com — fastest inference",
            label_visibility="collapsed"
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            st.markdown('<div class="success-bar">✅ Groq key saved</div>', unsafe_allow_html=True)
        hf_key = ""
    else:
        st.markdown('<div class="section-label">🔑 HuggingFace API Key</div>', unsafe_allow_html=True)
        hf_key = st.text_input(
            "HF Key",
            type="password",
            placeholder="hf_...",
            help="Get free at huggingface.co/settings/tokens",
            label_visibility="collapsed"
        )
        if hf_key:
            os.environ["HF_API_KEY"] = hf_key
            st.markdown('<div class="success-bar">✅ HuggingFace key saved</div>', unsafe_allow_html=True)
        groq_key = ""

    st.divider()

    # Summarizer
    st.markdown('<div class="section-label">🤖 LLM for Summarization</div>', unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "LLM",
        [
            "Groq — llama-3.3-70b (Recommended)",
            "Groq — llama-3.1-8b-instant (Fast)",
            "Groq — llama3-70b-8192 (Balanced)",
        ],
        label_visibility="collapsed"
    )
    llm_map = {
        "Groq — llama-3.3-70b (Recommended)": "llama-3.3-70b-versatile",
        "Groq — llama-3.1-8b-instant (Fast)":  "llama-3.1-8b-instant",
        "Groq — llama3-70b-8192 (Balanced)":   "llama3-70b-8192",
    }
    selected_llm = llm_map[llm_choice]

    st.divider()

    st.markdown('<div class="section-label">📝 Summary Depth</div>', unsafe_allow_html=True)
    summary_length = st.radio(
        "Length",
        ["Brief (3–5 points)", "Standard (5–8 points)", "Detailed (full breakdown)"],
        index=1,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown('<div class="section-label">🔬 Extra Analysis</div>', unsafe_allow_html=True)
    do_speakers = st.toggle("Speaker Diarization", value=False)
    do_timestamps = st.toggle("Show Timestamps", value=False)

    st.divider()
    st.caption("Voxify v2.0 — Groq + HuggingFace\nOpen-source speech intelligence")

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="accent-badge">⚡ Powered by Groq · HuggingFace · Open Source</div>
<div class="hero-headline">Meeting Intelligence,<br>Instantly.</div>
<div class="hero-sub">Upload or record your meeting. Get a transcript, smart summary, and action items — in seconds.</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_record = st.tabs(["📁  Upload Audio File", "🎤  Live Recording"])

audio_bytes = None
input_source = None

with tab_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Drop your meeting audio here")
    uploaded_file = st.file_uploader(
        "Supported: MP3, MP4, WAV, M4A, OGG, FLAC, WEBM",
        type=["mp3", "mp4", "wav", "m4a", "ogg", "flac", "webm"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.audio(uploaded_file)
        c1, c2, c3 = st.columns(3)
        name = uploaded_file.name
        c1.metric("📄 File", name[:22] + "…" if len(name) > 22 else name)
        c2.metric("📦 Size", f"{uploaded_file.size / 1024:.1f} KB")
        c3.metric("🎵 Format", name.rsplit(".", 1)[-1].upper())
        input_source = "upload"
        audio_bytes = uploaded_file.getbuffer()
    st.markdown('</div>', unsafe_allow_html=True)

with tab_record:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎙️ Live Recording")
    st.markdown(
        '<div class="warn-bar">🎤 Click <strong>Start recording</strong> below. '
        'Audio is automatically preprocessed to 16 kHz mono WAV before analysis.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    recorded_audio = st.audio_input("Record your meeting", key="live_audio")

    if recorded_audio is not None:
        raw_bytes = recorded_audio.read()

        # ── Preprocess: convert to 16 kHz mono 16-bit WAV ────────────────────
        with st.spinner("⚙️ Preprocessing audio (resampling to 16 kHz mono WAV)…"):
            try:
                processed_bytes = preprocess_wav_bytes(raw_bytes, target_rate=16000)
                proc_ok = True
            except Exception as e:
                st.warning(f"⚠️ Preprocessing skipped ({e}). Using raw audio.")
                processed_bytes = raw_bytes
                proc_ok = False

        # Compute duration from processed WAV
        try:
            with wave.open(io.BytesIO(processed_bytes)) as _wf:
                _dur = _wf.getnframes() / _wf.getframerate()
        except Exception:
            _dur = 0.0

        st.audio(processed_bytes, format="audio/wav")
        status_icon = "✅" if proc_ok else "⚠️"
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill"><div class="val">{_dur:.1f}s</div><div class="lbl">Duration</div></div>
          <div class="metric-pill"><div class="val">16 kHz</div><div class="lbl">Sample Rate</div></div>
          <div class="metric-pill"><div class="val">Mono</div><div class="lbl">Channels</div></div>
          <div class="metric-pill"><div class="val">{status_icon}</div><div class="lbl">Preprocessed</div></div>
        </div>
        """, unsafe_allow_html=True)

        audio_bytes   = processed_bytes
        input_source  = "record"

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ── Action Buttons ─────────────────────────────────────────────────────────────
col_go, col_clr, col_space = st.columns([1, 1, 3])
process_btn = col_go.button("🚀  Analyze Meeting", type="primary", use_container_width=True)
clear_btn   = col_clr.button("🗑️  Clear Results", use_container_width=True)

if clear_btn:
    for k in ["transcript", "summary", "action_items", "language", "duration", "segments", "speakers"]:
        st.session_state.pop(k, None)
    st.rerun()

# ── API key validation helper ─────────────────────────────────────────────────
def get_active_key():
    if use_groq:
        return os.environ.get("GROQ_API_KEY", groq_key)
    return os.environ.get("HF_API_KEY", hf_key)

# ── Processing ────────────────────────────────────────────────────────────────
if process_btn:
    active_key = get_active_key()

    if not active_key:
        engine_name = "Groq" if use_groq else "HuggingFace"
        st.error(f"❌ Please enter your {engine_name} API key in the sidebar first.")
        st.stop()
    if not audio_bytes:
        st.error("❌ Please upload an audio file or record audio first.")
        st.stop()

    # Save to temp file
    suffix = ".wav" if input_source == "record" else f".{uploaded_file.name.rsplit('.', 1)[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # ── Step 1: Transcription ─────────────────────────────────────────────────
    engine_label = "Groq Whisper Large v3 Turbo" if use_groq else "HuggingFace Whisper Large v3"
    with st.status(f"🎙️ Transcribing with {engine_label}…", expanded=True) as status:
        st.write("Sending audio to API…")
        try:
            if use_groq:
                transcript, language, duration, segments = transcribe_with_groq(tmp_path, active_key)
            else:
                transcript, language, duration, segments = transcribe_with_huggingface(tmp_path, active_key)

            if transcript:
                st.session_state.update({
                    "transcript": transcript,
                    "language": language,
                    "duration": duration,
                    "segments": segments
                })
                st.write(f"✅ Done — {language} detected · {duration:.1f}s")
                status.update(label="✅ Transcription complete!", state="complete")
            else:
                status.update(label="❌ Transcription failed", state="error")
                st.error("Transcription returned empty. Check your audio or API key.")
                st.stop()
        except Exception as e:
            status.update(label="❌ Transcription error", state="error")
            st.error(f"Transcription error: {e}")
            st.stop()

    # ── Step 2: Summarization ─────────────────────────────────────────────────
    with st.status("🤖 Generating summary with Groq LLM…", expanded=True) as status:
        st.write(f"Model: {selected_llm}")
        try:
            groq_llm_key = os.environ.get("GROQ_API_KEY", groq_key)
            if not groq_llm_key:
                groq_llm_key = active_key  # fallback

            summary      = summarize_text(st.session_state["transcript"], summary_length, selected_llm, groq_llm_key)
            action_items = extract_action_items(st.session_state["transcript"], selected_llm, groq_llm_key)

            st.session_state.update({"summary": summary, "action_items": action_items})
            st.write("✅ Summary & action items ready")
            status.update(label="✅ Analysis complete!", state="complete")
        except Exception as e:
            status.update(label="⚠️ Summarization failed", state="error")
            st.warning(f"Summarization error: {e}")

    # ── Optional: Speaker diarization ────────────────────────────────────────
    if do_speakers:
        with st.status("👥 Identifying speakers…", expanded=True) as status:
            try:
                groq_llm_key = os.environ.get("GROQ_API_KEY", groq_key) or active_key
                speaker_text = identify_speakers(st.session_state["transcript"], selected_llm, groq_llm_key)
                st.session_state["speakers"] = speaker_text
                status.update(label="✅ Speaker diarization done!", state="complete")
            except Exception as e:
                status.update(label="⚠️ Speaker ID failed", state="error")

    st.success("🎉 Processing complete! Results below ↓")

# ── Results ───────────────────────────────────────────────────────────────────
if "transcript" in st.session_state:
    st.divider()

    # Metrics row
    words = len(st.session_state["transcript"].split())
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-pill"><div class="val">{st.session_state.get('language','—')}</div><div class="lbl">Language</div></div>
      <div class="metric-pill"><div class="val">{words:,}</div><div class="lbl">Words</div></div>
      <div class="metric-pill"><div class="val">{st.session_state.get('duration',0):.1f}s</div><div class="lbl">Duration</div></div>
      <div class="metric-pill"><div class="val">{len(st.session_state['transcript']):,}</div><div class="lbl">Characters</div></div>
      <div class="metric-pill"><div class="val">{len(st.session_state.get('action_items',[]))}</div><div class="lbl">Action Items</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Transcript + Summary side by side ─────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">📄 Full Transcript</div>', unsafe_allow_html=True)

        # Timestamps view
        if do_timestamps and st.session_state.get("segments"):
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            for seg in st.session_state["segments"]:
                start = seg.get("start", 0)
                end   = seg.get("end", 0)
                text  = seg.get("text", "")
                m_s, s_s = divmod(int(start), 60)
                m_e, s_e = divmod(int(end), 60)
                ts = f"{m_s:02d}:{s_s:02d} → {m_e:02d}:{s_e:02d}"
                st.markdown(f"""
                <div style="margin-bottom:0.5rem">
                  <span style="font-size:0.72rem;color:#d97706;font-family:monospace">{ts}</span>
                  <span style="font-size:0.88rem;color:#d6d3cd;margin-left:8px">{text}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.text_area(
                label="Full Transcript",
                value=st.session_state["transcript"],
                height=380,
                key="transcript_area",
                label_visibility="collapsed"
            )

    with col_right:
        st.markdown('<div class="section-label">🤖 AI Summary</div>', unsafe_allow_html=True)
        if "summary" in st.session_state:
            with st.container():
                st.markdown(f'<div class="result-card">{st.session_state["summary"]}</div>', unsafe_allow_html=True)

    # ── Speaker Diarization ───────────────────────────────────────────────────
    if "speakers" in st.session_state:
        st.divider()
        st.markdown('<div class="section-label">👥 Speaker Diarization</div>', unsafe_allow_html=True)
        lines = st.session_state["speakers"].split("\n")
        for line in lines:
            if line.strip():
                if line.strip().startswith("Speaker"):
                    parts = line.split(":", 1)
                    label = parts[0].strip()
                    content = parts[1].strip() if len(parts) > 1 else ""
                    st.markdown(f"""
                    <div class="speaker-block">
                      <div class="speaker-label">{label}</div>
                      {content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="speaker-block">{line}</div>', unsafe_allow_html=True)

    # ── Action Items ──────────────────────────────────────────────────────────
    if st.session_state.get("action_items"):
        st.divider()
        st.markdown('<div class="section-label">✅ Action Items</div>', unsafe_allow_html=True)
        for item in st.session_state["action_items"]:
            st.markdown(f'<div class="action-item">✔ {item}</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">📥 Export Results</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        txt_data = export_to_txt(
            st.session_state.get("transcript", ""),
            st.session_state.get("summary", ""),
            st.session_state.get("action_items", [])
        )
        st.download_button("📄 Download TXT", data=txt_data, file_name="meeting_report.txt",
                           mime="text/plain", use_container_width=True)

    with e2:
        try:
            pdf_data = export_to_pdf(
                st.session_state.get("transcript", ""),
                st.session_state.get("summary", ""),
                st.session_state.get("action_items", [])
            )
            st.download_button("📑 Download PDF", data=pdf_data, file_name="meeting_report.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception:
            st.button("📑 PDF (pip install fpdf2)", disabled=True, use_container_width=True)

    with e3:
        st.download_button(
            "📝 Transcript Only",
            data=st.session_state.get("transcript", "").encode(),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True
        )

    with e4:
        if st.session_state.get("action_items"):
            items_txt = "\n".join(f"[ ] {i}" for i in st.session_state["action_items"])
            st.download_button(
                "✅ Action Items",
                data=items_txt.encode(),
                file_name="action_items.txt",
                mime="text/plain",
                use_container_width=True
            )
