# 🎙️ Voxify — AI Meeting Intelligence

> **Resume-grade AI project** — Speech-to-text + summarization using 100% open-source models via Groq & HuggingFace APIs. No OpenAI dependency.

![Python](https://img.shields.io/badge/Python-3.10+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red) ![Groq](https://img.shields.io/badge/Groq-Whisper%20Large%20v3-green) ![License](https://img.shields.io/badge/License-MIT-blue)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎙️ **Speech-to-Text** | Groq Whisper Large v3 Turbo (fastest) or HuggingFace Whisper Large v3 |
| 🤖 **AI Summarization** | LLaMA 3.3-70B / Mixtral-8x7B / Gemma2-9B via Groq |
| ✅ **Action Item Extraction** | Structured JSON output from LLM |
| 👥 **Speaker Diarization** | LLM-based speaker identification |
| ⏱️ **Timestamps** | Segment-level timestamps from Groq Whisper |
| 🎤 **Live Recording** | Browser microphone recording via `streamlit-audiorecorder` |
| 📥 **Export** | TXT, PDF, raw transcript, action items checklist |
| 🎨 **Aesthetic UI** | Warm earthy dark theme — not the default blue/purple |

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone <your-repo>
cd voxify
pip install -r requirements.txt

# 2. (Optional) Enable live recording
pip install streamlit-audiorecorder

# 3. Run
streamlit run app.py
```

---

## 🔑 API Keys Required

| Service | Get Key | Used For |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) — **Free** | Whisper transcription + LLM summarization |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — **Free** | Alternative Whisper via HF Inference API |

> Groq is recommended — it's 10–20× faster than HuggingFace Inference API for audio.

---

## 🏗️ Architecture

```
app.py                     ← Streamlit UI (warm earthy theme)
utils/
  transcriber.py           ← Groq Whisper API + HF Whisper API
  summarizer.py            ← Groq LLaMA/Mixtral chat completions
  exporter.py              ← TXT + PDF export
requirements.txt
```

### Tech Stack

- **Frontend**: Streamlit with custom CSS (warm/earthy dark theme)
- **STT**: Groq `whisper-large-v3-turbo` (500 tok/s) or HF `whisper-large-v3`
- **LLM**: `llama-3.3-70b-versatile` / `mixtral-8x7b-32768` / `gemma2-9b-it` — all via Groq
- **Audio**: `streamlit-audiorecorder` for live mic recording
- **Export**: `fpdf2` for PDF generation

---

## 📁 Supported Audio Formats

MP3, MP4, WAV, M4A, OGG, FLAC, WEBM — up to 25 MB (Groq limit)


1. **Upload tab** — drag & drop audio file, instant preview
2. **Live Recording tab** — mic button with waveform (requires `streamlit-audiorecorder`)
3. **Results** — metrics row, transcript + summary side-by-side, action items, speaker labels
4. **Export** — TXT, PDF, transcript-only, action items checklist
