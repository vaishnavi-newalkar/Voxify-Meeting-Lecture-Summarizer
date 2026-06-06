const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function transcribeAudio(file, engine, apiKey) {
  const form = new FormData();
  form.append("file", file);
  form.append("engine", engine);
  form.append("api_key", apiKey);

  const res = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Transcription failed");
  }
  return res.json();
}

export async function summarizeTranscript(transcript, lengthOption, model, apiKey) {
  const form = new FormData();
  form.append("transcript", transcript);
  form.append("length_option", lengthOption);
  form.append("model", model);
  form.append("api_key", apiKey);

  const res = await fetch(`${API_BASE}/api/summarize`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Summarization failed");
  }
  return res.json();
}

export async function identifySpeakers(transcript, model, apiKey) {
  const form = new FormData();
  form.append("transcript", transcript);
  form.append("model", model);
  form.append("api_key", apiKey);

  const res = await fetch(`${API_BASE}/api/speakers`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Speaker identification failed");
  }
  return res.json();
}

export async function exportTxt(transcript, summary, actionItems) {
  const form = new FormData();
  form.append("transcript", transcript);
  form.append("summary", summary);
  form.append("action_items", JSON.stringify(actionItems));

  const res = await fetch(`${API_BASE}/api/export/txt`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}

export async function exportPdf(transcript, summary, actionItems) {
  const form = new FormData();
  form.append("transcript", transcript);
  form.append("summary", summary);
  form.append("action_items", JSON.stringify(actionItems));

  const res = await fetch(`${API_BASE}/api/export/pdf`, { method: "POST", body: form });
  if (!res.ok) throw new Error("PDF export failed");
  return res.blob();
}
