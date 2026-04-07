"""
summarizer.py — AI summarization via Groq (LLaMA 3.3-70b / LLaMA 3.1-8b / LLaMA3-70b)
No OpenAI dependency — fully open-source models.
"""

import json
import requests


# ── Groq Chat helper ──────────────────────────────────────────────────────────
def _groq_chat(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int = 1200,
    temperature: float = 0.3,
) -> str:
    """Send a chat completion request to Groq and return the text response."""
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq LLM error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Summarize ────────────────────────────────────────────────────────────────
def summarize_text(
    transcript: str,
    length_option: str,
    model: str,
    api_key: str,
) -> str:
    """
    Summarize a meeting transcript using a Groq-hosted open-source LLM.

    Args:
        transcript    : Full transcript text
        length_option : UI radio option ("Brief", "Standard", "Detailed")
        model         : Groq model ID (e.g. llama-3.3-70b-versatile)
        api_key       : Groq API key

    Returns:
        Markdown-formatted summary string
    """
    if not transcript.strip():
        return "No transcript provided."

    length_map = {
        "Brief (3–5 points)":          "Provide a BRIEF summary with exactly 3–5 concise bullet points covering the core takeaways only.",
        "Standard (5–8 points)":       "Provide a STANDARD summary with 5–8 bullet points covering all main topics and key decisions.",
        "Detailed (full breakdown)":   (
            "Provide a DETAILED analysis with:\n"
            "1) **Overview** — 2-sentence recap\n"
            "2) **Key Discussion Points** — bullet list\n"
            "3) **Decisions Made** — bullet list\n"
            "4) **Open Questions / Next Steps** — bullet list"
        ),
    }
    # Fuzzy match the key
    matched_instruction = next(
        (v for k, v in length_map.items() if k in length_option or length_option in k),
        length_map["Standard (5–8 points)"]
    )

    system = (
        "You are an expert meeting analyst and executive assistant. "
        "Analyze meeting or lecture transcripts and produce clear, professional summaries. "
        "Always use Markdown formatting. Be concise and highlight what matters most. "
        "Do not invent details not present in the transcript."
    )

    user = f"""{matched_instruction}

TRANSCRIPT:
\"\"\"
{transcript[:14000]}
\"\"\"

Produce the summary now:"""

    return _groq_chat(system, user, model, api_key, max_tokens=1200, temperature=0.3)


# ── Extract Action Items ──────────────────────────────────────────────────────
def extract_action_items(
    transcript: str,
    model: str,
    api_key: str,
) -> list[str]:
    """
    Extract concrete action items / tasks from a transcript.

    Returns:
        List of plain-text action item strings
    """
    if not transcript.strip():
        return []

    system = "You extract action items from meeting transcripts. Always respond ONLY with a JSON array of strings. No explanation, no markdown fences."

    user = f"""Extract all ACTION ITEMS, tasks, decisions, and follow-ups from the transcript below.
Return ONLY a JSON array of strings. Example: ["Send report by Friday", "Schedule follow-up with team"]

TRANSCRIPT:
\"\"\"
{transcript[:10000]}
\"\"\"
"""

    try:
        raw   = _groq_chat(system, user, model, api_key, max_tokens=500, temperature=0.1)
        clean = raw.replace("```json", "").replace("```", "").strip()
        items = json.loads(clean)
        return items if isinstance(items, list) else []
    except Exception as e:
        print(f"[Summarizer] Action item extraction failed: {e}")
        return []


# ── Speaker Diarization ───────────────────────────────────────────────────────
def identify_speakers(
    transcript: str,
    model: str,
    api_key: str,
) -> str:
    """
    Attempt to identify and label different speakers in the transcript.

    Returns:
        Reformatted transcript with Speaker labels
    """
    if not transcript.strip():
        return transcript

    system = (
        "You are an expert at analyzing conversation transcripts. "
        "Identify different speakers based on conversational shifts, question/answer patterns, and topic changes. "
        "Label them Speaker 1, Speaker 2, etc. For monologues, use Speaker 1 throughout."
    )

    user = f"""Analyze this transcript and reformat it by identifying different speakers.
Label each speaker clearly as 'Speaker 1:', 'Speaker 2:', etc.
If it is clearly a single-speaker monologue, prefix each paragraph with 'Speaker 1:'.
Return ONLY the reformatted transcript — no preamble.

TRANSCRIPT:
\"\"\"
{transcript[:10000]}
\"\"\"
"""

    try:
        return _groq_chat(system, user, model, api_key, max_tokens=2500, temperature=0.2)
    except Exception as e:
        print(f"[Summarizer] Speaker ID failed: {e}")
        return transcript
