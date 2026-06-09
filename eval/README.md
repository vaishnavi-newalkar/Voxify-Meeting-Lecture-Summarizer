# Voxify RAGAS Evaluation

## Folder structure
```
voxify/                        ← your project root
├── utils/
│   └── summarizer.py          ← your existing file (unchanged)
├── .env                       ← must contain GROQ_API_KEY=gsk_...
└── eval/
    ├── evaluate_model.py      ← main script (run this)
    ├── eval_data.json         ← 5 sample transcripts
    ├── evaluation_results.json← auto-generated after run
    └── README.md              ← this file
```

## Setup (one time)
```bash
pip install ragas langchain-groq python-dotenv
```

## Run
```bash
# from your voxify project ROOT (not from inside eval/)
python eval/evaluate_model.py
```

## What it does
1. Loads 5 transcripts from eval_data.json
2. Calls YOUR summarize_text() function for each one (real Groq API calls)
3. Passes each (transcript, summary) pair to RAGAS Faithfulness scorer
4. Prints your average score + the exact resume bullet to copy

## Add more samples
Edit eval_data.json and add more objects:
```json
{
  "transcript": "paste your raw Whisper transcript here...",
  "length_option": "Standard (5–8 points)"
}
```
Valid length_option values:
- "Brief (3–5 points)"
- "Standard (5–8 points)"  
- "Detailed (full breakdown)"

## Expected runtime
~2–4 minutes for 5 samples (2 Groq API calls per sample)
~8–15 minutes for 20 samples