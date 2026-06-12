"""
evaluate_model.py — RAGAS Faithfulness evaluation for Voxify
Tailored to utils/summarizer.py (sync, requests-based, no async)

Run from your voxify project ROOT:
    python eval/evaluate_model.py

Requirements:
    pip install ragas langchain-groq python-dotenv
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv



# ── RAGAS imports (v0.2+ API) ────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq


# ── Path setup — lets us import from utils/ ──────────────────────────────────
ROOT = Path(__file__).parent.parent          # voxify project root
sys.path.insert(0, str(ROOT))
from utils.summarizer import summarize_text   # noqa: E402
load_dotenv(ROOT / ".env")                   # loads GROQ_API_KEY



# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. "
        "Add it to your .env file at the project root: GROQ_API_KEY=gsk_..."
    )

MODEL          = "llama-3.3-70b-versatile"
EVAL_DATA_PATH = Path(__file__).parent / "eval_data.json"
RESULTS_PATH   = Path(__file__).parent / "evaluation_results.json"


# ── Evaluator LLM — same Groq key you already have ───────────────────────────
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,          # deterministic scoring
    )
)

# Faithfulness: fraction of claims in the summary supported by the transcript
# Score 0.0–1.0. Higher = less hallucination.
# No embeddings needed — LLM-only metric. Simplest and most resume-relevant.
faithfulness_metric = Faithfulness(llm=evaluator_llm)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation():
    # 1. Load eval cases
    if not EVAL_DATA_PATH.exists():
        print(f"ERROR: {EVAL_DATA_PATH} not found.")
        print("Make sure eval_data.json is in the same folder as this script.")
        sys.exit(1)

    with open(EVAL_DATA_PATH, encoding="utf-8") as f:
        eval_cases = json.load(f)

    print(f"Loaded {len(eval_cases)} eval cases from {EVAL_DATA_PATH.name}")
    print("=" * 55)

    # 2. Generate summaries using YOUR actual Voxify pipeline
    samples = []
    for i, case in enumerate(eval_cases, 1):
        transcript   = case["transcript"]
        length_option = case["length_option"]

        print(f"[{i}/{len(eval_cases)}] Generating summary ...", end=" ", flush=True)

        try:
            # ── This calls YOUR exact function with YOUR exact signature ──────
            summary = summarize_text(
                transcript=transcript,
                length_option=length_option,
                model=MODEL,
                api_key=GROQ_API_KEY,
            )
            print("done")

            # Map to RAGAS SingleTurnSample (v0.2+ field names)
            samples.append(SingleTurnSample(
                user_input=length_option,         # the instruction / "question"
                response=summary,                 # Voxify's generated summary
                retrieved_contexts=[transcript],  # the source transcript (must be list)
            ))

        except Exception as e:
            print(f"FAILED — {e}")
            print("  Skipping this sample and continuing...")

    if not samples:
        print("\nNo samples were successfully processed. Check your GROQ_API_KEY.")
        sys.exit(1)

    print(f"\nSuccessfully generated {len(samples)} summaries.")
    print("=" * 55)

    # 3. Build RAGAS EvaluationDataset (v0.2+ API — NOT Dataset.from_dict)
    dataset = EvaluationDataset(samples=samples)

    # 4. Run evaluation — llm= is required to prevent silent OpenAI fallback
    print("Running RAGAS Faithfulness evaluation (this takes 1–3 min)...")
    print("Each sample makes 2 LLM calls: keyphrase extraction + claim verification.\n")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric],
        llm=evaluator_llm,          # required — prevents OpenAI fallback
    )

    # 5. Parse and display results
    df = results.to_pandas()

    avg_score  = round(float(df["faithfulness"].mean()), 4)
    best_score = round(float(df["faithfulness"].max()), 4)
    low_score  = round(float(df["faithfulness"].min()), 4)

    print("\n" + "=" * 55)
    print(f"  Samples evaluated  : {len(df)}")
    print(f"  Average Faithfulness : {avg_score}  ← USE THIS NUMBER")
    print(f"  Best               : {best_score}")
    print(f"  Worst              : {low_score}")
    print("=" * 55)

    # 6. Save full results to JSON
    output = {
        "avg_faithfulness": avg_score,
        "best": best_score,
        "worst": low_score,
        "n_samples": len(df),
        "per_sample": [
            {
                "case": i + 1,
                "length_option": eval_cases[i]["length_option"],
                "faithfulness": round(float(row["faithfulness"]), 4),
                "summary_preview": str(row["response"])[:120] + "...",
            }
            for i, row in df.iterrows()
        ]
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to: {RESULTS_PATH}")

    # 7. Print the exact resume bullet to copy
    print("\n" + "━" * 55)
    print("RESUME BULLET — copy and paste this:")
    print("━" * 55)
    print(
        f"Evaluated LLM output quality using RAGAS Faithfulness "
        f"metric — achieved {avg_score} on a {len(df)}-sample internal "
        f"test set, confirming {round((1 - avg_score) * 100, 1)}% hallucination "
        f"rate on real meeting and lecture transcripts."
    )
    print("━" * 55)


if __name__ == "__main__":
    run_evaluation()
