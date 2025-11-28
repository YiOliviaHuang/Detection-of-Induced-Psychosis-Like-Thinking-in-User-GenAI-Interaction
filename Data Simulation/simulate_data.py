#!/usr/bin/env python3
"""
Simulate a ChatGPT-related delusion dataset with severity labels (1–3)
and save to Excel.

Quick start:
    pip install openai pandas python-docx python-dotenv

Run:
    python simulate_delusion_dataset.py --definitions "/path/to/definition.docx" --out dataset.xlsx --model gpt-4o --per_type 10

Notes:
    - This script sets OPENAI_API_KEY in the environment at runtime using a placeholder.
      Replace "sk-YOUR-REAL-KEY" below with your real key OR set it in your shell.
    - Provide a definitions document (.txt or .docx). If omitted or unreadable,
      a compact default is used.
"""

import os
import json
import time
from typing import List, Dict, Optional
import argparse
import pandas as pd

# ----------------- SET YOUR API KEY IN THE ENV HERE -----------------
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "XXX") # replace your API key here

# ---- OpenAI client (Responses API) ----
from openai import OpenAI

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-YOUR-REAL-KEY":
        print("WARNING: OPENAI_API_KEY not set or still a placeholder.")
    return OpenAI()

# ---- Load definition text ----
def load_definition_text(path: Optional[str]) -> str:
    """
    Load the definitions text from .txt or .docx.
    Falls back to a built-in default if missing.
    """
    default_def = """\
Delusion Type Definitions (concise, DSM-5 inspired)
- Persecutory: belief one is harmed, harassed, or conspired against.
- Referential: belief that certain gestures, comments, environmental cues, and so forth are directed at oneself.
- Grandiose: belief that one has exceptional abilities, wealth, or fame
- Erotomanic: an individual believes falsely that another person is in love with him or her.
- Nihilistic: Belief that a major catastrophe will occur, or that one or the world no longer exists.
- Somatic: bodily abnormalities or infestations despite contrary evidence.
- Jealous: certainty a partner is unfaithful without proof.
- Bizarre: clearly implausible beliefs not derived from ordinary experience.
- Mixed: multiple delusional themes without one predominant.
- Unspecified: delusional content not fitting standard categories.
"""
    if not path:
        return default_def

    p = path.strip()
    if not os.path.exists(p):
        return default_def

    if p.lower().endswith(".txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return default_def

    if p.lower().endswith(".docx"):
        try:
            from docx import Document
            doc = Document(p)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception:
            return default_def

    return default_def

# call function to load definition.txt
text = load_definition_text("definition.txt")

# ---- Config ----
DELUSION_TYPES = [
    "Persecutory",
    "Referential",
    "Grandiose",
    "Erotomanic",
    "Nihilistic",
    "Somatic",
    "Jealous",
    "Bizarre",
    "Mixed",
    "Unspecified",
]

SEVERITY_RUBRIC = """\
Assign severity (1–3) for each sentence:
1 = Mild: fleeting belief, partial insight or doubt, limited distress/impairment.
2 = Moderate: persistent preoccupation, noticeable distress/functional impact.
3 = Severe: fixed, highly implausible, marked distress or danger, no insight.
"""

SYSTEM_INSTRUCTIONS = """\
You are an expert clinical rater helping create a synthetic research dataset
about delusional thoughts specifically expressed *by users directly to ChatGPT*
in first-person perspective.

Use the provided definitions document verbatim for category boundaries.

Task:
- For the requested delusion type, generate short, natural sentences (<= 30 words) describing the person's belief/feeling *caused by ChatGPT* as if the user is speaking directly to ChatGPT ("you"), not about ChatGPT in the third person ("GPT").
- Vary style, syntax, and vocabulary across sentences.
- Each output item must be valid JSON with keys:
  {"id": <int 1,2..>
   "label": "delusional" or "normal",
   "type": <str>, # delusion type if delusional, else "Normal"
   "sentence": <str>,
   "severity": <int 1..3 or 0>, #1-3 for delusions, 0 for normal
   "rationale": <short justification>}

Rules:
- Delusional thoughts: must follow the delusion definitions and severity rubric.
- Normal thoughts: ordinary or realistic experiences with ChatGPT (no delusional content).
- Proportion: about 50% delusional, 50% normal.
- No extra prose, only JSON objects.

Hard constraints:
- No personally identifying information.
- No self-harm instructions or illegal activity.
- Keep each sentence self-contained, realistic, and non-repetitive.
- Do not include clinical jargon inside the sentence itself.
"""

def build_messages(
    definition_text: str,
    type_name: str,
    n_delusional: int = 10,
    n_normal: int = 10
) -> List[Dict]:
    """
    Build strict prompts that FORCE exactly n_delusional + n_normal JSONL items.
    - Lines 1..n_delusional  : delusional items for `type_name`
    - Lines n_delusional+1.. : normal items with type "Normal"
    - First-person, user addressing ChatGPT as "you"
    - Strict JSON per line with required keys (no missing 'type')
    """
    user_text = f"""
Definitions document:
{definition_text}

Severity rubric:
{SEVERITY_RUBRIC}

OUTPUT REQUIREMENTS (STRICT, NON-NEGOTIABLE):
- Produce EXACTLY {n_delusional + n_normal} lines of output.
- Each line MUST be a SINGLE valid JSON object (JSON Lines format).
- Output NOTHING except these JSON lines (no prose, no markdown, no code fences).
- Use FIRST-PERSON voice directed at ChatGPT as "you" (e.g., "You are watching me"), never "GPT" or third-person.

COUNT & ORDER (STRICT):
- Lines 1 to {n_delusional}: DELUSIONAL items ONLY.
  - "label": "delusional"
  - "type": "{type_name}"   # must be exactly this string
  - "severity": 1, 2, or 3
- Lines {n_delusional + 1} to {n_delusional + n_normal}: NORMAL items ONLY.
  - "label": "normal"
  - "type": "Normal"        # must be exactly "Normal"
  - "severity": 0

FORMAT (STRICT JSON KEYS ON EVERY LINE):
{{
  "id": <int running from 1 to {n_delusional + n_normal}>,
  "label": "delusional" | "normal",
  "type": "{type_name}" | "Normal",
  "sentence": "<= 30 words, first-person to 'you'>",
  "severity": <1..3 for delusional; 0 for normal>,
  "rationale": "<short justification>"
}}

CONTENT RULES:
- Delusional items MUST strictly match the "{type_name}" definition from the document.
- Sentences must be short (<= 30 words), self-contained, realistic, and non-repetitive.
- No clinical jargon inside the sentence itself.
- No PII, no self-harm instructions, no illegal content.

VALIDATION CHECKS (FOLLOW BEFORE EMITTING):
- Ensure there are EXACTLY {n_delusional} delusional + {n_normal} normal items.
- Ensure all delusional items have "type": "{type_name}" and severity in [1,2,3].
- Ensure all normal items have "type": "Normal" and severity == 0.
- Ensure IDs are integers 1..{n_delusional + n_normal} with no gaps or duplicates.
- Ensure every line parses as standalone JSON (no trailing commas, no comments).

OUTPUT NOW:
- Print ONLY the {n_delusional + n_normal} JSON objects, one per line, in the exact order specified above.
"""

    return [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS}]},
        {"role": "user",   "content": [{"type": "input_text", "text": user_text}]},
    ]


def call_model(client: OpenAI, model: str, messages: List[Dict], temperature: float, max_output_tokens: int = 2000) -> str:
    """
    Call the Responses API and return the concatenated output text.
    """
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=messages,
    )
    # Prefer the convenience helper:
    try:
        return resp.output_text
    except Exception:
        # Fallback: stitch together text chunks if structure differs
        chunks = []
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            chunks.append(getattr(c, "text", ""))
        return "\n".join(chunks)

def parse_jsonl(text: str) -> List[Dict]:
    """
    Parse JSON Lines. Skip malformed lines gracefully.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if all(k in obj for k in ("type", "sentence", "severity")):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out

def simulate_dataset(
    definition_path: Optional[str],
    out_xlsx: str = "chatgpt_delusion_dataset.xlsx",
    model: str = "gpt-4o",
    per_type: int = 10,
    sleep_s: float = 0.4,
    temperature: float = 0.4,
) -> pd.DataFrame:
    """
    Generate a dataset by calling the model per delusion type.
    Returns the DataFrame and writes it to Excel.
    """
    client = get_client()
    definitions = load_definition_text(definition_path)

    rows: List[Dict] = []
    for t in DELUSION_TYPES:
        messages = build_messages(definitions, t, per_type)
        txt = call_model(client, model=model, messages=messages, temperature=0.4)
        items = parse_jsonl(txt)

        for i, it in enumerate(items, 1):
            rows.append({
                "type": it.get("type", t),
                "sentence_id": i,
                "sentence": it.get("sentence", ""),
                "severity": it.get("severity", None),
                "rationale": it.get("rationale", ""),
                "model": model,
                "temperature": temperature,
            })

        time.sleep(sleep_s)  # pacing

    df = pd.DataFrame(rows, columns=[
        "type","sentence_id","sentence","severity","rationale","model","temperature"
    ])
    # QA helpers
    df["word_count"] = df["sentence"].str.split().apply(len)
    df["over_30_words"] = df["word_count"] > 30

    df.to_excel(out_xlsx, index=False)
    print(f"Saved {len(df)} rows to {out_xlsx}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Simulate ChatGPT-related delusion dataset with severity labels.")
    parser.add_argument("--definitions", type=str, default=None, help="Path to definitions (.txt or .docx). Optional.")
    parser.add_argument("--out", type=str, default="chatgpt_delusion_dataset.xlsx", help="Output Excel path.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model, e.g., gpt-5-mini, gpt-4o-mini.")
    parser.add_argument("--per_type", type=int, default=10, help="Number of sentences per delusion type.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Pause between requests (seconds).")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    args = parser.parse_args()

    simulate_dataset(
        definition_path=args.definitions,
        out_xlsx=args.out,
        model=args.model,
        per_type=args.per_type,
        sleep_s=args.sleep,
        temperature=args.temperature,
    )

if __name__ == "__main__":
    main()
