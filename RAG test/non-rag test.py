# nonrag_run.py
# Purpose:
# - Do NOT use rag.xlsx or any retrieval (no RAG)
# - Use test.xlsx as the test set only
# - For each row in test.xlsx, ask GPT to classify based on the sentence alone
# - Save predictions to test_with_predictions_noRAG.xlsx (+ metrics sheet if gold available)

import os, json, time
import pandas as pd
from openai import OpenAI

# ========= SETTINGS =========================================
API_KEY = "XXX"   # set your own OpenAI API key here or use environment variable OPENAI_API_KEY

TEST_PATH  = "test.xlsx"
OUT_PATH   = "test_with_predictions_noRAG.xlsx"

TEXT_COL        = "sentence"          # column in test.xlsx
HUMAN_DIAG_COLS = ["diagnosis", "dignosis"]  # auto-detect which exists
HUMAN_SEV_COL   = "severity"          # optional

MODEL_LLM  = "gpt-4o-mini"
TEMP       = 0                        # deterministic
MAX_ROWS   = None                     # e.g., 50 for a dry run. None -> all
PAUSE_S    = 0.05                     # gentle pacing
# =============================================================================

NONRAG_PROMPT_TEMPLATE = """
You are a psychology content classifier.

[USER STATEMENT]
{user_text}

Classify and return STRICT JSON with the following keys:
- "diagnosis": 1 or 0 (1 means "delusional", 0 means "non-delusional")
- "severity": null or 1/2/3 (1=mild, 2=moderate, 3=severe); only set if diagnosis is "delusional"
- "confidence": integer 0-100 indicating classifier confidence (not clinical certainty)
- "reasoning": one brief, neutral sentence explaining the decision (no PHI, no clinical claims)

Return ONLY the JSON.
"""

def call_llm(client: OpenAI, prompt: str) -> dict:
    """Call the chat model and robustly parse JSON from its output."""
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
    )
    content = resp.choices[0].message.content.strip()

    # Try direct JSON first
    try:
        if content.startswith("{"):
            return json.loads(content)
        # Fallback: extract first {...} block
        start = content.find("{"); end = content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
    except Exception:
        pass
    # If parsing fails, return raw
    return {"raw": content}

def normalize_prediction(obj: dict) -> dict:
    """Normalize model outputs into consistent types/labels."""
    if not isinstance(obj, dict):
        return {
            "pred_diagnosis": None,
            "pred_severity": None,
            "pred_confidence": None,
            "pred_reasoning": str(obj),
        }

    diag = obj.get("diagnosis")
    sev  = obj.get("severity")
    conf = obj.get("confidence")
    reas = obj.get("reasoning") or obj.get("explanation")

    # Normalize diagnosis: accept 0/1 or text variants
    if isinstance(diag, (int, float)):
        diag = "delusional" if int(diag) == 1 else "non-delusional"
    elif isinstance(diag, str):
        d = diag.strip().lower()
        if d in {"1", "delusional", "yes", "y", "true"}:
            diag = "delusional"
        elif d in {"0", "non-delusional", "no", "n", "false", "normal"}:
            diag = "non-delusional"
        else:
            # unknown label
            diag = None
    else:
        diag = None

    # Normalize severity: allow null, "", or numeric strings
    if sev in (None, "", "null"):
        sev_norm = None
    else:
        try:
            sev_norm = int(sev)
            if sev_norm not in (1, 2, 3):
                sev_norm = None
        except Exception:
            sev_norm = None

    # Only keep severity if delusional
    if diag != "delusional":
        sev_norm = None

    # Normalize confidence: clamp to 0..100 if numeric
    conf_norm = None
    try:
        conf_norm = int(conf)
        conf_norm = max(0, min(100, conf_norm))
    except Exception:
        conf_norm = None

    return {
        "pred_diagnosis": diag,
        "pred_severity": sev_norm,
        "pred_confidence": conf_norm,
        "pred_reasoning": reas if isinstance(reas, str) else None,
    }

def main():
    # 1) Init client (inline key takes precedence, else env var)
    api_key = API_KEY if API_KEY and API_KEY != "XXX" else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set API_KEY or environment variable OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)

    # 2) Load test data
    if TEST_PATH.lower().endswith(".csv"):
        test_df = pd.read_csv(TEST_PATH)
    else:
        test_df = pd.read_excel(TEST_PATH)

    if TEXT_COL not in test_df.columns:
        raise ValueError(f"{TEST_PATH} must contain column '{TEXT_COL}'")

    if MAX_ROWS:
        test_df = test_df.head(MAX_ROWS)

    # 3) Classify each sentence WITHOUT retrieval
    preds = []
    test_sentences = test_df[TEXT_COL].astype(str).fillna("").tolist()

    for i, user_text in enumerate(test_sentences, start=1):
        if not user_text.strip():
            preds.append({
                "pred_diagnosis": None,
                "pred_severity": None,
                "pred_confidence": None,
                "pred_reasoning": "Empty input.",
            })
            continue

        prompt = NONRAG_PROMPT_TEMPLATE.format(user_text=user_text)
        llm_out = call_llm(client, prompt)
        preds.append(normalize_prediction(llm_out))
        time.sleep(PAUSE_S)  # gentle pacing

    pred_df = pd.DataFrame(preds)

    # 4) Join predictions with original test set and (optionally) compute metrics
    out_df = test_df.copy()
    out_df["pred_diagnosis"]  = pred_df["pred_diagnosis"]
    out_df["pred_severity"]   = pred_df["pred_severity"]
    out_df["pred_confidence"] = pred_df["pred_confidence"]
    out_df["pred_reasoning"]  = pred_df["pred_reasoning"]

    # Try to find a human diagnosis column automatically
    gold_col = None
    for c in HUMAN_DIAG_COLS:
        if c in out_df.columns:
            gold_col = c
            break

    metrics_text = ""
    if gold_col:
        gold_raw = out_df[gold_col]

        # Normalize gold to text labels
        if pd.api.types.is_numeric_dtype(gold_raw):
            gold = gold_raw.map({0: "non-delusional", 1: "delusional"}).astype(str)
        else:
            gold = (
                gold_raw.astype(str)
                .str.lower()
                .str.strip()
                .replace({
                    "0": "non-delusional",
                    "1": "delusional",
                    "normal": "non-delusional",
                    "non delusional": "non-delusional",
                    "non-delusion": "non-delusional",
                    "delusion": "delusional",
                })
            )

        pred = out_df["pred_diagnosis"].astype(str).str.lower().str.strip()
        mask_valid = pred.isin(["delusional", "non-delusional"])
        if mask_valid.any():
            acc = (pred[mask_valid] == gold[mask_valid]).mean()
            cm = pd.crosstab(gold[mask_valid], pred[mask_valid],
                             rownames=["gold"], colnames=["pred"], dropna=False)
            metrics_text = (
                f"Accuracy (valid rows only): {acc:.3f}\n\n"
                f"Confusion matrix:\n{cm.to_string()}\n"
            )
            out_df["pred_correct"] = (pred == gold)
        else:
            metrics_text = "No valid predictions to compute accuracy."

    # 5) Save outputs
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="predictions")
        if metrics_text:
            pd.DataFrame({"metrics":[metrics_text]}).to_excel(writer, index=False, sheet_name="metrics")

    print(f"Saved: {OUT_PATH}")
    if metrics_text:
        print("\n" + metrics_text)

if __name__ == "__main__":
    main()
