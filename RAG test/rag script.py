# rag_run.py
# Purpose:
# - Use rag.xlsx as retrieval base (RAG)
# - Use test.xlsx as test set
# - For each row in test.xlsx, retrieve similar examples from rag.xlsx and ask GPT to classify
# - Save predictions to test_with_predictions.xlsx

import os, json, time, hashlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from openai import OpenAI

# ========= SETTINGS =========================================
API_KEY = "XXX"   # set your own OpenAI API key here

RAG_PATH   = "rag.xlsx"
TEST_PATH  = "test.xlsx"
OUT_PATH   = "test_with_predictions.xlsx"

TEXT_COL        = "sentence"          # both files use 'sentence'
HUMAN_DIAG_COLS = ["diagnosis", "dignosis"]  # script will auto-detect which exists
HUMAN_SEV_COL   = "severity"          # optional

MODEL_EMB  = "text-embedding-3-large"
MODEL_LLM  = "gpt-4o-mini"
TOP_K      = 3                        # retrieve 3 similar examples per test row
TEMP       = 0                        # deterministic
BATCH_EMB  = 64                       # batch size for embeddings
MAX_ROWS   = None                     # for quick dry runs (e.g., 50). Leave None to run all.
# =============================================================================

# ---- Helper: caching embeddings so you don't pay twice unnecessarily --------
def file_sig(path: str) -> str:
    st = os.stat(path)
    return f"{path}-{st.st_mtime_ns}-{st.st_size}"

def emb_cache_name(path: str) -> str:
    sig = hashlib.md5(file_sig(path).encode()).hexdigest()
    base = os.path.basename(path)
    return f".emb_cache_{base}_{sig}.npy"

def load_any_excel_or_csv(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of strings with batching + tiny pacing."""
    out = []
    for i in range(0, len(texts), BATCH_EMB):
        batch = texts[i:i+BATCH_EMB]
        resp = client.embeddings.create(model=MODEL_EMB, input=batch)
        out.extend([d.embedding for d in resp.data])
        time.sleep(0.05)  # gentle pacing
    return np.array(out, dtype=np.float32)

def retrieve_context(client: OpenAI, query_text: str, nn: NearestNeighbors,
                     rag_texts: list[str], top_k: int = TOP_K) -> list[str]:
    """Return top-k similar sentences from rag_texts for the given query."""
    q_emb = client.embeddings.create(model=MODEL_EMB, input=query_text).data[0].embedding
    q_emb = normalize(np.array([q_emb], dtype=np.float32))
    distances, idxs = nn.kneighbors(q_emb, n_neighbors=top_k, return_distance=True)
    idxs = idxs[0].tolist()
    return [rag_texts[j] for j in idxs]

def build_prompt(user_text: str, ctx_list: list[str]) -> str:
    ctx_str = "\n\n---\n".join(ctx_list)
    return f"""
You are a psychology content classifier.

[USER STATEMENT]
{user_text}

[RETRIEVED EXAMPLES] (similar statements from the corpus; may be imperfect)
{ctx_str}

Classify and return STRICT JSON with the following keys:
- "diagnosis": "1" (means "delusional") or "0" (means "non-delusional")
- "severity": null or 1/2/3 (1=mild, 2=moderate, 3=severe); only set if diagnosis is "delusional"
- "confidence": integer 0-100 indicating classifier confidence (not clinical certainty)
- "reasoning": one brief, neutral sentence explaining the decision (no PHI, no clinical claims)

Return ONLY the JSON.
"""

def call_llm(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
    )
    content = resp.choices[0].message.content.strip()
    # Robust JSON parse
    try:
        if content.startswith("{"):
            return json.loads(content)
        start = content.find("{"); end = content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
    except Exception:
        pass
    return {"raw": content}

def normalize_prediction(obj: dict) -> dict:
    diag = obj.get("diagnosis") if isinstance(obj, dict) else None

    # Normalize diagnosis to "delusional"/"non-delusional"
    if isinstance(diag, (int, float)):
        diag_norm = "delusional" if int(diag) == 1 else "non-delusional"
    elif isinstance(diag, str):
        d = diag.strip().lower()
        if d in {"1","delusional","yes","y","true"}:
            diag_norm = "delusional"
        elif d in {"0","non-delusional","no","n","false","normal"}:
            diag_norm = "non-delusional"
        else:
            diag_norm = None
    else:
        diag_norm = None

    # Normalize severity (only if delusional)
    sev = obj.get("severity") if isinstance(obj, dict) else None
    try:
        sev_norm = int(sev)
        if sev_norm not in (1,2,3):
            sev_norm = None
    except Exception:
        sev_norm = None
    if diag_norm != "delusional":
        sev_norm = None

    # Normalize confidence -> int 0..100
    try:
        conf_norm = int(obj.get("confidence"))
        conf_norm = max(0, min(100, conf_norm))
    except Exception:
        conf_norm = None

    reasoning = None
    if isinstance(obj, dict):
        reasoning = obj.get("reasoning") or obj.get("explanation")

    return {
        "pred_diagnosis": diag_norm,
        "pred_severity": sev_norm,
        "pred_confidence": conf_norm,
        "pred_reasoning": reasoning,
    }

def main():
    # 1) Init client with your inline API key
    client = OpenAI(api_key=API_KEY)

    # 2) Load data
    rag_df  = pd.read_excel("rag.xlsx")
    test_df = pd.read_excel("test.xlsx")

    # Basic checks
    if TEXT_COL not in rag_df.columns:
        raise ValueError(f"{RAG_PATH} must contain column '{TEXT_COL}'")
    if TEXT_COL not in test_df.columns:
        raise ValueError(f"{TEST_PATH} must contain column '{TEXT_COL}'")

    if MAX_ROWS:
        rag_df  = rag_df.head(MAX_ROWS)
        test_df = test_df.head(MAX_ROWS)
    
    # 3) Prepare RAG texts and cached embeddings
    rag_texts = rag_df[TEXT_COL].astype(str).fillna("").tolist()
    cache_path = emb_cache_name(RAG_PATH)
    if os.path.exists(cache_path):
        rag_embs = np.load(cache_path)
    else:
        rag_embs = embed_texts(client, rag_texts)
        np.save(cache_path, rag_embs)

    # Normalize for cosine similarity
    rag_embs_norm = normalize(rag_embs)

    # 4) Fit a simple cosine nearest-neighbor retriever (works fine on Windows)
    nn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine", algorithm="brute")
    nn.fit(rag_embs_norm)

    # 5) For each test row, retrieve context + classify
    preds = []
    test_sentences = test_df[TEXT_COL].astype(str).fillna("").tolist()

    for i, user_text in enumerate(test_sentences):
        if not user_text.strip():
            preds.append({
                "pred_diagnosis": None,
                "pred_severity": None,
                "pred_confidence": None,
                "pred_reasoning": "Empty input."
            })
            continue

        ctx = retrieve_context(client, user_text, nn, rag_texts, TOP_K)
        prompt = build_prompt(user_text, ctx)
        llm_out = call_llm(client, prompt)

        preds.append(normalize_prediction(llm_out))
        time.sleep(0.05)  # gentle pacing

    pred_df = pd.DataFrame(preds)

    # 6) prediction of the JSON

    # 6) Join predictions with original test set and (optionally) compute metrics
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
        # Map numeric labels 0/1 to text if needed
        # Normalize gold and predictions to consistent text labels
        gold = (
            out_df[gold_col].astype(str).str.lower().str.strip()
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
            metrics_text = f"Accuracy (valid rows only): {acc:.3f}\n\nConfusion matrix:\n{cm.to_string()}\n"
            out_df["pred_correct"] = (pred == gold)
        else:
            metrics_text = "No valid predictions to compute accuracy."

    # 7) Save outputs
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="predictions")
        if metrics_text:
            pd.DataFrame({"metrics":[metrics_text]}).to_excel(writer, index=False, sheet_name="metrics")

    print(f"Saved: {OUT_PATH}")
    if metrics_text:
        print("\n" + metrics_text)

if __name__ == "__main__":
    main()

    

