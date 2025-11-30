# Detection of Induced Psychosis-Like Thinking in User–GenAI Interaction  
### Code and Data Repository for:  
**Huang, Y. (2025). *Detection of Induced Psychosis-Like Thinking in User–GenAI Interaction: A Retrieval-Augmented Generation (RAG) Approach*.**

---

## 1. Repository Contents

This repository contains all code and data used in the study, organised into four parts:

---

### **File 1 — Data Simulation**
Folder: `data_simulation/`

Contains:
- **definition.txt**  
  A text file describing the DSM-5–based definitions of delusion and hallucination used to guide data generation.
- **simulation_script.py**  
  Python script used to simulate the initial 200 user statements (50% delusional, 50% non-delusional).
- **chatgpt_delusion_dataset.xlsx**  
  The simulation dataset output

---

### **File 2 — Validated Dataset**
Folder: `validated_dataset/`

Contains:
- **validated_dataset.xlsx**  
  The full set of 200 statements after revision and verification by a licensed psychotherapist.  

---

### **File 3 — RAG test**
Folder: `rag_test/`

Contains:
- **rag script.py** — RAG classifier (retrieves top-3 semantically similar examples before classification)  
- **non-rag test.py** — baseline classifier (no retrieval)  
- **test.xlsx** — test set used to evaluate accuracy, precision, recall, and confusion matrices
- **rag.xlsx** — the knowledge resource data for RAG
- **test_with_predictions.xlsx** — RAG method prediction results saved as Excel files for analysis 
- **test_with_predictions_noRAG.xlsx** — baseline model prediction results saved as Excel files for analysis

---

### **File 4 — Statistical Analysis (Jamovi)**
Folder: `jamovi_analysis/`

Contains:
- **severity_tests.omv**  
  Jamovi file used to compute:
  - descriptive statistics  
  - Wilcoxon signed-rank tests  
  - paired t-tests  
  for comparing prediction accuracy and severity estimation between the RAG and baseline models.

---

## 2. Citation

If you use any part of this repository (data, code, or analysis workflow), please cite the following paper:

**Huang, Y. (2025, November 21). Detection of Induced Psychosis-Like Thinking in User–GenAI Interaction: A Retrieval-Augmented Generation (RAG) Approach. https://doi.org/10.31234/osf.io/fghrp_v1**


