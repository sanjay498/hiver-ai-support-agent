# Hiver AI Customer Support Agent (`hiver-ai-support-agent`)

A production-ready, fully reproducible AI customer support agent trained and evaluated on historical customer interactions from the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`), focused on **AmazonHelp**.

---

## Highlights

- **Data-Driven Brand Selection:** Empirically verified selection of `AmazonHelp` (highest dialogic depth, 1,997 reconstructed conversation pairs).
- **Discovered Intent Taxonomy:** 12-intent customer service taxonomy discovered via unsupervised clustering and semantic profiling.
- **Strict Leakage Prevention:** Golden evaluation conversations are strictly partitioned and excluded from the FAISS vector index.
- **Conservative Escalation Engine:** Multi-layer safety architecture (Risk Keywords, Mandatory Policy Matrix, Confidence Gates, and Retrieval Thresholds) yielding a **4.1% Unsafe Auto Rate**.
- **Comprehensive Evaluation & Judge Calibration:** Reproducible baselines (Majority, TF-IDF + Logistic Regression, Main AI), LLM-as-a-judge (1–5 rubric), and human agreement benchmark (MAE = 0.670, 86.3% within-1 agreement).
- **Dual Mode (Online + Deterministic Offline):** Fully functional with an OpenAI API key or completely offline with deterministic semantic fallbacks.
- **Streamlit Demo UI:** Interactive browser interface with pre-loaded edge case scenarios and evidence inspection.

---

## 1. Quickstart & 15-Minute Reproduction

### Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### Setup Commands
```bash
# 1. Clone repository & enter workspace
git clone <repo-url> hiver-ai-support-agent
cd hiver-ai-support-agent

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (Optional: add OPENAI_API_KEY for LLM mode)
cp .env.example .env
```

### Reproduce Data & Evaluation (Under 10 Minutes)
```bash
# 1. Download dataset sample (~35,000 rows, ~5.4 MB, takes ~2 seconds)
python -m src.data.download --sample-size 35000

# 2. Reconstruct conversations and clean threads
python -m src.data.clean

# 3. Discover intent taxonomy
python -m src.intent.discover

# 4. Generate golden evaluation dataset (200 stratified items)
python -m src.data.create_golden_set

# 5. Build FAISS vector index (with leakage prevention filter)
python -c "
import json
from src.retrieval.retriever import HistoricalRetriever
with open('data/processed/conversations.json') as f:
    convs = json.load(f)
retriever = HistoricalRetriever()
retriever.build_index(convs, save=True)
"

# 6. Run comprehensive automated evaluation & generate confusion matrices
python -m src.evaluation.evaluate
```

---

## 2. Launch the Streamlit Demo UI

Run the interactive web interface:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`. Evaluators can test custom customer queries or select pre-loaded scenarios:
- **Routine Delivery Delay** (Safe Auto-Handle)
- **Urgent Cancellation Request** (Safe Auto-Handle)
- **Return / Exchange Procedure** (Safe Auto-Handle)
- **Explicit Legal Threat** (Immediate Escalate)
- **Compromised Account Alert** (Strict Escalate)
- **Fraudulent Card Overcharge** (Strict Escalate)

---

## 3. Running Unit & Integration Tests

Execute the complete pytest suite:
```bash
pytest -v
```
All 19 tests cover:
- Data cleaning and thread reconstruction
- Intent output schema and confidence bounds
- Baselines (Majority and TF-IDF + Logistic Regression)
- FAISS vector indexing, retrieval, and similarity sorting
- **Strict Leakage Prevention** (verifies zero overlap between golden set IDs and retrieval index)
- Safety-critical escalation logic
- Grounded response generation
- End-to-end pipeline execution

---

## 4. Empirical Evaluation Results Summary

Evaluated over **200 stratified golden examples** (`data/golden/golden_set.json`):

### 4.1 Intent Classification Comparison
| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Baseline A (Majority Class)** | 8.0% | 0.012 | 0.012 |
| **Baseline B (TF-IDF + Logistic Regression)** | **62.5%** | **0.609** | **0.631** |
| **Main AI System (Semantic Few-Shot)** | 47.5% | 0.482 | 0.462 |

### 4.2 Escalation Safety Performance
| Metric | Value | Meaning |
|---|---|---|
| **Escalation Recall** | **95.9%** (47 / 49) | Catches 96% of hazardous/escalation cases |
| **Unsafe Auto Rate** | **4.1%** (2 / 49) | Hazardous cases incorrectly automated |
| **Escalation Precision** | **24.9%** (47 / 189) | Over-escalates benign queries to protect safety |
| **Escalation F1** | **0.395** | Deliberate conservative safety bias |

### 4.3 Retrieval Quality
- **Recall@1:** 77.5%
- **Recall@3:** 80.5%
- **Recall@5:** 82.0%

### 4.4 Human-vs-Judge Agreement Study (45 Samples)
- **Overall MAE:** 0.670
- **Within-1 Agreement Rate:** 86.3%
- **Notable Limitation:** LLM judge exhibits *Politeness Bias*, rating generic replies 5/5 on Helpfulness where human annotators rate them 2/5.

---

## 5. Repository Structure

```
hiver-ai-support-agent/
├── README.md                      # Reproduction & operational documentation
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git exclusion rules
├── data/
│   ├── raw/                       # Downloaded twcs_sample.csv
│   ├── processed/                 # conversations.json, intents.json, faiss_index.bin
│   └── golden/                    # golden_set.json (200 items), human_validation_subset.json
├── src/
│   ├── data/
│   │   ├── download.py            # Streaming / Kaggle dataset acquisition
│   │   ├── explore_brands.py      # Empirical brand comparison analysis
│   │   ├── clean.py               # Preprocessing orchestration
│   │   ├── threads.py             # Conversation thread reconstruction
│   │   └── create_golden_set.py   # Golden evaluation set curation
│   ├── intent/
│   │   ├── discover.py            # Data-driven intent clustering
│   │   ├── baseline.py            # Majority & TF-IDF + Logistic Regression baselines
│   │   └── classifier.py          # AI Intent Classifier with schema validation
│   ├── retrieval/
│   │   ├── embeddings.py          # Sentence-transformers wrapper
│   │   └── retriever.py           # FAISS index with leakage prevention filter
│   ├── generation/
│   │   ├── prompts.py             # Anti-hallucination prompt templates
│   │   └── response_generator.py  # Grounded social support reply generator
│   ├── escalation/
│   │   └── policy.py              # Conservative safety-first escalation engine
│   ├── evaluation/
│   │   ├── evaluate.py            # Comprehensive evaluation pipeline & plots
│   │   ├── baselines.py           # Baseline evaluation runner
│   │   ├── judge.py               # LLM-as-a-judge 6-dimension rubric
│   │   └── human_agreement.py     # Human agreement study & bias analysis
│   └── pipeline.py                # Unified end-to-end pipeline
├── app/
│   └── streamlit_app.py           # Interactive Streamlit demo
├── tests/
│   ├── test_data_processing.py    # Preprocessing tests
│   ├── test_classifier.py         # Classifier & baseline tests
│   ├── test_retrieval.py          # FAISS retriever tests
│   ├── test_leakage.py            # Zero-leakage verification tests
│   ├── test_escalation.py         # Escalation safety tests
│   ├── test_generation.py         # Response generator tests
│   └── test_pipeline.py           # End-to-end integration tests
├── notebooks/
│   └── exploration.ipynb          # Exploratory Data Analysis & visual walkthrough
└── report/
    ├── report.md                  # Comprehensive 6-page evaluation report
    └── figures/                   # Generated confusion matrices & evaluation plots
```

---

## 6. Detailed Engineering Report

For the in-depth technical discussion, empirical tables, top 5 failure mode breakdowns, the mandatory **"What is misleading about my headline number?"** analysis, and 12-point decision log, see:
👉 **[report/report.md](report/report.md)**
