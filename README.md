# ReviewPulse — Product Review Sentiment Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46e3b7?style=for-the-badge&logo=render)](https://sentiment-analysis-dashboard-veb9.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%20CPU-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

ReviewPulse is a data science portfolio project designed to classify the sentiment (Positive, Neutral, Negative) of product reviews and visualize key business metrics in an interactive web dashboard.

The application utilizes a **FastAPI backend**, a **cached SQLite database**, a **Chart.js frontend dashboard**, and a **fine-tuned BERT-Tiny transformer** — optimized for deployment on memory-constrained free-tier cloud infrastructure.

> 🔴 **Live Deployment:** [https://sentiment-analysis-dashboard-veb9.onrender.com](https://sentiment-analysis-dashboard-veb9.onrender.com)

---

## 1. Problem Statement

E-commerce businesses receive thousands of customer reviews daily. Manual inspection at scale is impossible. ReviewPulse automates customer feedback analysis by:

1. Classifying review text into three sentiment classes: **Positive**, **Neutral**, and **Negative**.
2. Aggregating volumes over time to discover satisfaction trends.
3. Extracting top descriptive keywords per sentiment class to pinpoint product highlights or pain points.
4. Serving an interactive **"Try it yourself"** live classification tester.

---

## 2. Dataset Spec & Mapping

| Field | Value |
|---|---|
| **Source** | Hugging Face `mteb/amazon_reviews_multi` (English subset) |
| **Sample Size** | 3,000 reviews — stratified (1,000 per class) |
| **Stars → Positive** | Ratings 4–5 |
| **Stars → Neutral** | Rating 3 |
| **Stars → Negative** | Ratings 1–2 |

### Mapping Limitations
Using star ratings as ground-truth sentiment labels is a convenient simplification with real-world caveats:
- **Subjective Ratings:** Users may write critical text yet leave 4 stars, or glowing text with 3 stars.
- **Noisy Neutral:** 3-star reviews are notoriously ambiguous — ranging from mild praise to mild disappointment — rather than a clear neutral signal.

---

## 3. Modeling Approach

### Fine-tuned BERT-Tiny (Memory-Optimized for Cloud Deployment)

The production model is **`prajjwal1/bert-tiny`** (4.4M parameters, ~17.5MB weights), fine-tuned for 5 epochs on the 3,000-review training set using the Hugging Face `Trainer` API.

| Property | Value |
|---|---|
| **Base Model** | `prajjwal1/bert-tiny` |
| **Parameters** | 4.4M |
| **Model Size** | ~17.5 MB |
| **Training Hardware** | CPU (local) |
| **Training Time** | ~51 seconds / 5 epochs |
| **Output Labels** | `LABEL_0` → negative · `LABEL_1` → neutral · `LABEL_2` → positive |

This model was specifically chosen to fit within Render's **512MB free-tier RAM limit**. Standard models like `distilbert-base-uncased` (66M params) or `lxyuan/distilbert-base-multilingual-cased-sentiments-student` (~540MB at runtime) exceeded this limit and caused OOM crashes.

### Memory Architecture Decisions

| Component | Decision | Reason |
|---|---|---|
| PyTorch Build | `torch==2.4.0+cpu` (CPU-only wheel) | Avoids 2GB+ NVIDIA CUDA packages pulled in by default `torch` |
| NumPy | `numpy<2.0.0` | Prevents binary incompatibility with torch 2.4's C extensions |
| Transformer Library | `transformers>=4.30.0,<5.0.0` | Stable API compatible with torch 2.4 |
| Thread Limit | `torch.set_num_threads(1)` | Prevents CPU thread explosion on single-core containers |
| Model Loading | Eager warm-up at **server startup** (lifespan event) | Prevents cold-start OOM crash on first `/predict` request |

### Performance Optimizations (Batch Inference & SQLite Caching)

- **High-Performance Batch Inference:** The `/analyze` endpoint sends all 3,000 reviews to the pipeline in batches (`batch_size=32`). This reduces inference time from ~6 minutes (sequential) to **under 40 seconds on CPU**.
- **Precomputed Caching:** Inference runs once. Results (sentiment counts, time series trends, TF-IDF keywords) are serialized as JSON and stored in SQLite. The dashboard reads from cache instantly on every subsequent load.

---

## 4. Evaluation and Results

### Quantitative Metrics

Evaluated on a hold-out test set of 500 stratified reviews:

| Metric | Value |
|---|---|
| **Overall Accuracy** | **55.8%** |
| **Training Epochs** | 5 |
| **Training Time** | ~51 seconds (CPU) |

#### Per-Class Classification Report

```
              precision    recall  f1-score   support

    Negative       0.52      0.65      0.58       167
     Neutral       0.59      0.16      0.25       166
    Positive       0.58      0.86      0.70       167
```

**Key Takeaway:** The **Neutral** class exhibits the lowest F1-score (0.25) due to its inherent ambiguity from 3-star rating noise. The model achieves strong recall on **Positive** reviews (0.86) and solid performance on **Negative** reviews (0.65). Performance is appropriate for a 4.4M-parameter model trained in under one minute.

#### Confusion Matrix

The confusion matrix is saved to `evaluation/confusion_matrix.png`. It highlights:
- Actual positive reviews are rarely misclassified as negative.
- Neutral reviews are heavily misclassified as positive or negative, confirming the star-rating ambiguity hypothesis.

### Qualitative Error Analysis

A manual inspection of misclassified reviews (detailed in `evaluation/misclassified_examples.md`) reveals three common failure modes:

1. **Mixed Sentiment:** *"The packaging was crushed, but the product inside works perfectly."* (Label: Neutral; Predicted: Positive). The model weighted "perfectly" higher than "crushed".
2. **Ambiguous 3-Star:** *"It's okay. Nothing special."* (Label: Neutral; Predicted: Negative). "Nothing special" carries negative valence in word embeddings but describes average quality.
3. **Sarcasm / Tone:** *"Thanks for sending a broken charger."* (Label: Negative; Predicted: Positive). Without contextual depth, "Thanks" misleads the classifier.

---

## 5. System Architecture & Repo Structure

```
sentiment-analysis-dashboard/
├── README.md
├── reviews.db                   # SQLite database (cached metrics + review records)
├── Dockerfile                   # Docker configuration (python:3.11-slim)
├── docker-compose.yml           # Docker Compose configuration
├── data/
│   └── reviews_sample.csv       # Stratified sample dataset (3,000 rows)
├── notebooks/
│   └── 01_analysis_and_eval.ipynb  # EDA and metrics Jupyter Notebook
├── app/
│   ├── main.py                  # FastAPI app — endpoints + startup model warm-up
│   ├── sentiment_utils.py       # Model loader & inference helpers (BERT-Tiny)
│   ├── keyword_utils.py         # TF-IDF keyword extraction
│   └── requirements.txt         # CPU-only pinned dependencies
├── evaluation/
│   ├── confusion_matrix.png     # Heatmap plot of confusion matrix
│   ├── classification_report.txt   # Detailed per-class report
│   └── misclassified_examples.md   # Qualitative error analysis
├── models/
│   └── sentiment_model/         # Fine-tuned BERT-Tiny weights (safetensors)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── scripts/
    ├── download_data.py         # Ingests and samples HF reviews
    ├── train_and_eval.py        # Fine-tunes BERT-Tiny + generates eval reports
    ├── generate_notebook.py     # Generates Jupyter Notebook programmatically
    └── verify_app.py            # Integration verification tests
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend dashboard (index.html) |
| `GET` | `/status` | Returns model + cache readiness status |
| `POST` | `/analyze` | Runs batch inference on all 3,000 reviews, caches results |
| `POST` | `/predict` | Classifies a single review text in real-time |
| `GET` | `/dashboard-data` | Returns precomputed dashboard metrics from SQLite cache |

---

## 6. How to Run Locally

### 1. Set Up Environment & Ingest Data

```bash
# Clone the repository
git clone https://github.com/dakeshav2028/Sentiment-Analysis-Dashboard.git
cd Sentiment-Analysis-Dashboard

# Install CPU-only dependencies
pip install -r app/requirements.txt

# Download and sample the Amazon product reviews dataset
python scripts/download_data.py
```

### 2. Fine-tune the Model & Generate Evaluation Reports

```bash
# Fine-tune BERT-Tiny on the 3,000 reviews (completes in ~1 minute on CPU)
python scripts/train_and_eval.py
```

This saves the fine-tuned model to `models/sentiment_model/` and writes evaluation artifacts to `evaluation/`.

### 3. Launch the Backend & Dashboard

```bash
# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at `http://127.0.0.1:8000`. If you see **"Analysis Pending"**, click the **Run Batch Analysis** button in the header. The page will process all 3,000 reviews in under 40 seconds and populate the interactive charts.

### 4. Run via Docker

```bash
# Build and start the containerized application
docker-compose up --build
```

Access the dashboard at `http://127.0.0.1:8000`.

---

## 7. Deployment Notes (Render Free Tier)

The application is deployed on **Render's free tier** (512MB RAM). Key constraints overcome:

- **OOM Fix #1:** Replaced the default `torch` pip package (which pulls in ~2GB of NVIDIA CUDA libraries) with the `torch==2.4.0+cpu` CPU-only wheel (~200MB).
- **OOM Fix #2:** Migrated from `distilbert-base-uncased` (66M params, ~540MB runtime) to `prajjwal1/bert-tiny` (4.4M params, ~17.5MB) to stay within the memory budget.
- **Cold-Start Fix:** Added a FastAPI `lifespan` startup event that eagerly loads and warm-ups the model when the server boots, preventing the first `/predict` request from triggering an OOM crash.
- **NumPy Pinning:** Pinned `numpy<2.0.0` to avoid binary incompatibility with torch's C extensions.

---

## 8. Future Improvements

- **Larger Model on Paid Tier:** Upgrade to `distilbert-base-uncased` fine-tuned for 3 epochs on a paid instance (≥1GB RAM) for significantly higher accuracy.
- **Aspect-Based Sentiment Analysis (ABSA):** Break down sentiment per product feature (e.g., *shipping*, *build quality*, *price*) for richer business insights.
- **Fine-grained Sentiment:** Classify into 5 classes (1–5 stars) rather than compressing into 3.
- **Streaming Inference:** Replace batch pre-computation with a streaming queue (e.g., Celery + Redis) for real-time review ingestion.
- **Authentication:** Add API key or OAuth2 protection for the `/analyze` admin endpoint.
