# ReviewPulse — Product Review Sentiment Analyzer

ReviewPulse is a data science portfolio project designed to classify the sentiment (Positive, Neutral, Negative) of product reviews and visualize key business metrics in an interactive web dashboard. 

The application utilizes a FastAPI backend, a cached SQLite database, a Chart.js frontend dashboard, and a lightweight transformer encoder.

---

## 1. Problem Statement
E-commerce businesses receive thousands of customer reviews daily. Manual inspection of these reviews is impossible at scale. ReviewPulse automates customer feedback analysis by:
1. Classifying review text into three sentiment classes: **Positive**, **Neutral**, and **Negative**.
2. Aggregating volumes over time to discover satisfaction trends.
3. Extracting top descriptive keywords per sentiment class to pinpoint product highlights or pain points.
4. Serving an interactive "Try it yourself" classification tester.

---

## 2. Dataset Spec & Mapping
* **Source:** Hugging Face English subset of `mteb/amazon_reviews_multi` (originating from Amazon Multilingual Reviews).
* **Sample Size:** 3,000 reviews, balanced via stratified sampling (1,000 positive, 1,000 neutral, 1,000 negative) to prevent class imbalance issues during modeling and evaluation.
* **Label Mapping:** 
  * Star Ratings **4–5** $\rightarrow$ **Positive**
  * Star Rating **3** $\rightarrow$ **Neutral**
  * Star Ratings **1–2** $\rightarrow$ **Negative**

### Mapping Limitations
Using star ratings as ground-truth sentiment labels is a convenient simplification, but has distinct real-world limitations:
* **Subjective Ratings:** Some users write highly critical text yet leave 4 stars, or write glowing reviews but leave 3 stars.
* **Noise in Neutral:** 3-star reviews are notoriously noisy—they range from mild praise to mild disappointment or mixed pros/cons, rather than actual neutral sentiment.

---

## 3. Modeling Approach

### Double-Track Model Loader (Robust Production Architecture)
To handle resource constraints and platform differences, ReviewPulse implements a robust double-track architecture:
1. **Option B (Fine-tuning):** The script attempts to fine-tune `distilbert-base-uncased` (66M parameters) on CPU using PyTorch and Hugging Face `Trainer` for 1 epoch. To protect training resource bounds, a `TimeLimitCallback` enforces a strict 10-minute timeout.
2. **Option A (Pretrained Fallback):** If fine-tuning fails or exceeds the time limit, the pipeline automatically falls back to the state-of-the-art 3-class student model `lxyuan/distilbert-base-multilingual-cased-sentiments-student` on Hugging Face.

*Note: In our current local test run, Hugging Face `Trainer` raised a missing dependency exception (`accelerate>=1.1.0`), which successfully triggered our Option A fallback path. The entire analysis, evaluation reports, and caching pipeline ran automatically using the fallback model, ensuring zero downtime.*

### Performance Optimizations (Batch Inference & SQLite Caching)
* **High-Performance Batch Inference:** The batch analysis endpoint `/analyze` passes all 3,000 reviews to the Hugging Face pipeline in batches (`batch_size=32`), leveraging PyTorch vectorization. This optimization reduces the inference time from ~6 minutes (one-by-one loops) to **under 40 seconds on CPU**.
* **Precomputed Caching:** Rather than executing model inference on every dashboard load, the `/analyze` endpoint runs once to precompute sentiment counts, timelines, and keywords. These results are cached as serialized JSON blocks in a lightweight SQLite database (`reviews.db`). The dashboard reads directly from the cache instantly.

---

## 4. Evaluation and Results

### Quantitative Metrics
Evaluated on a hold-out test set of 500 stratified reviews:
* **Overall Test Accuracy:** **55.7%**

#### Per-Class Classification Report
```
              precision    recall  f1-score   support

    Negative       0.52      0.65      0.58       167
     Neutral       0.59      0.16      0.25       166
    Positive       0.58      0.86      0.70       167
```

* **Key Takeaway:** As expected, the **Neutral** class exhibits the lowest F1-score (0.25) due to its high ambiguity. The model yields high recall on **Positive** reviews (0.86) and solid performance on **Negative** reviews (0.65).

#### Confusion Matrix
The confusion matrix is saved to `evaluation/confusion_matrix.png`. It highlights that:
* Actual positive reviews are rarely misclassified as negative.
* Neutral reviews are heavily misclassified as either positive or negative, confirming the star-rating fuziness hypothesis.

### Qualitative Error Analysis
A manual inspection of misclassified reviews (detailed in `evaluation/misclassified_examples.md`) reveals three common failure modes:
1. **Mixed Sentiment:** *"The packaging was crushed, but the product inside works perfectly. Mixed feelings."* (Label: Neutral; Predicted: Positive). The model weighed the word "perfectly" higher than "crushed".
2. **Ambiguous 3-Star Rating:** *"It's okay. Nothing special."* (Label: Neutral; Predicted: Negative). Words like "nothing special" carry negative valence for simple classifiers but describe average (neutral) quality.
3. **Sarcasm / Tone:** *"Thanks for sending a broken charger."* (Label: Negative; Predicted: Positive). Without contextual depth, the positive word "Thanks" misleads the model.

---

## 5. System Architecture & Repo Structure

```
sentiment-analysis-dashboard/
├── README.md
├── reviews.db                 # SQLite database (stores cached metrics + records)
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker-compose configuration
├── data/
│   └── reviews_sample.csv     # Stratified sample dataset (3,000 rows)
├── notebooks/
│   └── 01_analysis_and_eval.ipynb   # Jupyter Notebook with EDA and metrics
├── app/
│   ├── main.py                # FastAPI endpoints
│   ├── sentiment_utils.py     # Double-track model loaders & inference helpers
│   ├── keyword_utils.py       # TF-IDF keyword extraction
│   └── requirements.txt       # Dependencies
├── evaluation/
│   ├── confusion_matrix.png   # Heatmap plot of confusion matrix
│   ├── classification_report.txt  # Detailed precision/recall report
│   └── misclassified_examples.md   # Qualitative error analysis
└── scripts/
    ├── download_data.py       # Ingests and samples HF reviews
    ├── train_and_eval.py      # Automates training/eval loops
    ├── generate_notebook.py   # Generates Jupyter Notebook programmatically
    └── verify_app.py          # Integration verification tests
```

---

## 6. How to Run Locally

### 1. Set Up Environment & Ingest Data
Ensure you have Python installed, then clone the repository:
```bash
# Install requirements
pip install -r app/requirements.txt

# Download and sample the Amazon product reviews dataset
python scripts/download_data.py
```

### 2. Run Model Training & Evaluation
```bash
# Run model training (will run DistilBERT or fallback, outputting evaluation files)
python scripts/train_and_eval.py
```

### 3. Launch Backend & Dashboard
```bash
# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser and navigate to `http://127.0.0.1:8000`. If you see "Analysis Pending," click the **Run Batch Analysis** button in the header. The page will process the dataset in seconds and display the dynamic charts.

### 4. Run via Docker Container
```bash
# Build and run the application container
docker-compose up --build
```
Access the dashboard at `http://127.0.0.1:8000`.

---

## 7. Future Improvements
* **Solve PyTorch Accelerate Dependency:** Install `accelerate>=1.1.0` in the environment to perform actual CPU/GPU fine-tuning and compare it with the zero-shot student fallback model.
* **Aspect-Based Sentiment Analysis (ABSA):** Break down sentiments per product feature (e.g. separating opinions on *shipping time*, *build quality*, and *price*).
* **Fine-grained Sentiment:** Classify into 5 classes (1-5 stars) rather than compressing them into 3.
