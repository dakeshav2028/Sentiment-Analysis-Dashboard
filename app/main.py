import os
import sqlite3
import json
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.sentiment_utils import classify_sentiment, get_sentiment_pipeline
from app.keyword_utils import extract_top_keywords

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the sentiment model at startup to avoid first-request OOM crashes."""
    print("[STARTUP] Pre-loading sentiment model...")
    get_sentiment_pipeline()
    print("[STARTUP] Model loaded and ready.")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="ReviewPulse - Product Review Sentiment Analyzer",
    description="FastAPI backend with cached sentiment analysis metrics and SQLite persistence.",
    lifespan=lifespan
)

# Allow cross-origin requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "reviews.db"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reviews_sample.csv")

class TextPayload(BaseModel):
    text: str

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Table for reviews
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            review_text TEXT,
            stars INTEGER,
            ground_truth_sentiment TEXT,
            predicted_sentiment TEXT,
            confidence REAL,
            review_date TEXT
        )
    """)
    
    # Table for precomputed dashboard cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_cache (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.get("/status")
def get_status():
    """Check if the dataset is analyzed and cached."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reviews")
    review_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT value FROM dashboard_cache WHERE key = 'precomputed_metrics'")
    cache_exists = cursor.fetchone() is not None
    conn.close()
    
    return {
        "status": "ready" if (review_count > 0 and cache_exists) else "pending_analysis",
        "review_count": review_count,
        "cache_cached": cache_exists
    }

@app.post("/analyze")
def run_batch_analysis():
    """
    Run batch sentiment inference on the 3,000 reviews dataset,
    generate random date distribution, extract keywords via TF-IDF,
    and cache the metrics in SQLite.
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=404, 
            detail=f"Sample dataset not found at {DATA_PATH}. Please run download_data.py first."
        )
        
    print("[BACKEND] Loading sample reviews...")
    df = pd.read_csv(DATA_PATH)
    
    # Initialize SQLite database
    conn = get_db()
    cursor = conn.cursor()
    
    # Clear existing reviews
    cursor.execute("DELETE FROM reviews")
    cursor.execute("DELETE FROM dashboard_cache")
    conn.commit()
    
    print("[BACKEND] Running batch inference (3,000 reviews)...")
    texts = df['review_text'].fillna("").tolist()
    
    from app.sentiment_utils import get_sentiment_pipeline
    nlp = get_sentiment_pipeline()
    
    print("[BACKEND] Running batch model predictions...")
    results = nlp(texts, batch_size=32, truncation=True, max_length=128)
    
    reviews_to_insert = []
    base_date = datetime.now()
    
    for idx, row in df.iterrows():
        review_text = str(row['review_text'])
        stars = int(row['stars'])
        gt_sentiment = str(row['sentiment'])
        r_id = str(row['id'])
        
        res = results[idx]
        label = res['label'].lower()
        confidence = float(res['score'])
        
        if label == 'positive' or label == 'label_2':
            pred_sentiment = 'positive'
        elif label == 'neutral' or label == 'label_1':
            pred_sentiment = 'neutral'
        elif label == 'negative' or label == 'label_0':
            pred_sentiment = 'negative'
        else:
            pred_sentiment = 'neutral'
            
        random_days = random.randint(0, 30)
        review_date = (base_date - timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        reviews_to_insert.append((
            r_id,
            review_text,
            stars,
            gt_sentiment,
            pred_sentiment,
            confidence,
            review_date
        ))
        
    # Insert reviews in bulk
    cursor.executemany("""
        INSERT INTO reviews (id, review_text, stars, ground_truth_sentiment, predicted_sentiment, confidence, review_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, reviews_to_insert)
    conn.commit()
    
    print("[BACKEND] Precomputing dashboard metrics...")
    
    # Retrieve updated dataset from SQLite for metrics calculation
    db_df = pd.read_sql_query("SELECT * FROM reviews", conn)
    
    # 1. Overall counts (Predicted vs Ground Truth)
    pred_counts = db_df['predicted_sentiment'].value_counts().to_dict()
    gt_counts = db_df['ground_truth_sentiment'].value_counts().to_dict()
    
    # Ensure all keys exist
    for c in ['positive', 'neutral', 'negative']:
        pred_counts[c] = int(pred_counts.get(c, 0))
        gt_counts[c] = int(gt_counts.get(c, 0))
        
    # 2. Volume trends over time
    # Group by date and predicted sentiment
    trend_group = db_df.groupby(['review_date', 'predicted_sentiment']).size().unstack(fill_value=0)
    
    # Reindex to make sure all sentiment columns exist
    for col in ['positive', 'neutral', 'negative']:
        if col not in trend_group.columns:
            trend_group[col] = 0
            
    trend_group = trend_group[['positive', 'neutral', 'negative']]
    trend_group = trend_group.sort_index()
    
    volume_trends = {
        "dates": trend_group.index.tolist(),
        "positive": [int(x) for x in trend_group['positive'].tolist()],
        "neutral": [int(x) for x in trend_group['neutral'].tolist()],
        "negative": [int(x) for x in trend_group['negative'].tolist()]
    }
    
    # 3. TF-IDF Keyword Extraction based on predicted sentiment
    print("[BACKEND] Extracting keywords using TF-IDF...")
    keywords = extract_top_keywords(db_df, text_column="review_text", sentiment_column="predicted_sentiment", top_n=15)
    
    # 4. Accuracy metrics (simplified confusion matrix and accuracy score)
    correct_preds = (db_df['predicted_sentiment'] == db_df['ground_truth_sentiment']).sum()
    accuracy = float(correct_preds / len(db_df))
    
    precomputed = {
        "summary": {
            "total_reviews": len(db_df),
            "accuracy": accuracy,
            "predicted_distribution": pred_counts,
            "ground_truth_distribution": gt_counts
        },
        "trends": volume_trends,
        "keywords": keywords
    }
    
    # Save to Cache table in SQLite
    cursor.execute(
        "INSERT OR REPLACE INTO dashboard_cache (key, value) VALUES (?, ?)",
        ("precomputed_metrics", json.dumps(precomputed))
    )
    conn.commit()
    conn.close()
    
    print("[BACKEND] Batch analysis and caching completed successfully.")
    return {
        "status": "success",
        "message": "Sample reviews analyzed and metrics cached.",
        "metrics": precomputed['summary']
    }

@app.post("/predict")
def predict_single_review(payload: TextPayload):
    """Predict sentiment for a single user-submitted review text."""
    result = classify_sentiment(payload.text)
    return result

@app.get("/dashboard-data")
def get_dashboard_data():
    """Retrieve precomputed dashboard metrics from SQLite cache."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM dashboard_cache WHERE key = 'precomputed_metrics'")
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {
            "status": "pending_analysis",
            "message": "Dashboard data has not been precomputed. Call /analyze to run batch processing."
        }
        
    return json.loads(row['value'])

# Mount static files folder
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def read_root():
    """Serve index.html at root."""
    index_file = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "ReviewPulse Sentiment Dashboard Backend is running. Frontend static files folder not found."}
