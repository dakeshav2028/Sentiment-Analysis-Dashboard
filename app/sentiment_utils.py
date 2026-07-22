import os
import torch
torch.set_num_threads(1)
from transformers import pipeline

# Global model and pipeline references
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
        
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_model")
    fallback_flag = os.path.join(os.path.dirname(__file__), "..", "models", "fallback_active.txt")
    
    fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Check if fallback is explicitly active or fine-tuned model does not exist
    use_fallback = False
    model_path = model_dir
    
    if os.path.exists(fallback_flag):
        use_fallback = True
        with open(fallback_flag, "r") as f:
            model_path = f.read().strip()
        print(f"[SENTIMENT] Fallback flag found. Using pretrained model: {model_path}")
    elif not os.path.exists(os.path.join(model_dir, "config.json")):
        use_fallback = True
        model_path = fallback_model
        print(f"[SENTIMENT] Fine-tuned model config not found. Defaulting to pretrained model: {model_path}")
    else:
        print(f"[SENTIMENT] Loading fine-tuned model from: {model_path}")
        
    try:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            device=-1  # Force CPU
        )
        print("[SENTIMENT] Model loaded successfully.")
    except Exception as e:
        print(f"[SENTIMENT] Error loading model from {model_path}: {e}")
        if not use_fallback:
            print(f"[SENTIMENT] Attempting emergency fallback to: {fallback_model}")
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                tokenizer=fallback_model,
                device=-1
            )
            print("[SENTIMENT] Emergency fallback model loaded successfully.")
        else:
            raise e
            
    return _sentiment_pipeline

def classify_sentiment(text: str):
    """
    Classify the sentiment of a single review text.
    Returns:
        dict: {
            'sentiment': 'positive' | 'neutral' | 'negative',
            'confidence': float (0.0 to 1.0)
        }
    """
    nlp = get_sentiment_pipeline()
    # Handle empty text
    if not text or not text.strip():
        return {"sentiment": "neutral", "confidence": 1.0}
        
    try:
        # Run inference
        result = nlp(text, truncation=True, max_length=128)[0]
        label = result['label'].lower()
        score = float(result['score'])
        
        # Map label names
        # Fallback model (lxyuan) uses 'positive', 'neutral', 'negative'
        # Fine-tuned model uses 'label_0', 'label_1', 'label_2'
       if label in ('positive', 'label_2'):
    mapped_label = 'positive'
elif label in ('negative', 'label_0'):
    mapped_label = 'negative'
elif label in ('neutral', 'label_1'):
    mapped_label = 'neutral'
else:
    mapped_label = 'neutral'
            
        return {
            "sentiment": mapped_label,
            "confidence": score
        }
    except Exception as e:
        print(f"[SENTIMENT] Error during inference: {e}")
        return {"sentiment": "neutral", "confidence": 0.0}
