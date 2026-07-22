import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_top_keywords(df: pd.DataFrame, text_column: str = "review_text", sentiment_column: str = "sentiment", top_n: int = 15):
    """
    Extract the top TF-IDF keywords for each sentiment group.
    Returns:
        dict: { 'positive': [{'word': str, 'score': float}, ...], 'neutral': ..., 'negative': ... }
    """
    sentiments = ['positive', 'neutral', 'negative']
    result = {}
    
    # Custom additional stopwords to filter out common product review noise
    custom_stopwords = {
        'product', 'item', 'bought', 'buy', 'ordered', 'order', 'got', 'get', 'use', 
        'used', 'using', 'review', 'amazon', 'star', 'stars', 'just', 'like', 'really',
        'would', 'one', 'time', 'box', 'day', 'days', 'package', 'shipping', 'received'
    }
    
    # Combined english + custom stopwords
    from sklearn.feature_extraction import text
    stop_words = list(text.ENGLISH_STOP_WORDS.union(custom_stopwords))
    
    for sentiment in sentiments:
        # Get texts for this sentiment
        texts = df[df[sentiment_column] == sentiment][text_column].fillna("").tolist()
        
        if len(texts) == 0:
            result[sentiment] = []
            continue
            
        try:
            # We want single words (unigrams)
            vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=100, ngram_range=(1, 1))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Sum tf-idf scores for each term across all documents in this group
            sums = tfidf_matrix.sum(axis=0)
            
            # Map terms to their sums
            data = []
            words = vectorizer.get_feature_names_out()
            for col_idx, term in enumerate(words):
                data.append({
                    "word": term,
                    "score": float(sums[0, col_idx])
                })
                
            # Sort by score descending
            data.sort(key=lambda x: x['score'], reverse=True)
            result[sentiment] = data[:top_n]
            
        except Exception as e:
            print(f"[KEYWORDS] Error extracting keywords for {sentiment}: {e}")
            result[sentiment] = []
            
    return result
