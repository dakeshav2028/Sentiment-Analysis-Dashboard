import os
import pandas as pd

def download_and_sample():
    print("Starting data ingestion from Hugging Face parquet branch...")
    # Direct parquet URL for English train split
    url = "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/refs%2Fconvert%2Fparquet/en/train/0000.parquet"
    
    try:
        df = pd.read_parquet(url)
        print(f"Successfully loaded raw dataset. Total rows: {len(df)}")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Attempting local check or fallback if available...")
        raise e

    # Columns: ['id', 'text', 'label', 'label_text']
    # label corresponds to stars 1-5, encoded as 0-4.
    # Map raw labels (0-4) to 1-5 star ratings for clarity
    df['stars'] = df['label'] + 1
    
    # Map star ratings to 3-class sentiment
    # Ratings 4-5 (labels 3, 4) -> Positive
    # Rating 3 (label 2) -> Neutral
    # Ratings 1-2 (labels 0, 1) -> Negative
    def get_sentiment_label(stars):
        if stars >= 4:
            return 2  # Positive
        elif stars == 3:
            return 1  # Neutral
        else:
            return 0  # Negative

    def get_sentiment_name(stars):
        if stars >= 4:
            return "positive"
        elif stars == 3:
            return "neutral"
        else:
            return "negative"

    df['sentiment_label'] = df['stars'].apply(get_sentiment_label)
    df['sentiment'] = df['stars'].apply(get_sentiment_name)
    
    print("\nInitial sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Stratified sampling: 1,000 samples per class to get 3,000 total samples
    print("\nPerforming stratified sampling (1,000 per class)...")
    sampled_dfs = []
    for label in [0, 1, 2]:
        sub_df = df[df['sentiment_label'] == label]
        sampled_sub = sub_df.sample(n=1000, random_state=42)
        sampled_dfs.append(sampled_sub)
        
    sampled_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Keep only relevant columns for our project
    output_df = sampled_df[['id', 'text', 'stars', 'sentiment_label', 'sentiment']]
    output_df = output_df.rename(columns={'text': 'review_text'})
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    output_path = "data/reviews_sample.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved sampled dataset of size {len(output_df)} to {output_path}")
    print("\nSample records:")
    print(output_df.head(3))
    
    print("\nValue counts in final sample:")
    print(output_df['sentiment'].value_counts())

if __name__ == "__main__":
    download_and_sample()
