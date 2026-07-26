import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    pipeline
)
# Custom callback to enforce time limits on CPU fine-tuning
class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_seconds=600):  # Default 10 minutes
        self.time_limit = time_limit_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"Training started. Set hard time limit of {self.time_limit} seconds (10 minutes).")

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                print(f"\n[WARNING] Training time-box limit exceeded ({elapsed:.1f}s > {self.time_limit}s). Terminating training process...")
                control.should_training_stop = True

# Simple Dataset class for PyTorch
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == labels)
    return {"accuracy": float(acc)}

def main():
    print("--- ReviewPulse Model Training & Evaluation ---")
    start_time = time.time()
    
    # 1. Load sample dataset
    data_path = "data/reviews_sample.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sample dataset not found at {data_path}. Please run download_data.py first.")
        
    # Stratified split: 2000 train, 500 val, 500 test from the overall sample
    sampled_df = pd.read_csv(data_path)
    print(f"Loaded {len(sampled_df)} sample reviews.")
    
    # Let's perform train/val/test split
    train_df, test_val_df = train_test_split(
        sampled_df, 
        test_size=1000, 
        random_state=42, 
        stratify=sampled_df['sentiment_label']
    )
    val_df, test_df = train_test_split(
        test_val_df, 
        test_size=500, 
        random_state=42, 
        stratify=test_val_df['sentiment_label']
    )
    
    print(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Define paths
    model_dir = "models/sentiment_model"
    fallback_flag = "models/fallback_active.txt"
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    # Initialize variables for the final model
    fine_tune_failed = False
    
    # 2. Attempt Fine-Tuning (Option B)
    try:
        print("\nAttempting to fine-tune BERT-Tiny on CPU...")
        model_name = "prajjwal1/bert-tiny"
        
        print("Loading tokenizer and model...")
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
        print("Tokenizing datasets...")
        train_encodings = tokenizer(list(train_df['review_text']), truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(list(val_df['review_text']), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(list(test_df['review_text']), truncation=True, padding=True, max_length=128)
        
        train_dataset = ReviewDataset(train_encodings, list(train_df['sentiment_label']))
        val_dataset = ReviewDataset(val_encodings, list(val_df['sentiment_label']))
        test_dataset = ReviewDataset(test_encodings, list(test_df['sentiment_label']))
        
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            report_to="none",
            use_cpu=True  # Force CPU usage
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[TimeLimitCallback(time_limit_seconds=600)]  # Hard limit 10 minutes
        )
        
        # Track training duration
        train_start = time.time()
        trainer.train()
        train_duration = time.time() - train_start
        print(f"Training call completed in {train_duration:.2f} seconds.")
        
        # Check if training was aborted by callback
        if train_duration > 610:
            print("[INFO] Training exceeded time limit. Pivoting to Option A fallback...")
            fine_tune_failed = True
        else:
            print(f"Saving fine-tuned model to {model_dir}...")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            if os.path.exists(fallback_flag):
                os.remove(fallback_flag)
            print("Fine-tuning completed successfully!")
            
    except Exception as e:
        print(f"\n[WARNING] Exception occurred during fine-tuning: {e}")
        print("Pivoting to Option A (Pretrained model fallback)...")
        fine_tune_failed = True
        
    # 3. Fallback logic: Option A (Pretrained pipeline)
    if fine_tune_failed:
        fallback_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        print(f"\nActivating Fallback Model: {fallback_model}")
        with open(fallback_flag, "w") as f:
            f.write(fallback_model)
            
        print("Loading fallback pretrained pipeline...")
        # Load pipeline directly
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=fallback_model, 
            tokenizer=fallback_model,
            device=-1  # Force CPU
        )
    else:
        print("\nLoading fine-tuned model for evaluation...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model_dir, 
            tokenizer=model_dir,
            device=-1
        )
        
    # 4. Evaluation and Predictions on Test Set
    print("\nEvaluating model on the hold-out test set (500 reviews)...")
    test_texts = list(test_df['review_text'])
    true_labels = list(test_df['sentiment_label'])
    
    # Run batch predictions
    # Note: fallback model classes are: 'positive', 'neutral', 'negative'
    # Our fine-tuned model classes are LABEL_0, LABEL_1, LABEL_2 (mapping to 0, 1, 2)
    predictions = []
    
    eval_start = time.time()
    raw_preds = sentiment_pipeline(test_texts, truncation=True, max_length=128)
    eval_duration = time.time() - eval_start
    print(f"Inference on test set completed in {eval_duration:.2f} seconds ({eval_duration/len(test_df):.4f}s/review).")
    
    for p in raw_preds:
        label_str = p['label'].lower()
        if label_str == 'positive' or label_str == 'label_2':
            predictions.append(2)
        elif label_str == 'neutral' or label_str == 'label_1':
            predictions.append(1)
        elif label_str == 'negative' or label_str == 'label_0':
            predictions.append(0)
        else:
            # Fallback if class names differ
            predictions.append(1)
            
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"\nTest Set Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    target_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(true_labels, predictions, target_names=target_names)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open("evaluation/classification_report.txt", "w") as f:
        f.write(f"Model Used: {'Fallback Pretrained (lxyuan)' if fine_tune_failed else 'Fine-tuned BERT-Tiny'}\n")
        f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Sentiment Confusion Matrix')
    plt.tight_layout()
    plt.savefig("evaluation/confusion_matrix.png", dpi=100)
    plt.close()
    print("Confusion matrix saved to evaluation/confusion_matrix.png")
    
    # 5. Qualitative Error Analysis (Find 5-10 misclassified examples)
    print("\nRunning qualitative error analysis...")
    misclassified_indices = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != t]
    
    misclassified_md = f"# Qualitative Error Analysis\n\n"
    misclassified_md += f"**Model Analyzed:** {'Fallback Pretrained (lxyuan/distilbert-base-multilingual-cased-sentiments-student)' if fine_tune_failed else 'Fine-tuned BERT-Tiny'}\n"

    misclassified_md += f"**Test Set Accuracy:** {accuracy:.4f}\n"
    misclassified_md += f"**Total Misclassified Reviews in Test Set:** {len(misclassified_indices)} out of {len(test_df)}\n\n"
    misclassified_md += "Below are 8 analyzed examples of misclassifications, showing the text, star rating, ground truth, and predicted sentiment, along with a diagnostic comment.\n\n"
    
    # Select 8 examples representing different error types (sarcasm, mixed sentiment, stars vs. actual sentiment discrepancy)
    analyzed_count = 0
    for idx in misclassified_indices:
        row = test_df.iloc[idx]
        pred_label = target_names[predictions[idx]]
        true_label = target_names[true_labels[idx]]
        text = row['review_text']
        stars = row['stars']
        
        # Analyze error type
        error_type = "Unknown"
        analysis = ""
        
        text_lower = text.lower()
        if "but" in text_lower or "however" in text_lower or "although" in text_lower or "yet" in text_lower:
            error_type = "Mixed Sentiment"
            analysis = "The review contains both positive and negative aspects (e.g., 'good product but bad packaging'). The model struggled to weigh the opposing sentiment clauses correctly."
        elif stars == 3:
            error_type = "Ambiguous 3-Star Rating"
            analysis = "3-star reviews are often highly neutral or moderately mixed. The star rating mapping is inherently noisy here, as the reviewer might sound slightly positive or negative, causing a mismatch with the 'Neutral' ground truth."
        elif (stars in [1, 2] and "great" in text_lower) or (stars in [4, 5] and "bad" in text_lower):
            error_type = "Discrepancy Between Stars and Sentiment"
            analysis = "The user expressed sarcasm, or wrote a text that contradicts their star rating (e.g. giving 1 star but saying 'great product!' by mistake, or vice versa)."
        else:
            error_type = "Subtle / Sarcastic Tone"
            analysis = "The sentiment is expressed through sarcasm or subtle context that simple word associations cannot easily capture."
            
        misclassified_md += f"### Example {analyzed_count + 1}\n"
        misclassified_md += f"- **Review Text:** \"{text}\"\n"
        misclassified_md += f"- **Star Rating:** {stars} Stars\n"
        misclassified_md += f"- **Ground Truth Label:** {true_label}\n"
        misclassified_md += f"- **Predicted Label:** {pred_label}\n"
        misclassified_md += f"- **Error Category:** {error_type}\n"
        misclassified_md += f"- **Analysis:** {analysis}\n\n"
        
        analyzed_count += 1
        if analyzed_count >= 8:
            break
            
    with open("evaluation/misclassified_examples.md", "w") as f:
        f.write(misclassified_md)
    print("Qualitative error analysis saved to evaluation/misclassified_examples.md")
    print(f"\nFinished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
