"""
═══════════════════════════════════════════════════════════════════════
           📰 Fake News Detection — Training Pipeline
═══════════════════════════════════════════════════════════════════════
End-to-end execution:
  1. Load and inspect Kaggle Fake and Real news datasets
  2. Generate Word Clouds (Fake vs. Real common terms)
  3. Preprocess text (clean, remove stopwords, lemmatize)
  4. TF-IDF vectorization
  5. Train & Evaluate Logistic Regression & SVM classifiers
  6. Plot Confusion Matrices and serialize trained models
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
from data_preprocess import prepare_dataset, ensure_nltk_resources
from train import train_all
from utils import generate_wordclouds, format_time


def main():
    total_start = time.time()
    print("=" * 70)
    print("           🚀 FAKE NEWS DETECTION PIPELINE LAUNCHED")
    print("=" * 70 + "\n")

    # 1. Load Configuration
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Ensure NLTK resources
    ensure_nltk_resources()

    # 3. Data Preprocessing & Train/Test Split
    X_train, X_test, y_train, y_test, df = prepare_dataset(config)

    # 4. Generate Word Clouds (Bonus Task)
    assets_dir = config.get("assets_dir", "Assets")
    generate_wordclouds(df, save_dir=assets_dir)

    # 5. Train & Evaluate Models
    results = train_all(X_train, y_train, X_test, y_test, config)

    # 6. Summary Banner
    total_time = time.time() - total_start
    print("=" * 70)
    print("       🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"       ⏱️ Total Pipeline Time: {format_time(total_time)}")
    print(f"       📂 Models saved in: {config.get('models_dir', 'models')}/")
    print(f"       📊 Visualizations in: {assets_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
