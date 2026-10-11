import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = ['stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)


def load_data(fake_path, true_path):
    """
    Load Fake and True news datasets, merge them with binary labels,
    and inspect dataframe statistics.
    
    Label Convention:
      0 = REAL news (from True.csv)
      1 = FAKE news (from Fake.csv)
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news dataset not found at {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Real news dataset not found at {true_path}")

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Assign binary labels
    df_fake['label'] = 1
    df_fake['label_name'] = 'Fake'
    df_true['label'] = 0
    df_true['label_name'] = 'Real'

    # Concatenate and shuffle
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Combine title and text content
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = (df['title'] + " " + df['text']).str.strip()

    # Drop exact duplicate content
    initial_count = len(df)
    df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
    duplicates_dropped = initial_count - len(df)

    print("================ Data Loaded & Filtered ================")
    print(f"Total articles loaded: {len(df)} (Duplicates dropped: {duplicates_dropped})")
    print(f"Label distribution:\n{df['label_name'].value_counts()}")
    print("-------------------------------------------------------")
    print(df.info())
    print("========================================================\n")

    return df


def clean_text(text):
    """
    Clean raw text:
    - Lowercase conversion
    - Remove Reuters publisher dateline artifact (e.g., 'WASHINGTON (Reuters) - ')
    - Strip URLs, email addresses, and non-alphabetic characters
    - Collapse excessive whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove Reuters prefixes like "washington (reuters) -" or "(reuters) -"
    text = re.sub(r'^[a-z\s,/\(\)]*\(reuters\)\s*[-–—]\s*', ' ', text)
    text = re.sub(r'\breuters\b', ' ', text)

    # Remove URLs and links
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove HTML tags if present
    text = re.sub(r'<.*?>', ' ', text)

    # Keep only alphabetic characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_preprocessor():
    """Instantiate and return lemmatizer and stopword set."""
    ensure_nltk_resources()
    stop_words = set(stopwords.words('english'))
    # Extend stopwords with common news boilerplate words
    extra_stopwords = {'said', 'would', 'also', 'one', 'two', 'told', 'year', 'years', 'say'}
    stop_words.update(extra_stopwords)
    lemmatizer = WordNetLemmatizer()
    return lemmatizer, stop_words


def preprocess_text(text, lemmatizer, stop_words):
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes a single text string.
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(processed_tokens)


def preprocess_corpus(texts, show_progress=True):
    """
    Preprocess an iterable of text documents with lemmatization and stopword removal.
    """
    lemmatizer, stop_words = build_preprocessor()
    iterator = tqdm(texts, desc="Preprocessing Text", unit="doc") if show_progress else texts
    return [preprocess_text(t, lemmatizer, stop_words) for t in iterator]


def prepare_dataset(config):
    """
    End-to-end dataset preparation:
    - Loads Fake and Real news data
    - Preprocesses title + text content (cleaning, stopwords, lemmatization)
    - Splits into stratified train and test sets
    """
    df = load_data(config["fake_path"], config["true_path"])

    print("Preprocessing content (cleaning, stopword removal, lemmatization)...")
    df['clean_content'] = preprocess_corpus(df['content'], show_progress=True)

    # Filter out any rare rows that become empty after cleaning
    valid_mask = df['clean_content'].str.strip().str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_content'],
        df['label'],
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=df['label']
    )

    print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples\n")
    return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    import json
    with open("config.json") as f:
        conf = json.load(f)
    X_train, X_test, y_train, y_test, df = prepare_dataset(conf)
    print("Data preprocessing module test completed successfully.")
