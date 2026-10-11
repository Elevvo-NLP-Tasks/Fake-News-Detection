"""
╔══════════════════════════════════════════════════════════════════════╗
║                📰 Fake News Detection System                        ║
║                    Interactive Inference CLI                         ║
╚══════════════════════════════════════════════════════════════════════╝

Classify news articles as REAL or FAKE using trained Machine Learning
classifiers (Logistic Regression & Linear SVM) and TF-IDF representations.
"""

import os
import sys
import json
import joblib
import numpy as np
from data_preprocess import clean_text, preprocess_text, build_preprocessor

# ─────────────────────────────────────────────────────────────────────
# 🎨 Terminal colours & UI helpers
# ─────────────────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

DIVIDER = f"{DIM}{'─' * 70}{RESET}"


# ─────────────────────────────────────────────────────────────────────
# 📚 Curated news bank – diverse real-world articles for quick testing
# ─────────────────────────────────────────────────────────────────────
CURATED_ARTICLES = [
    {
        "type": "Real",
        "title": "U.S. House passes landmark spending bill to avoid government shutdown",
        "text": (
            "The U.S. House of Representatives voted on Thursday to approve a comprehensive "
            "spending package funding federal agencies through the end of the fiscal year. "
            "Lawmakers passed the bipartisan legislation with overwhelming support, sending "
            "the measure to the Senate ahead of the midnight deadline. Congressional leaders "
            "praised the compromise as a victory for fiscal stability and national defense."
        )
    },
    {
        "type": "Fake",
        "title": "SHOCKING: Secret underground bunker found containing millions of counterfeit ballots",
        "text": (
            "Patriots have uncovered an unbelievable conspiracy! Anonymous military sources "
            "confirm that an abandoned warehouse concealed a deep subterranean bunker filled "
            "with millions of pre-marked fraudulent ballots shipped from overseas. Corrupt "
            "politicians and mainstream media are frantically covering up this bombshell revelation!"
        )
    },
    {
        "type": "Real",
        "title": "NASA rover confirms evidence of ancient riverbed on Mars surface",
        "text": (
            "NASA's Perseverance rover has captured compelling sedimentary evidence demonstrating "
            "that a fast-moving, deep river once flowed into Jezero Crater on Mars. Scientists "
            "analyzing the latest high-resolution rock scans observed curved layers of coarse grain "
            "and cobbles that suggest active water systems shaped Martian geology billions of years ago."
        )
    },
    {
        "type": "Fake",
        "title": "Pope Francis shocks world by endorsing presidential candidate during private mass",
        "text": (
            "In an unprecedented historic announcement that stunned the Vatican and global religious "
            "leaders, Pope Francis openly endorsed a political candidate yesterday. Insiders report "
            "that Vatican bishops were left speechless as the Pontiff made the controversial "
            "endorsement live on undisclosed radio broadcasts before official broadcasts were cut."
        )
    },
    {
        "type": "Real",
        "title": "European Central Bank leaves benchmark interest rates unchanged amid cooling inflation",
        "text": (
            "The European Central Bank maintained its benchmark interest rates at historic levels on Thursday, "
            "signaling that price pressures across the eurozone are continuing to ease. ECB President Christine Lagarde "
            "noted in a press conference that while core inflation has slowed, policymakers will remain data-dependent."
        )
    },
    {
        "type": "Fake",
        "title": "Scientists develop microchip that cures all human diseases within 24 hours",
        "text": (
            "A rogue group of rebellious biochemists has secretly created a miracle quantum chip capable of "
            "eradicating every known illness in less than one day. Big pharma executives are panicking and trying to ban "
            "the device worldwide, but courageous freedom fighters have leaked the blueprints across alternative media."
        )
    }
]


def load_model_artifacts(models_dir="models"):
    """Loads TF-IDF vectorizer and trained models."""
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    svm_path = os.path.join(models_dir, "svm_model.joblib")

    for p in [vec_path, lr_path, svm_path]:
        if not os.path.exists(p):
            print(f"{RED}Error: Model artifact not found at {p}. Please run main.py first!{RESET}")
            sys.exit(1)

    vectorizer = joblib.load(vec_path)
    lr_model = joblib.load(lr_path)
    svm_model = joblib.load(svm_path)
    return vectorizer, lr_model, svm_model


def make_confidence_bar(prob_fake, bar_length=25):
    """Renders an ASCII percentage progress bar for fake confidence."""
    filled = int(round(prob_fake * bar_length))
    empty = bar_length - filled
    bar = f"{'█' * filled}{'░' * empty}"
    return bar


def explain_prediction(text_processed, vectorizer, lr_model, top_n=5):
    """Extracts top discriminating TF-IDF terms in the text based on Logistic Regression weights."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]
    tokens = text_processed.split()
    
    # Find matching indices in vocabulary
    vocab = vectorizer.vocabulary_
    present_indices = [vocab[w] for w in tokens if w in vocab]
    if not present_indices:
        return []

    present_indices = list(set(present_indices))
    scores = [(feature_names[i], coefs[i]) for i in present_indices]
    # Sort by absolute influence
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return scores[:top_n]


def print_banner():
    banner = f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║                📰 Fake News Detection System                        ║
║                 Interactive Prediction Interface                     ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}
{DIM}Model options: Logistic Regression & Calibrated Linear Support Vector Machine{RESET}
"""
    print(banner)


def display_curated_menu():
    print(f"\n{YELLOW}{BOLD}Curated Articles Bank:{RESET}")
    for idx, item in enumerate(CURATED_ARTICLES, 1):
        type_badge = f"{GREEN}[REAL]{RESET}" if item["type"] == "Real" else f"{RED}[FAKE]{RESET}"
        print(f"  {BOLD}[{idx}]{RESET} {type_badge} {item['title'][:65]}...")
    print(f"  {BOLD}[0]{RESET} {CYAN}Enter Custom Article / Headline{RESET}")
    print(f"  {BOLD}[q]{RESET} {DIM}Quit{RESET}")


def run_prediction(title, text, vectorizer, lr_model, svm_model, lemmatizer, stop_words):
    full_content = (title + " " + text).strip()
    processed = preprocess_text(full_content, lemmatizer, stop_words)

    if not processed.strip():
        print(f"\n{RED}Error: Article is empty after preprocessing!{RESET}")
        return

    vec = vectorizer.transform([processed])

    # 1. Logistic Regression
    lr_prob_fake = lr_model.predict_proba(vec)[0][1]
    lr_pred = lr_model.predict(vec)[0]

    # 2. Linear SVM (Calibrated)
    svm_prob_fake = svm_model.predict_proba(vec)[0][1]
    svm_pred = svm_model.predict(vec)[0]

    print(f"\n{DIVIDER}")
    print(f"{BOLD}Article Analysis Results:{RESET}")
    print(f"{DIVIDER}")
    
    # Primary Verdict (Consensus or SVM)
    is_fake = (svm_pred == 1)
    if is_fake:
        verdict = f"{RED}{BOLD}🚨 FAKE NEWS DETECTED{RESET}"
    else:
        verdict = f"{GREEN}{BOLD}✅ VERIFIED REAL NEWS{RESET}"
    
    print(f"\nOverall Verdict: {verdict}\n")

    # Detailed model breakdown
    print(f"{BOLD}Model Probabilities & Confidence:{RESET}")
    
    # Logistic Regression output
    lr_bar = make_confidence_bar(lr_prob_fake)
    lr_badge = f"{RED}Fake{RESET}" if lr_pred == 1 else f"{GREEN}Real{RESET}"
    print(f"  • {BOLD}Logistic Regression:{RESET} [{lr_bar}] {lr_prob_fake * 100:.1f}% Fake  ({lr_badge})")

    # SVM output
    svm_bar = make_confidence_bar(svm_prob_fake)
    svm_badge = f"{RED}Fake{RESET}" if svm_pred == 1 else f"{GREEN}Real{RESET}"
    print(f"  • {BOLD}Linear SVM (Calib):{RESET}  [{svm_bar}] {svm_prob_fake * 100:.1f}% Fake  ({svm_badge})")

    # Top influential keywords
    influential = explain_prediction(processed, vectorizer, lr_model)
    if influential:
        print(f"\n{BOLD}Key Linguistic Indicators detected:{RESET}")
        for word, weight in influential:
            bias = f"{RED}pushes Fake (+{weight:.2f}){RESET}" if weight > 0 else f"{GREEN}pushes Real ({weight:.2f}){RESET}"
            print(f"  - '{BOLD}{word}{RESET}': {bias}")

    print(f"{DIVIDER}\n")


def interactive_cli():
    print_banner()
    print(f"{DIM}Loading models from disk...{RESET}")
    vectorizer, lr_model, svm_model = load_model_artifacts()
    lemmatizer, stop_words = build_preprocessor()
    print(f"{GREEN}✓ Models & Vectorizer loaded successfully!{RESET}")

    while True:
        display_curated_menu()
        choice = input(f"\n{BOLD}Select an option (0-6, q): {RESET}").strip()

        if choice.lower() == 'q':
            print(f"\n{CYAN}Goodbye! 👋{RESET}\n")
            break

        if choice == '0':
            print(f"\n{CYAN}--- Custom Article Input ---{RESET}")
            title = input(f"{BOLD}Article Title (optional, press Enter to skip): {RESET}").strip()
            print(f"{BOLD}Enter or paste article text (press Enter when done):{RESET}")
            text = input("> ").strip()
            if not text and not title:
                print(f"{YELLOW}Input was empty. Returning to menu.{RESET}")
                continue
            run_prediction(title, text, vectorizer, lr_model, svm_model, lemmatizer, stop_words)

        elif choice.isdigit() and 1 <= int(choice) <= len(CURATED_ARTICLES):
            idx = int(choice) - 1
            article = CURATED_ARTICLES[idx]
            print(f"\n{CYAN}Selected: {BOLD}{article['title']}{RESET}")
            print(f"{DIM}Ground Truth Label: {article['type']}{RESET}")
            run_prediction(article['title'], article['text'], vectorizer, lr_model, svm_model, lemmatizer, stop_words)

        else:
            print(f"{RED}Invalid selection. Please choose an option from the menu.{RESET}")


if __name__ == "__main__":
    interactive_cli()
