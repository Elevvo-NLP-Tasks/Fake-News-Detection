import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from wordcloud import WordCloud


def format_time(elapsed_seconds):
    """Takes a time in seconds and returns hh:mm:ss format."""
    return str(datetime.timedelta(seconds=int(round(elapsed_seconds))))


def evaluate_model(y_true, y_pred, y_prob=None, model_name="Classifier"):
    """
    Compute key classification metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
    Returns metrics dict and prints a formatted summary.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f1_macro": f1_macro,
        "roc_auc": auc
    }

    print(f"\n==================== Evaluation: {model_name} ====================")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    if auc is not None:
        print(f"  ROC-AUC   : {auc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real (0)", "Fake (1)"], digits=4))
    print("==================================================================\n")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Plot and save an aesthetically styled confusion matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = ["Real News", "Fake News"]
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1.5,
        linecolor="#f0f0f0",
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def generate_wordclouds(df, save_dir="Assets"):
    """
    Generate and save high-resolution Word Clouds for Fake vs. Real news articles,
    as well as a side-by-side comparison figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Generating Word Clouds for Fake and Real news...")
    
    # Extract cleaned/filtered text for each class
    text_fake = " ".join(df[df['label'] == 1]['clean_content'].dropna().sample(n=min(5000, len(df)), random_state=42))
    text_real = " ".join(df[df['label'] == 0]['clean_content'].dropna().sample(n=min(5000, len(df)), random_state=42))

    # Configuration for word cloud visuals
    wc_kwargs = {
        "width": 1200,
        "height": 700,
        "background_color": "#121212",
        "max_words": 150,
        "contour_width": 1,
        "contour_color": "#444444",
        "random_state": 42
    }

    wc_fake = WordCloud(colormap="magma", **wc_kwargs).generate(text_fake)
    wc_real = WordCloud(colormap="viridis", **wc_kwargs).generate(text_real)

    # Save individual word clouds
    fake_path = os.path.join(save_dir, "wordcloud_fake.png")
    real_path = os.path.join(save_dir, "wordcloud_real.png")

    wc_fake.to_file(fake_path)
    wc_real.to_file(real_path)
    print(f"Saved: {fake_path}")
    print(f"Saved: {real_path}")

    # Generate side-by-side comparison visual
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#121212')
    
    axes[0].imshow(wc_fake, interpolation='bilinear')
    axes[0].set_title("Fake News — Most Common Terms", fontsize=16, color='#ff5555', pad=15, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(wc_real, interpolation='bilinear')
    axes[1].set_title("Real News — Most Common Terms", fontsize=16, color='#50fa7b', pad=15, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(save_dir, "wordclouds_comparison.png")
    plt.savefig(comparison_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Saved comparison word cloud to {comparison_path}")
