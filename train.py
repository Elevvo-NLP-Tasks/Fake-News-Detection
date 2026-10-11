import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from utils import evaluate_model, plot_confusion_matrix, format_time


def vectorize_text(X_train, X_test, vec_config):
    """
    Fits TF-IDF vectorizer on training text and transforms both train and test sets.
    """
    print("Vectorizing corpus using TF-IDF...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        max_features=vec_config.get("max_features", 10000),
        ngram_range=tuple(vec_config.get("ngram_range", [1, 2])),
        sublinear_tf=vec_config.get("sublinear_tf", True),
        min_df=vec_config.get("min_df", 3)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    elapsed = time.time() - start_time
    print(f"TF-IDF Vectorization completed in {format_time(elapsed)}.")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")
    print(f"X_train_vec shape: {X_train_vec.shape}")
    print(f"X_test_vec shape:  {X_test_vec.shape}\n")
    return vectorizer, X_train_vec, X_test_vec


def train_logistic_regression(X_train_vec, y_train, X_test_vec, y_test, lr_config, assets_dir):
    """
    Trains and evaluates a Logistic Regression classifier.
    """
    print("------------------ Training Logistic Regression ------------------")
    start_time = time.time()
    lr_model = LogisticRegression(
        C=lr_config.get("C", 1.0),
        max_iter=lr_config.get("max_iter", 1000),
        solver=lr_config.get("solver", "lbfgs"),
        random_state=42
    )
    lr_model.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    print(f"Logistic Regression trained in {format_time(train_time)}.")

    y_pred = lr_model.predict(X_test_vec)
    y_prob = lr_model.predict_proba(X_test_vec)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="Logistic Regression")
    metrics["train_time"] = format_time(train_time)

    # Plot Confusion Matrix
    cm_path = os.path.join(assets_dir, "confusion_matrix_logistic_regression.png")
    plot_confusion_matrix(y_test, y_pred, "Logistic Regression", cm_path)

    return lr_model, metrics


def train_svm(X_train_vec, y_train, X_test_vec, y_test, svm_config, assets_dir):
    """
    Trains and evaluates a Linear SVM classifier with probability calibration.
    """
    print("---------------------- Training Linear SVM -----------------------")
    start_time = time.time()
    base_svm = LinearSVC(
        C=svm_config.get("C", 1.0),
        max_iter=svm_config.get("max_iter", 2000),
        random_state=42
    )
    # CalibratedClassifierCV allows producing calibrated probability estimates for SVM
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    print(f"Support Vector Machine trained in {format_time(train_time)}.")

    y_pred = calibrated_svm.predict(X_test_vec)
    y_prob = calibrated_svm.predict_proba(X_test_vec)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="Linear SVM (Calibrated)")
    metrics["train_time"] = format_time(train_time)

    # Plot Confusion Matrix
    cm_path = os.path.join(assets_dir, "confusion_matrix_svm.png")
    plot_confusion_matrix(y_test, y_pred, "Linear SVM", cm_path)

    return calibrated_svm, metrics


def save_artifacts(models_dir, vectorizer, lr_model, svm_model):
    """
    Serializes trained models and vectorizer to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    svm_path = os.path.join(models_dir, "svm_model.joblib")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(lr_model, lr_path)
    joblib.dump(svm_model, svm_path)

    print(f"Artifacts successfully saved to '{models_dir}':")
    print(f"  • Vectorizer : {vec_path}")
    print(f"  • Logistic Regression : {lr_path}")
    print(f"  • SVM Model : {svm_path}\n")


def train_all(X_train, y_train, X_test, y_test, config):
    """
    Executes complete vectorization, model training, evaluation, and serialization.
    """
    models_dir = config.get("models_dir", "models")
    assets_dir = config.get("assets_dir", "Assets")

    # 1. Vectorize
    vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test, config["vectorizer"])

    # 2. Train Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(
        X_train_vec, y_train, X_test_vec, y_test, config["logistic_regression"], assets_dir
    )

    # 3. Train Support Vector Machine
    svm_model, svm_metrics = train_svm(
        X_train_vec, y_train, X_test_vec, y_test, config["svm"], assets_dir
    )

    # 4. Save Models & Vectorizer
    save_artifacts(models_dir, vectorizer, lr_model, svm_model)

    # 5. Summary comparison table
    summary_df = pd.DataFrame([lr_metrics, svm_metrics])
    print("=================== 🏆 Model Comparison Summary ===================")
    for idx, row in summary_df.iterrows():
        print(f"Model: {row['model']:<25} | Accuracy: {row['accuracy']*100:.2f}% | F1-Score: {row['f1_score']*100:.2f}% | Time: {row['train_time']}")
    print("===================================================================\n")

    return {
        "vectorizer": vectorizer,
        "lr_model": lr_model,
        "svm_model": svm_model,
        "metrics": {"logistic_regression": lr_metrics, "svm": svm_metrics}
    }
