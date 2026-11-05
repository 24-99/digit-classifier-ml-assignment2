"""
Train and evaluate five classifiers on the UCI Optical Recognition of
Handwritten Digits dataset (64 features, 1797 instances, 10 classes).

Run from the project root:
    python model/train_models.py

Outputs
-------
model/<name>.joblib      one fitted pipeline per model
model/metrics.csv        evaluation metrics on the held-out test split
test_data.csv            held-out test split (features + 'target' column)
train_data.csv           training split (for reference only)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

# file-name -> display name (shared with app.py)
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


def build_models() -> dict:
    """Return an unfitted pipeline for each of the five required models."""
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "decision_tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=2,
                                           random_state=SEED)),
        ]),
        "knn": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "naive_bayes": Pipeline([
            ("clf", GaussianNB(var_smoothing=1e-6)),
        ]),
        "svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=SEED)),
        ]),
    }


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Six required metrics. Multi-class -> macro averages, one-vs-rest AUC."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    digits = load_digits(as_frame=True)
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    # Save the splits (test_data.csv is what gets uploaded into the Streamlit app)
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    train_df.to_csv(ROOT / "train_data.csv", index=False)

    rows = []
    for key, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{key}.joblib")
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        m = compute_metrics(y_test, y_pred, y_proba)
        rows.append({"ML Model Name": MODEL_NAMES[key], **m})

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
