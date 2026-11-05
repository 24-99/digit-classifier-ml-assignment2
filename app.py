"""
Streamlit app: compare five classifiers on the UCI handwritten-digits dataset.

Features
  a. CSV upload (test data)         d. Confusion matrix + classification report
  b. Model selection dropdown       e. Side-by-side comparison of all models
  c. Evaluation metrics             f. Sample-digit inspector
"""
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

st.set_page_config(page_title="Digit Classifier Lab", page_icon="🔢", layout="wide")


# --------------------------------------------------------------------------
# Model loading (falls back to retraining if the pickles don't match the
# scikit-learn version installed on the server)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models() -> dict:
    models, needs_retrain = {}, False
    for display, key in MODEL_FILES.items():
        try:
            models[display] = joblib.load(MODEL_DIR / f"{key}.joblib")
        except Exception:
            needs_retrain = True
            break
    if needs_retrain:
        import sys
        sys.path.insert(0, str(MODEL_DIR))
        from train_models import build_models  # noqa: WPS433

        train = pd.read_csv(ROOT / "train_data.csv")
        Xtr, ytr = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
        built = build_models()
        models = {}
        for display, key in MODEL_FILES.items():
            built[key].fit(Xtr, ytr)
            models[display] = built[key]
    return models


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    out = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    try:
        proba = model.predict_proba(X)
        classes = model.classes_
        if len(np.unique(y)) == len(classes):
            out["AUC"] = roc_auc_score(y, proba, multi_class="ovr",
                                       average="macro", labels=classes)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🔢 Digit Classifier Lab")
st.caption(
    "Five classical ML models on the UCI *Optical Recognition of Handwritten "
    "Digits* dataset (8x8 pixel images, 64 features, 10 classes). "
    "Upload the held-out test CSV, pick a model, and inspect its results."
)

models = load_models()

# --------------------------------------------------------------------------
# Sidebar: upload + model selection
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    st.caption(
        "The CSV needs the 64 pixel columns plus a `target` column. "
        "Use `test_data.csv` from the repo."
    )
    use_default = st.checkbox("No file handy? Use the bundled test set",
                              value=uploaded is None, disabled=uploaded is not None)

    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()), index=4)

# Resolve data source
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: {uploaded.name}"
elif use_default:
    df = pd.read_csv(ROOT / "test_data.csv")
    source = "Bundled test_data.csv"
else:
    st.info("Upload a CSV in the sidebar to get started.")
    st.stop()

# Validate columns
expected = list(models[model_name].feature_names_in_) \
    if hasattr(models[model_name], "feature_names_in_") else None
if TARGET_COL not in df.columns:
    st.error(f"The CSV must contain a `{TARGET_COL}` column with the true digit labels.")
    st.stop()
X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
if expected is not None:
    missing = [c for c in expected if c not in X.columns]
    if missing:
        st.error(f"Missing {len(missing)} feature columns, e.g. `{missing[0]}`. "
                 "Please upload a file in the same format as test_data.csv.")
        st.stop()
    X = X[expected]

c1, c2, c3 = st.columns(3)
c1.metric("Source", source.split(":")[0])
c2.metric("Rows", f"{len(df):,}")
c3.metric("Features", X.shape[1])

with st.expander("Preview data"):
    st.dataframe(df.head(20), width="stretch")

# --------------------------------------------------------------------------
# Selected model results
# --------------------------------------------------------------------------
st.subheader(f"Results: {model_name}")
model = models[model_name]
res = evaluate(model, X, y)

cols = st.columns(6)
for col, (k, v) in zip(cols, res.items()):
    col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    y_pred = model.predict(X)
    labels = sorted(np.unique(np.concatenate([y, y_pred])))
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = pd.DataFrame(classification_report(y, y_pred, output_dict=True,
                                             zero_division=0)).T.round(3)
    st.dataframe(rep, width="stretch", height=420)

# --------------------------------------------------------------------------
# All-model comparison
# --------------------------------------------------------------------------
st.subheader("Compare all models on this data")
comp = pd.DataFrame(
    {name: evaluate(m, X, y) for name, m in models.items()}
).T.round(4)
comp.index.name = "ML Model Name"
st.dataframe(comp.style.highlight_max(axis=0, color="#cfe8cf"),
             width="stretch")
best = comp["MCC"].idxmax()
st.success(f"Best model by MCC on this data: **{best}** ({comp.loc[best, 'MCC']:.4f})")

# --------------------------------------------------------------------------
# Sample inspector
# --------------------------------------------------------------------------
st.subheader("Inspect individual digits")
idx = st.slider("Row index", 0, len(X) - 1, 0)
img = X.iloc[idx].to_numpy().reshape(8, 8)
a, b = st.columns([1, 2])
with a:
    fig2, ax2 = plt.subplots(figsize=(2.6, 2.6))
    ax2.imshow(img, cmap="gray_r")
    ax2.axis("off")
    st.pyplot(fig2)
with b:
    pred = model.predict(X.iloc[[idx]])[0]
    st.write(f"**True label:** {y.iloc[idx]}  |  **{model_name} predicts:** {pred}")
    st.write("✅ Correct" if pred == y.iloc[idx] else "❌ Misclassified")
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(X.iloc[[idx]])[0],
                          index=model.classes_, name="probability")
        st.bar_chart(proba)
