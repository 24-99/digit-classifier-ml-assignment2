# Digit Classifier Lab: Comparing Five Classifiers on Handwritten Digits

**Course:** Machine Learning, M.Tech (DSE), BITS Pilani WILP  |  **Assignment 2**

## a. Problem statement

Handwritten-digit recognition is a standard multi-class classification task with real uses such as postal sorting, cheque processing and form digitisation. The goal is to predict which digit (0-9) is shown in an 8x8 grayscale image, given its 64 pixel intensities.

This project trains five classical classifiers on the same dataset, evaluates them with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and exposes them through a deployed Streamlit app where a user can upload test data, choose a model and inspect its performance.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Optical Recognition of Handwritten Digits (UCI ML Repository; test portion distributed with scikit-learn as `load_digits`) |
| Task | Multi-class classification (10 classes: digits 0-9) |
| Instances | 1,797 (minimum required: 500) |
| Features | 64 (minimum required: 12), each the pixel intensity (0-16) of an 8x8 image block |
| Missing values | None |
| Class balance | Near-balanced, 174-183 samples per class |
| Target | `target`: the digit shown |

**Split:** 75% train (1,347 rows) / 25% test (450 rows), stratified, `random_state=42`. The test split is saved as `test_data.csv` (64 feature columns plus `target`) and is the file to upload in the Streamlit app.

**Preprocessing:** `StandardScaler` is applied inside the pipelines for Logistic Regression, kNN and SVM, which are scale-sensitive. Decision Tree and Naive Bayes use raw pixel values.

**Metric averaging (multi-class):** Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro-averaged using predicted probabilities; MCC is the multi-class generalisation.

## c. GitHub repository link

`<PASTE YOUR GITHUB REPO URL HERE>`

Live app: `<PASTE YOUR STREAMLIT COMMUNITY CLOUD URL HERE>`

```
digit-classifier-lab/
|-- app.py                  Streamlit application
|-- requirements.txt
|-- README.md
|-- test_data.csv           held-out test split (upload this in the app)
|-- train_data.csv          training split (used only if saved models must be rebuilt)
|-- model/
|   |-- train_models.py     training + evaluation script
|   |-- *.joblib            one saved pipeline per model
|   |-- metrics.csv         metrics table shown below
```

Reproduce: `pip install -r requirements.txt && python model/train_models.py && streamlit run app.py`

## d. Models used

Hyperparameters: Logistic Regression (`max_iter=2000`); Decision Tree (`max_depth=12`, `min_samples_leaf=2`); kNN (`k=5`, distance-weighted); Gaussian Naive Bayes (`var_smoothing=1e-6`); SVM (RBF kernel, `C=10`, `gamma='scale'`, `probability=True`).

### Comparison table (450-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9778 | 0.9991 | 0.9778 | 0.9777 | 0.9776 | 0.9753 |
| Decision Tree | 0.8267 | 0.9186 | 0.8294 | 0.8258 | 0.8262 | 0.8077 |
| kNN | 0.9644 | 0.9923 | 0.9650 | 0.9642 | 0.9641 | 0.9606 |
| Naive Bayes | 0.8667 | 0.9661 | 0.8760 | 0.8660 | 0.8669 | 0.8528 |
| Support Vector Machine | 0.9800 | 0.9992 | 0.9804 | 0.9798 | 0.9799 | 0.9778 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches 97.8% accuracy and an AUC of 0.999 with a plain linear boundary, which shows the ten digit classes are close to linearly separable in pixel space once features are standardised. Its few errors are mostly 8 and 1 confusions. It is fast to train, easy to interpret and is the strongest simple baseline. |
| Decision Tree | The weakest model (82.7% accuracy, MCC 0.81). Axis-aligned splits on individual pixels are a poor fit for images, where the signal lies in spatial combinations of pixels. It struggles most with digit 8 (only about 65% recall, often predicted as 1 or 7). Its AUC of 0.92 is also the lowest, since leaf probabilities are coarse. |
| kNN | Strong at 96.4% accuracy and 0.992 AUC. Digits of the same class form tight clusters in the standardised 64-dimensional space, so nearest neighbours vote correctly most of the time. Errors are mainly 4 vs 7 and 5 vs 9. Prediction is slower than the other models because it stores the full training set. |
| Naive Bayes | Gets 86.7% accuracy despite its independence assumption, which is clearly violated by neighbouring pixels. It confuses 2 with 8 most often (10 cases), and digit 2 has only about 70% recall. The Gaussian assumption is also a loose fit for bounded 0-16 pixel counts. It is very fast but sits clearly below the discriminative models. |
| Support Vector Machine | Best overall: 98.0% accuracy, AUC 0.9992 and MCC 0.978. The RBF kernel captures non-linear class boundaries while the margin objective keeps it from overfitting. Only 9 of 450 test images are misclassified, mainly 8 vs 1. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)**, with the highest value on every metric, though Logistic Regression is within 0.3 percentage points of accuracy. With 450 test rows, that gap is about 1 image, so the two are statistically close. SVM is chosen on the basis of the metric table; Logistic Regression is the better pick if interpretability and speed matter more. |

## e. Streamlit app features

- **CSV upload** of test data (or a bundled test set for a quick demo)
- **Model dropdown** to choose any of the five models
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) shown for the selected model
- **Confusion matrix and classification report** for the selected model
- **All-model comparison table** on the uploaded data, with the best value per metric highlighted
- **Digit inspector** that renders any test image with the true label, the prediction and class probabilities

## f. Notes

Scores come from a single stratified split with a fixed seed. Small differences between the top models are within the noise of a 450-row test set.
