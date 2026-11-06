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
GitHub repository: [https://github.com/24-99/digit-classifier-ml-assignment2](https://github.com/24-99/digit-classifier-ml-assignment2)
 
Live app: [https://digit-classifier-ml-assignment2-vzreevfwcxscqekzt2cpdv.streamlit.app/](https://digit-classifier-ml-assignment2-vzreevfwcxscqekzt2cpdv.streamlit.app/)


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
| Logistic Regression | Scored 97.78% accuracy with an AUC of 0.9991, second only to SVM. A simple linear model doing this well tells me the pixel patterns of different digits can be separated almost with straight boundaries once the features are standardised. It made only a handful of mistakes, the most common being true 8s predicted as 1 (3 cases). It trains quickly and its weights are easy to inspect, so it is a very good value baseline. |
| Decision Tree | This was the weakest model, with 82.67% accuracy, MCC 0.81 and the lowest AUC (0.92). A tree asks one-pixel questions at a time, but a digit is defined by how groups of neighbouring pixels combine, so the tree has to approximate shapes with many rough splits. Digit 8 was hit hardest: recall was only about 65%, with 7 of the 8s predicted as 1 and 4 as 7. The coarse leaf probabilities also pull the AUC down. |
| kNN | Achieved 96.44% accuracy and an AUC of 0.9923. After scaling, images of the same digit sit close together in the 64-dimensional space, so the nearest neighbours usually agree on the label. Its errors were few and scattered, mainly 4 predicted as 7 (3 cases) and 5 predicted as 9 (2 cases). The cost is that it keeps the whole training set and compares every new image against it, so prediction is slower than for the other models. |
| Naive Bayes | Reached 86.67% accuracy (MCC 0.85), clearly better than the tree but well behind the top three. It assumes every pixel is independent of the others, which is not true for images where neighbouring pixels move together, and the Gaussian assumption also fits the bounded 0 to 16 pixel values poorly. The biggest weakness was digit 2, with about 70% recall; 10 of the 2s were predicted as 8. It is extremely quick to train, but it pays for its simplicity in accuracy. |
| Support Vector Machine | Top performer on every metric: accuracy 98.00%, AUC 0.9992, precision 0.9804, recall 0.9798, F1 0.9799 and MCC 0.9778. The RBF kernel lets it draw curved boundaries between digit classes, while maximising the margin keeps it from overfitting the 1,347 training images. Only 9 of the 450 test images were misclassified, and the most repeated error was an 8 predicted as 1 (2 cases). |
| Overall winner | Support Vector Machine (RBF) gives the best results on this dataset. Logistic Regression is extremely close, about one test image behind in accuracy, so on 450 test rows the two cannot be separated with confidence. If speed and interpretability matter more than the last fraction of a percent, Logistic Regression would be the practical choice. Digit 8 was among the hardest classes for most of the models (lowest recall for the Decision Tree, kNN and SVM), and the models that ignore pixel relationships (Decision Tree, Naive Bayes) were clearly the weakest. |

## e. Streamlit app features

- **CSV upload** of test data (or a bundled test set for a quick demo)
- **Model dropdown** to choose any of the five models
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) shown for the selected model
- **Confusion matrix and classification report** for the selected model
- **All-model comparison table** on the uploaded data, with the best value per metric highlighted
- **Digit inspector** that renders any test image with the true label, the prediction and class probabilities

## f. Notes

Scores come from a single stratified split with a fixed seed. Small differences between the top models are within the noise of a 450-row test set.
