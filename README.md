# ML Assignment 2: Breast Cancer Classification with Streamlit

## a. Problem Statement

Predict whether a breast tumour is **malignant (1)** or **benign (0)** from 30 numeric features computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass. This is a **binary classification** problem. Five classifiers are trained on the same dataset, compared on six evaluation metrics, and demonstrated in an interactive Streamlit web app.

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic), WDBC |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Instances | 569 (no missing values) |
| Features | 30 real-valued features (requirement: ≥ 12) |
| Target | `diagnosis`: 1 = malignant (212), 0 = benign (357) |
| Train / Test split | 80 / 20, stratified, `random_state=42` → 455 train / 114 test (`test_data.csv`) |

There are ten cell-nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each one is recorded three ways: **mean**, **standard error** and **worst** (the mean of the three largest values). That gives 30 features. Example columns are `mean radius`, `radius error` and `worst radius`.

## c. GitHub Repository Link

- **Repository:** https://github.com/mukulpalmia07/ML-Assignment_2
- **Live Streamlit App:** `<add your Streamlit Community Cloud link here>`

## d. Models Used

All five models are trained on the same training split and evaluated on the same held-out test set (`test_data.csv`). Logistic Regression, KNN and SVM use a `StandardScaler` inside a scikit-learn `Pipeline`, so each saved `.joblib` file contains its preprocessing.

1. **Logistic Regression**: `LogisticRegression(max_iter=1000)`
2. **Decision Tree**: `DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)`
3. **K-Nearest Neighbors**: `KNeighborsClassifier(n_neighbors=7)`
4. **Naive Bayes**: `GaussianNB()`
5. **Support Vector Machine**: RBF `SVC` wrapped in `CalibratedClassifierCV` to give class probabilities for AUC

### Comparison Table (test set, n = 114)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | 0.9756 | **0.9524** | **0.9639** | **0.9433** |

Precision, Recall and F1 are for the positive class (malignant). The same values are saved in `output/metrics.csv`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with the **highest AUC (0.996)**, so its probability ranking is almost perfect. After scaling, the classes are close to linearly separable. It misses 3 malignant cases (recall 0.93). |
| Decision Tree | The **weakest model** on every metric (accuracy 0.90, MCC 0.79). Its axis-aligned splits fit the smooth class boundary poorly, and its recall (0.79) misses about 1 in 5 malignant tumours, even with depth limited to 5. |
| KNN | Very good results (accuracy 0.956, MCC 0.906) because scaled distances in feature space separate the classes well. It is slightly behind LR and SVM, and its lower AUC (0.98) comes from coarse vote-based probabilities. |
| Naive Bayes | **Perfect precision (1.0)**: every malignant prediction is correct. Recall is lower (0.83) because many features are strongly correlated (radius, perimeter and area), which breaks the independence assumption. AUC is still high (0.993). |
| Support Vector Machine | **Best overall**: highest accuracy (0.974), recall (0.952), F1 (0.964) and MCC (0.943). Its AUC (0.995) is close to the top. The RBF kernel with maximum margin handles the slightly non-linear boundary well. |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on 4 of the 6 metrics, including MCC (the most balanced single metric) and recall, which matters most clinically because a missed malignant case costs the most. Logistic Regression is a close second. |

## End-to-End Pipeline

```
train_models.py:  dataset → stratified 80/20 split → fit 5 models → model/*.joblib + test_data.csv + output/metrics.csv
app.py / API:     CSV upload or manual rows → validate → load saved models (cached) → P(malignant) per model
                  → predicted labels (+ metrics, confusion matrix, report when `diagnosis` is given)
```

The Streamlit UI and the FastAPI REST API use the same service classes (`DataProcessor`, `ModelService`, `Evaluator`). Neither one ever retrains on uploaded data.

## Streamlit App Features

- **Overview cards:** dataset size, feature count, number of models and the best model on the test set.
- **Batch evaluation (CSV upload):** use the bundled sample test data or upload your own CSV. The sample file can be downloaded from the sidebar. The results are split into three tabs:
  - **Model comparison:** a metrics table for all five models with the best value in each column highlighted, a bar chart for any chosen metric, and ROC curves for every model.
  - **Model details:** the selected model's six metrics with the gap to the best model, a confusion matrix heatmap and a classification report.
  - **Predictions:** a predicted label and a probability bar for each row, a filter for benign or malignant, a correctness column when labels are given, and a CSV download.
- **Single patient prediction:** a form with the 30 measurements grouped by Mean, Standard error and Worst. One-click benign and malignant examples are included. The result shows a verdict card with the selected model's confidence, how many of the five models agree, and each model's probability.
- **Model selection dropdown** in the sidebar, with a one-line description of each model. All views update instantly when you switch.

Input must contain all 30 feature columns, with the same names as `test_data.csv`. Column order does not matter. The `diagnosis` column (0/1) is optional. Without it, the app returns predictions only.

## REST API (FastAPI)

Run `uvicorn app.main:app` and open `http://127.0.0.1:8000/docs` to try the endpoints in the browser.

| Method | Endpoint | Input | Returns |
|---|---|---|---|
| GET | `/models` | none | model names, required feature names, target |
| POST | `/predict` | JSON list of `{feature: value}` rows | probabilities and labels for every model, plus metrics if `diagnosis` is given |
| POST | `/predict/csv` | CSV file (multipart `file`) | same as above, for bulk data |

## Project Structure

```
├── app.py                     # Streamlit UI (Dashboard class)
├── .streamlit/config.toml     # dark theme, font, upload limit
├── assets/style.css           # custom UI styling (hero, cards, pill tabs, verdict card)
├── train_models.py            # trains & saves all models, writes test_data.csv + output/metrics.csv
├── app/
│   ├── main.py                # FastAPI app
│   ├── api/routes.py          # REST endpoints
│   ├── config/config.py       # Settings: paths, constants, .env overrides
│   ├── config/logger.py       # logging setup
│   ├── services/data_processor.py     # DataProcessor: load, split, validate input
│   ├── services/model_service.py      # ModelService: build, train, save/load, predict
│   ├── services/evaluation_service.py # Evaluator: labels, metrics, confusion matrix, report
│   └── utilities/file_utils.py        # CSV save helper
├── data/dataset.csv           # full dataset (569 rows)
├── model/*.joblib             # 5 trained models
├── output/metrics.csv         # comparison metrics on test set
├── test_data.csv              # held-out test data (114 rows, includes target)
└── requirements.txt
```

## How to Run Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_models.py        # optional: models are already saved in model/
streamlit run app.py          # web UI
uvicorn app.main:app          # optional REST API at /docs
```

## Deployment

The app is deployed on Streamlit Community Cloud from this repository's `main` branch, with `app.py` as the entry point and Python 3.12. `requirements.txt` pins the exact library versions used for training, so the saved `.joblib` models load without errors.
