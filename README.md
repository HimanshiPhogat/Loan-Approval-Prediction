# 🏦 AI-Based Loan Approval Prediction

An end-to-end machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant financial and demographic data. It covers data generation, feature engineering, model comparison with cross-validation, evaluation with explainability, and a **Streamlit web app** for live predictions.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- Clean, modular, production-style structure (`src/`, `app/`, `tests/`)
- Single **scikit-learn `Pipeline`**: imputation → scaling → one-hot encoding → model (no data leakage)
- **Domain feature engineering**: total income, estimated EMI, debt-to-income ratio, loan-to-income ratio
- Compares **Logistic Regression, Random Forest, Gradient Boosting** using 5-fold stratified CV (ROC-AUC) and auto-selects the best
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, **permutation feature importance**
- Interactive **Streamlit** UI with an adjustable decision threshold
- Unit tests for data, features and the pipeline

## 📁 Project Structure

```
loan-approval-prediction/
├── app/
│   └── streamlit_app.py      # Web UI
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   └── loan_data.csv         # Generated dataset (5,000 rows)
├── models/
│   └── loan_model.joblib     # Trained pipeline
├── reports/
│   ├── metrics.json          # Saved metrics
│   └── figures/              # Confusion matrix, ROC, feature importance
├── src/
│   ├── config.py             # Paths, features, constants
│   ├── preprocessing.py      # Feature engineering + preprocessing pipeline
│   ├── train.py              # Train / compare / save best model
│   ├── evaluate.py           # Metrics and plots
│   └── predict.py            # Inference helper
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) regenerate the dataset
python data/generate_data.py --rows 5000

# 4. Train and evaluate
python src/train.py

# 5. Quick CLI prediction
python src/predict.py

# 6. Launch the web app
streamlit run app/streamlit_app.py

# 7. Run tests
pytest -q
```

## 📊 Dataset

The repo ships with a **synthetic** dataset (`data/generate_data.py`) that mimics real lending behaviour (credit score, debt-to-income ratio, education, property area, etc.) including realistic missing values. This keeps the project fully reproducible.

To use a real dataset (e.g. Kaggle's *Loan Prediction Problem Dataset*), place it at `data/loan_data.csv`, rename columns to match `src/config.py`, and re-run `python src/train.py`.

| Feature | Description |
|---|---|
| age, gender, married, dependents, education, self_employed | Applicant profile |
| applicant_income, coapplicant_income, existing_emi | Monthly financials |
| loan_amount, loan_term_months | Loan request |
| credit_score, credit_history | Credit bureau information |
| property_area | Urban / Semiurban / Rural |
| **loan_status** | **Target: 1 = Approved, 0 = Rejected** |

## 🧠 Methodology

1. **Split** – 80/20 stratified train/test split.
2. **Feature engineering** – `total_income`, `emi_new`, `dti_ratio`, `loan_to_income`, log transforms.
3. **Preprocessing** – median/mode imputation, standard scaling, one-hot encoding (inside the pipeline).
4. **Model selection** – 5-fold stratified CV on ROC-AUC across three algorithms.
5. **Evaluation** – held-out test set metrics + permutation importance for explainability.

## 📈 Results (held-out test set, 1,000 samples)

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Logistic Regression** (selected) | 0.848 ± 0.014 | 0.836 |
| Gradient Boosting | 0.841 ± 0.015 | 0.838 |
| Random Forest | 0.836 ± 0.016 | 0.823 |

Selected model on test set: **Accuracy 80.3% · Precision 91.9% · Recall 82.5% · F1 86.9% · ROC-AUC 0.836**

| Confusion Matrix | ROC Curve | Feature Importance |
|---|---|---|
| ![cm](reports/figures/confusion_matrix.png) | ![roc](reports/figures/roc_curve.png) | ![fi](reports/figures/feature_importance.png) |

**Key drivers:** credit score, credit history and existing EMI obligations are the most influential features, which matches real-world credit underwriting.

> Note: Results are on synthetic data with injected noise, so a linear model performs on par with tree ensembles. On real, messier datasets, Gradient Boosting / XGBoost often win. The pipeline picks the best automatically.

## 🛠️ Roadmap / Ideas

- [ ] Hyperparameter tuning with Optuna / `GridSearchCV`
- [ ] Add XGBoost / LightGBM candidates
- [ ] SHAP explanations per applicant
- [ ] Fairness audit (disparate impact across gender / marital status)
- [ ] Probability calibration and cost-sensitive thresholding
- [ ] Dockerfile + FastAPI inference service
- [ ] CI with GitHub Actions

## ⚠️ Disclaimer

This is an educational project built on synthetic data. It must **not** be used for real credit decisions. Real lending models require regulatory compliance, fairness testing and human oversight.

## 📄 License

MIT — see [LICENSE](LICENSE).
