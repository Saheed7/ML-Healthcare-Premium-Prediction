# Healthcare Insurance Premium Prediction — End-to-End ML System

Predicts annual health insurance premiums from demographic, lifestyle, and medical-history inputs. Covers the complete ML lifecycle: data cleaning and EDA → feature engineering → model selection and tuning → artifact serialization → an interactive Streamlit web app for real-time inference.

**Test R² = 0.993 · RMSE ≈ ₹745** on a held-out 30% test set (XGBoost, tuned via RandomizedSearchCV).

<!-- Add a screenshot or GIF of the running app here -->
<!-- ![App Demo](assets/demo.gif) -->

---

## Highlights

- **End-to-end pipeline, not just a notebook.** Training produces versioned artifacts (`model.joblib`, `scaler.joblib`); a decoupled inference module (`prediction_helper.py`) consumes them; a Streamlit UI serves predictions in real time.
- **Training/serving consistency by design.** The fitted scaler is serialized *together with* its column list in a single bundle, making it impossible for the app's preprocessing to drift from training — a common source of silent production bugs.
- **Domain-driven feature engineering.** Raw medical history is transformed into a normalized clinical risk score (e.g., heart disease = 8, diabetes = 6, summed across comorbidities and min-max normalized), and physical activity + stress level are combined into a lifestyle risk score. Both rank among the model's most informative features.
- **Statistical rigor.** Multicollinearity was diagnosed with Variance Inflation Factor (VIF); `income_level` was dropped due to redundancy with `income_lakhs`. Outliers were treated with IQR analysis and quantile capping (99.9th percentile income, 99th percentile dependants).
- **Error analysis beyond a single metric.** Residuals were examined as percentage deviations, with the share of predictions exceeding a 10% error threshold quantified — the kind of analysis that motivates model iteration in production settings.
- **Defensive inference code.** The prediction module validates input keys with explicit, actionable error messages and uses path-safe artifact loading (`pathlib`), so the app fails loudly and diagnosably rather than silently.

---

## Architecture

```
premiums.csv
     │
     ▼
┌─────────────────────────────┐
│  Training Notebook          │
│  cleaning → EDA → features  │
│  → encoding → scaling       │
│  → XGBoost + tuning         │
└──────────────┬──────────────┘
               │  serializes
               ▼
   artifacts/model.joblib
   artifacts/scaler.joblib   (scaler + cols_to_scale bundled)
               │  loaded by
               ▼
┌─────────────────────────────┐
│  prediction_helper.py       │
│  input validation →         │
│  feature construction →     │
│  scaling → model.predict    │
└──────────────┬──────────────┘
               │  called by
               ▼
┌─────────────────────────────┐
│  main.py (Streamlit UI)     │
│  13 user inputs → premium   │
└─────────────────────────────┘
```

---

## Model Development Summary

| Step | Decision | Rationale |
|---|---|---|
| Outliers | Age capped at 100; income capped at 99.9th percentile; dependants at 99th | Remove data-entry artifacts without discarding signal |
| Feature engineering | Normalized medical risk score (0–14 → 0–1); lifestyle risk score | Encode domain knowledge; both rank highly in feature importance |
| Encoding | Ordinal for `insurance_plan` / `income_level`; one-hot (drop-first) for 6 nominal features | Match variable semantics; avoid dummy trap |
| Multicollinearity | VIF analysis → dropped `income_level` | Redundant with `income_lakhs` |
| Scaling | MinMaxScaler on 6 numeric/ordinal features | Persisted with column list for serving parity |
| Baseline | Linear Regression | Interpretable benchmark |
| Final model | XGBoost, RandomizedSearchCV (3-fold CV, R² scoring) | Best params: 50 estimators, depth 5, lr 0.1 |
| Validation | 70/30 split; train R² 0.994 vs test R² 0.993 | No overfitting; residual %-error distribution inspected |

---

## Project Structure

```
├── ml_premium_prediction.ipynb   # Full training pipeline (EDA → artifacts)
├── main.py                       # Streamlit application
├── prediction_helper.py          # Preprocessing + inference module
├── artifacts/
│   ├── model.joblib              # Tuned XGBoost regressor
│   └── scaler.joblib             # {scaler, cols_to_scale} bundle
├── requirements.txt              # Pinned artifact-sensitive dependencies
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Saheed7/Healthcare-Premium-Prediction.git
cd Healthcare-Premium-Prediction
pip install -r requirements.txt

# 2. (Optional) Retrain — regenerates artifacts/ from the dataset
jupyter notebook ml_premium_prediction.ipynb   # Run All Cells

# 3. Launch the app
streamlit run main.py
```

Open `http://localhost:8501`, enter policyholder details, and click **Predict**.

> **Data note:** The dataset (10,000 policy records, 15 attributes) is excluded from the repo. Expected schema: age, gender, region, marital status, physical activity, stress level, number of dependants, BMI category, smoking status, employment status, income level, income (lakhs), medical history, insurance plan, and annual premium amount (target).

---

## Example Predictions

| Profile | Predicted Premium |
|---|---|
| 25y, healthy, non-smoker, Bronze plan | ₹ 4,974 |
| 55y, diabetes + heart disease, regular smoker, Gold plan | ₹ 41,689 |

Predictions scale sensibly with age, clinical risk, smoking status, and plan tier — consistent with actuarial expectations.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels (VIF) · Streamlit · joblib · Matplotlib/Seaborn

Artifact-sensitive packages (numpy, scikit-learn, xgboost, joblib) are version-pinned in `requirements.txt` to guarantee that serialized artifacts deserialize against the same library code that produced them.

---

## Roadmap

- [ ] Age-segmented models (young vs. rest) driven by residual error analysis
- [ ] Experiment tracking with MLflow
- [ ] Dockerized deployment + CI (GitHub Actions)
- [ ] Live demo on Streamlit Community Cloud

---

## Author

**Yakub Kayode Saheed, Ph.D.** — Machine Learning Engineer · [GitHub](https://github.com/Saheed7) · [LinkedIn](https://www.linkedin.com/in/yakub-kayode-saheed-94468672) · [Google Scholar](https://scholar.google.com/citations?user=faYh6iIAAAAJ)
