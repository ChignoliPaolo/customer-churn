## Customer Churn Prediction

Data Science project for **customer churn prediction** (customer attrition), designed as a portfolio repository.

The goal is to build an end-to-end pipeline:

- **Data generation / loading**
- **EDA (exploratory data analysis)**
- **Preprocessing and feature engineering**
- **Training of multiple Machine Learning models**
- **Model evaluation and comparison**
- **Saving the best model**

### Project structure

- `data/`
  - `raw/` – raw (or generated) data
  - `processed/` – data ready for modeling
- `notebooks/`
  - `01_eda_customer_churn.ipynb` – exploratory analysis
- `src/`
  - `data/`
    - `make_dataset.py` – script to generate / load data
    - `preprocess.py` – preprocessing and feature engineering
  - `models/`
    - `train.py` – model training
    - `evaluate.py` – evaluation and comparison
  - `utils/`
    - `metrics.py` – custom metrics and helper functions
- `models/`
  - trained models (e.g. `best_model.pkl`)
- `reports/`
  - `figures/` – main plots (ROC, confusion matrix, feature importance, etc.)
- `requirements.txt`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Quick start

1. **Generate or prepare data**

```bash
python -m src.data.make_dataset
python -m src.data.preprocess
```

2. **Train the models**

```bash
python -m src.models.train
```

3. **Evaluate the model**

```bash
python -m src.models.evaluate
```

4. **Explore the EDA notebook**

Open `notebooks/01_eda_customer_churn.ipynb` with Jupyter or VS Code.

### Implemented models

In `src/models/train.py` the following are implemented as examples:

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting (XGBoost or GradientBoostingClassifier)**

All models are compared using typical churn metrics:

- **ROC AUC**
- **Precision, Recall, F1**
- **Confusion Matrix**

### Using this in your portfolio

- Highlights a professional project structure (similar to `cookiecutter-data-science` standards).
- In the README you can emphasize your choices:
  - Feature engineering
  - Model evaluation
  - Business considerations (e.g. focus on recall to avoid missing at-risk customers).

You can customize:

- The data source (replacing synthetic data with real data).
- The models and interpretability techniques (SHAP, feature importance, etc.).
