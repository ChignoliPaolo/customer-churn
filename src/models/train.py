import os
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils.metrics import compute_classification_metrics, format_metrics_for_print


def get_project_paths() -> Tuple[str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return processed_dir, models_dir


def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    processed_dir, _ = get_project_paths()
    train_path = os.path.join(processed_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"File {train_path} non trovato. Esegui prima `python -m src.data.preprocess`."
        )
    df_train = pd.read_csv(train_path)
    y_train = df_train["churn"]
    X_train = df_train.drop(columns=["churn"])
    return X_train, y_train


def build_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs"
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
    }
    return models


def train_and_evaluate_models() -> str:
    X_train, y_train = load_train_data()
    processed_dir, models_dir = get_project_paths()

    test_path = os.path.join(processed_dir, "test.csv")
    df_test = pd.read_csv(test_path)
    y_test = df_test["churn"]
    X_test = df_test.drop(columns=["churn"])

    models = build_models()
    metrics_summary: Dict[str, Dict[str, float]] = {}

    best_model_name = None
    best_auc = -1.0

    for name, model in models.items():
        print(f"\n=== Training modello: {name} ===")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # fallback per modelli senza predict_proba
            from sklearn.preprocessing import MinMaxScaler

            scores = model.decision_function(X_test).reshape(-1, 1)
            y_proba = MinMaxScaler().fit_transform(scores).ravel()
        else:
            y_proba = None

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        metrics_summary[name] = metrics

        print(format_metrics_for_print(metrics))

        if y_proba is not None:
            auc_score = roc_auc_score(y_test, y_proba)
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_name = name

        # Salva modello singolo
        model_path = os.path.join(models_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Modello {name} salvato in: {model_path}")

    if best_model_name is None:
        # fallback: prendi il primo modello
        best_model_name = list(models.keys())[0]

    # Salva info sul miglior modello
    summary_df = pd.DataFrame(metrics_summary).T
    summary_path = os.path.join(models_dir, "model_comparison.csv")
    summary_df.to_csv(summary_path)
    print(f"\nConfronto modelli salvato in: {summary_path}")
    print(summary_df)

    best_path = os.path.join(models_dir, "best_model_name.txt")
    with open(best_path, "w") as f:
        f.write(best_model_name)

    print(f"Miglior modello (per ROC AUC): {best_model_name}")
    return best_model_name


def main() -> None:
    best_model_name = train_and_evaluate_models()
    print(f"Training completato. Miglior modello: {best_model_name}")


if __name__ == "__main__":
    main()

