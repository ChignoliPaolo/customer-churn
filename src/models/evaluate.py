import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.metrics import (
    compute_classification_metrics,
    format_metrics_for_print,
    plot_confusion_matrix,
    plot_roc_curve,
)


def get_project_paths() -> Tuple[str, str, str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return processed_dir, models_dir, figures_dir


def load_best_model(models_dir: str):
    best_name_path = os.path.join(models_dir, "best_model_name.txt")
    if not os.path.exists(best_name_path):
        raise FileNotFoundError(
            f"{best_name_path} non trovato. Esegui prima `python -m src.models.train`."
        )
    with open(best_name_path, "r") as f:
        best_model_name = f.read().strip()

    model_path = os.path.join(models_dir, f"{best_model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} non trovato. Esegui prima `python -m src.models.train`."
        )
    model = joblib.load(model_path)
    return best_model_name, model


def evaluate_best_model() -> None:
    processed_dir, models_dir, figures_dir = get_project_paths()

    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} non trovato. Esegui prima `python -m src.data.preprocess`."
        )

    df_test = pd.read_csv(test_path)
    y_test = df_test["churn"]
    X_test = df_test.drop(columns=["churn"])

    best_model_name, model = load_best_model(models_dir)
    print(f"Valutazione modello: {best_model_name}")

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    print("Metriche sul test set:")
    print(format_metrics_for_print(metrics))

    # Salva confusion matrix
    plt.figure(figsize=(5, 4))
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {best_model_name}")
    cm_path = os.path.join(figures_dir, f"cm_{best_model_name}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix salvata in: {cm_path}")

    # Salva ROC curve
    if y_proba is not None:
        plt.figure(figsize=(5, 4))
        plot_roc_curve(y_test, y_proba, title=f"ROC Curve - {best_model_name}")
        roc_path = os.path.join(figures_dir, f"roc_{best_model_name}.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"ROC curve salvata in: {roc_path}")


def main() -> None:
    evaluate_best_model()


if __name__ == "__main__":
    main()

