from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def compute_classification_metrics(
    y_true, y_pred, y_proba=None
) -> Dict[str, float]:
    """
    Calcola metriche di classificazione standard per il churn.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "precision_churn": report["1"]["precision"],
        "recall_churn": report["1"]["recall"],
        "f1_churn": report["1"]["f1-score"],
    }

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics["roc_auc"] = auc(fpr, tpr)

    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()


def plot_roc_curve(y_true, y_proba, title: str = "ROC Curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"{title} (AUC = {roc_auc:.3f})")
    plt.tight_layout()


def format_metrics_for_print(metrics: Dict[str, float]) -> str:
    lines = []
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

