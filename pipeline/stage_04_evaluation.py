"""Stage 4: evaluate on the 10% held-out test set; write scores.json, plots; log to MLflow."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from tensorflow.keras.models import load_model

from pipeline.common import read_params, setup_mlflow
from pipeline.stage_03_training import MODEL_PATH, make_generator


def main():
    p = read_params()
    model = load_model(MODEL_PATH)
    # shuffle=False is essential: predictions must line up with test.classes
    test = make_generator("artifacts/preprocessing/test.csv", p, augment=False, shuffle=False)

    y_true = np.asarray(test.classes)                                    # 0 = no, 1 = yes
    probs = model.predict(test, verbose=0).ravel()           # P(yes)
    y_pred = (probs >= p["threshold"]).astype(int)
    eps = 1e-7
    loss = float(-np.mean(y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps)))

    scores = {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),     # sensitivity: tumors caught
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan"),
        "test_images": int(len(y_true)),
    }
    with open("scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    out = "artifacts/evaluation"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/classification_report.json", "w") as f:
        json.dump(classification_report(y_true, y_pred, target_names=p["classes"],
                                        output_dict=True, zero_division=0), f, indent=2)


    cm_path, roc_path = f"{out}/confusion_matrix.png", f"{out}/roc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                xticklabels=p["classes"], yticklabels=p["classes"], ax=ax)
    ax.set_xlabel("Predicted tumor")
    ax.set_ylabel("Actual tumor")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {scores['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)

    mlflow = setup_mlflow(p)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tag("stage", "evaluation")
        mlflow.log_metrics({f"test_{k}": v for k, v in scores.items() if v == v})   # skip NaN
        mlflow.log_artifacts(out)
        mlflow.log_artifact("scores.json")
    print(scores)


if __name__ == "__main__":
    main()
