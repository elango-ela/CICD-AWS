"""Stage 3: train a binary (tumor yes / no) InceptionV3 transfer-learning model, track with MLflow."""
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pipeline.common import read_params, setup_mlflow

MODEL_PATH = "artifacts/training/model.h5"
SERVE_PATH = "model/model.h5"       # copy committed to git and baked into the Docker image


class MlflowEpochLogger(Callback):
    def __init__(self, mlflow):
        super().__init__()
        self.mlflow = mlflow

    def on_epoch_end(self, epoch, logs=None):
        self.mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def make_generator(csv, p, augment, shuffle):
    """class_mode='binary' -> label is 0/1 following p['classes'] order (no=0, yes=1)."""
    kwargs = dict(rescale=1 / 255)
    if augment:
        kwargs.update(p["augmentation"])
    size = p["image_size"]
    return ImageDataGenerator(**kwargs).flow_from_dataframe(
        pd.read_csv(csv), x_col="image_filepaths", y_col="tumor_class",
        classes=p["classes"], target_size=(size, size), batch_size=p["batch_size"],
        color_mode="rgb", class_mode="binary", shuffle=shuffle, seed=p["seed"])


def build_model(p):
    size = p["image_size"]
    base = InceptionV3(include_top=False, weights=p["weights"],
                       input_shape=(size, size, 3), pooling="avg")
    base.trainable = False
    model = Sequential([
        base,
        Dense(256, activation="relu", name="Hidden-Layer"),
        Dropout(0.5),
        Dense(1, activation="sigmoid", name="Output-Layer"),    # P(tumor = yes)
    ])
    model.compile(optimizer=Adam(p["learning_rate"]), loss="binary_crossentropy",
                  metrics=["accuracy", AUC(name="auc"), Precision(name="precision"),
                           Recall(name="recall")])
    return model


def class_weights(labels) -> dict:
    counts = np.bincount(labels)
    return {i: float(len(labels) / (len(counts) * c)) for i, c in enumerate(counts)}


def main():
    p = read_params()
    tf.keras.utils.set_random_seed(p["seed"])

    train = make_generator("artifacts/preprocessing/train.csv", p, augment=True, shuffle=True)
    val = make_generator("artifacts/preprocessing/val.csv", p, augment=False, shuffle=False)
    weights = class_weights(train.classes) if p["use_class_weight"] else None
    model = build_model(p)

    mlflow = setup_mlflow(p)
    with mlflow.start_run(run_name="training"):
        mlflow.set_tag("stage", "training")
        mlflow.log_params({k: p[k] for k in ["image_size", "batch_size", "epochs", "learning_rate",
                                             "weights", "early_stopping_patience", "seed",
                                             "use_class_weight", "threshold"]})
        mlflow.log_params({f"aug_{k}": v for k, v in p["augmentation"].items()})
        mlflow.log_params({"train_images": train.n, "val_images": val.n})

        history = model.fit(
            train, validation_data=val, epochs=p["epochs"], class_weight=weights,
            callbacks=[EarlyStopping(monitor="val_loss", patience=p["early_stopping_patience"],
                                     restore_best_weights=True),
                       MlflowEpochLogger(mlflow)])

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        os.makedirs(os.path.dirname(SERVE_PATH), exist_ok=True)
        shutil.copy(MODEL_PATH, SERVE_PATH)


        mlflow.log_metric("best_val_accuracy", max(history.history["val_accuracy"]))
        mlflow.log_artifact(MODEL_PATH)
    print("Model saved to", MODEL_PATH, "and", SERVE_PATH)


if __name__ == "__main__":
    main()
