"""Inference used by the Flask app (no mlflow / dvc needed)."""
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from pipeline.common import read_params

MODEL_PATH = "model/model.h5"


class PredictionPipeline:
    def __init__(self, model_path: str = MODEL_PATH):
        p = read_params()
        self.size = p["image_size"]
        self.threshold = p["threshold"]
        self.negative, self.positive = p["classes"]          # "no", "yes"
        self.labels = p["labels"]
        self.model = load_model(model_path)                  # loaded once, not per request

    def predict(self, image_file) -> dict:
        img = Image.open(image_file).convert("RGB").resize((self.size, self.size))
        x = np.expand_dims(np.asarray(img, dtype="float32") / 255.0, axis=0)   # same rescale as training
        p_yes = float(self.model.predict(x, verbose=0)[0][0])
        is_yes = p_yes >= self.threshold
        cls = self.positive if is_yes else self.negative
        return {"prediction": cls,
                "label": self.labels[cls],
                "confidence": round(p_yes if is_yes else 1 - p_yes, 4),
                "probability_tumor": round(p_yes, 4)}
