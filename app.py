from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)
classifier = PredictionPipeline()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify(error="Upload an image in the 'file' field"), 400
    try:
        return jsonify(classifier.predict(file.stream))
    except Exception:
        return jsonify(error="Could not read the image"), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
