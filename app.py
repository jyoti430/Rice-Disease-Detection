from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
from PIL import Image
import os
from flask_cors import CORS
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# Paths & Model Loading

MODEL_PATH = "models/rice_model_v2.keras"
CLASS_PATH = "models/rice_class_names_v2.json"
PLANT_INFO_PATH = "models/plant_disease.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

plant_info = {}
if os.path.exists(PLANT_INFO_PATH):
    try:
        with open(PLANT_INFO_PATH, "r", encoding="utf-8") as pf:
            plant_info = json.load(pf)
    except:
        plant_info = {}

IMG_SIZE = (224, 224)

# Ensure upload directory exists

UPLOAD_DIR = "uploadimages"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded images

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# Home page

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

# Correct Preprocessing (EfficientNetV2)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img)

    # IMPORTANT: exact same preprocessing as training
    img_array = preprocess_input(img_array)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Model Prediction

def model_predict(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)

    pred_index = int(np.argmax(preds))
    disease = class_names[pred_index]
    confidence = float(preds[0][pred_index])

    crop = "Rice"
    return crop, disease, confidence

# HTML Upload Route

@app.route("/upload/", methods=["POST", "GET"])
def uploadimage():
    if request.method == "POST":
        image = request.files.get('img')

        if not image or not image.filename:
            return redirect('/')

        unique_name = f"{UPLOAD_DIR}/temp_{uuid.uuid4().hex}_{image.filename}"
        image.save(unique_name)

        crop, disease, confidence = model_predict(unique_name)
        info = plant_info.get(disease, {})

        return render_template(
            "home.html",
            result=True,
            imagepath="/" + unique_name,
            crop=crop,
            disease=disease,
            confidence=round(confidence * 100, 2),
            info=info
        )

    return redirect('/')

# JSON API Route (/api/predict)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "img" not in request.files:
        return jsonify({"error": "No file part 'img'"}), 400

    f = request.files["img"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save temp file
    unique_name = f"temp_{uuid.uuid4().hex}_{f.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    f.save(save_path)

    try:
        crop, disease, confidence = model_predict(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    info = plant_info.get(disease, {})
    image_url = f"/{UPLOAD_DIR}/{unique_name}"

    return jsonify({
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
        "image_url": image_url,
        "info": info
    })

# Run the App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
