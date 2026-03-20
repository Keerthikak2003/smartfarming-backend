import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# App Initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Paths & Model
# -----------------------------
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/crop_disease_model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# Class Names (must match dataset)
# -----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------------
# Home Route (IMPORTANT)
# -----------------------------
@app.route("/")
def home():
    return "✅ Smart Farming Backend is running successfully!"

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Fertilizer Suggestions
# -----------------------------
fertilizer_map = {
    "Tomato_Early_blight": "Use balanced NPK fertilizer and copper fungicide",
    "Tomato_Late_blight": "Apply potassium-rich fertilizer and fungicide",
    "Potato___Early_blight": "Use nitrogen and phosphorus fertilizer",
    "Potato___Late_blight": "Apply potash fertilizer and disease control spray",
    "Pepper__bell___Bacterial_spot": "Use calcium-based fertilizer",
}

# -----------------------------
# Disease Detection API
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    # Predict disease
    img_array = preprocess_image(save_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    disease = class_names[predicted_index]

    fertilizer = fertilizer_map.get(
        disease, "Use general organic fertilizer and monitor plant health"
    )

    return jsonify({
        "disease": disease,
        "fertilizer": fertilizer
    })

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
