from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# ----------------- Load Model -----------------
acne_model = load_model("acne_model.h5")
labels = ["Acne", "Clear"]   # 0 = Acne, 1 = Clear

# ----------------- Ensure folders exist -----------------
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/product-image", exist_ok=True)

# ----------------- Face Detection -----------------
def detect_face_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.45) as face_det:
        results = face_det.process(img_rgb)

        if not results.detections:
            return None

        # Choose largest face
        best = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                          d.location_data.relative_bounding_box.height
        )

        bbox = best.location_data.relative_bounding_box
        h, w, c = img.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        # Padding for more face area
        pad_w = int(0.15 * bw)
        pad_h = int(0.25 * bh)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        face = img[y1:y2, x1:x2]
        return face


# ----------------- Skin Type Detection -----------------
def detect_skin_type(face_img):
    if face_img is None:
        return "Unknown"

    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    brightness = np.mean(rgb)

    if brightness < 100:
        return "Oily"
    elif brightness < 150:
        return "Combination"
    elif brightness < 200:
        return "Normal"
    else:
        return "Dry"


# ----------------- Acne Prediction -----------------
def predict_acne(face_img):
    resized = cv2.resize(face_img, (128, 128))
    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = acne_model.predict(arr)[0][0]  # sigmoid output

    label = labels[1] if pred >= 0.5 else labels[0]  # Clear or Acne
    return label


# ----------------- Flask -----------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    upload_path = "static/uploads/uploaded.jpg"
    file.save(upload_path)

    # Detect face
    face = detect_face_from_path(upload_path)

    if face is None:
        return render_template("result.html", error="No face detected. Please upload a clearer photo.")

    # Save cropped face
    face_path = "static/uploads/face.jpg"
    cv2.imwrite(face_path, face)

    # AI detections
    skin_type = detect_skin_type(face)
    acne_label = predict_acne(face)

    # ----------------- Product Suggestions -----------------
    suggestions = []

    if skin_type == "Oily":
        suggestions.append("Oil-free cleanser")
        suggestions.append("Lightweight non-comedogenic moisturizer")

    elif skin_type == "Dry":
        suggestions.append("Hydrating cleanser")
        suggestions.append("Ceramide-rich moisturizer")

    else:
        suggestions.append("Gentle cleanser")
        suggestions.append("Daily SPF moisturizer")

    if acne_label == "Acne":
        suggestions.insert(0, "Salicylic acid cleanser")
        suggestions.insert(1, "Benzoyl peroxide spot treatment")

    return render_template(
        "result.html",
        skin_type=skin_type,
        acne_label=acne_label,
        suggestions=suggestions,
        face_url=url_for("static", filename="uploads/face.jpg")
    )


if __name__ == "__main__":
    app.run(debug=True)
