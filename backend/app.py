
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import pytesseract
# import re
# import cv2

# app = Flask(__name__)
# CORS(app)

# print("🚀 Starting HandwritingAI Backend...")

# # ----------------------------------------------------
# # LOAD MODEL
# # ----------------------------------------------------
# MODEL_PATH = "models/personality_efficientnet.h5"

# try:
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     print("✅ EfficientNet personality model loaded!")
# except Exception as e:
#     print("❌ MODEL LOADING ERROR:", e)
#     raise SystemExit("Model failed to load.")

# # ----------------------------------------------------
# # CONSTANTS
# # ----------------------------------------------------
# IMG_SIZE = (224, 224)

# class_names = [
#     "Agreeableness",
#     "Conscientiousness",
#     "Extraversion",
#     "Neuroticism",
#     "Openness"
# ]

# trait_summary = {
#     "Agreeableness": "Cooperative, kind, empathetic nature.",
#     "Conscientiousness": "Organized, responsible, disciplined.",
#     "Extraversion": "Energetic, talkative, socially expressive.",
#     "Neuroticism": "Emotionally sensitive, easily stressed.",
#     "Openness": "Creative, imaginative, open to new ideas."
# }

# # ----------------------------------------------------
# # IMAGE PREPROCESSING (CRITICAL FIX)
# # ----------------------------------------------------
# def crop_text_region(pil_img):
#     img = np.array(pil_img.convert("L"))

#     _, thresh = cv2.threshold(
#         img, 0, 255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     coords = cv2.findNonZero(thresh)
#     if coords is None:
#         return pil_img

#     x, y, w, h = cv2.boundingRect(coords)
#     cropped = img[y:y+h, x:x+w]
#     return Image.fromarray(cropped)


# def clean_image_for_model(pil_img):
#     img = np.array(pil_img.convert("RGB"))
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )

#     rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
#     return Image.fromarray(rgb)

# # ----------------------------------------------------
# # HANDWRITING VALIDATION
# # ----------------------------------------------------
# def validate_handwriting(img):
#     try:
#         gray = np.array(img.convert("L"))

#         # 1. Resize for consistency
#         gray = cv2.resize(gray, (800, 800))

#         # 2. Binarize strongly
#         thresh = cv2.adaptiveThreshold(
#             gray, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV,
#             51, 15
#         )

#         # 3. Text pixel ratio
#         text_ratio = np.sum(thresh > 0) / thresh.size

#         # ❌ Photos have too much or too little ink
#         if text_ratio < 0.03 or text_ratio > 0.25:
#             return False, "Upload handwriting image only."

#         # 4. Connected components
#         contours, _ = cv2.findContours(
#             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )

#         letter_count = 0
#         for c in contours:
#             x, y, w, h = cv2.boundingRect(c)
#             aspect = w / (h + 1e-5)

#             # Handwritten characters geometry
#             if 12 < h < 90 and 0.3 < aspect < 4:
#                 letter_count += 1

#         # ❌ Real handwriting must have MANY characters
#         if letter_count < 40:
#             return False, "Handwriting not detected."

#         return True, "OK"

#     except Exception as e:
#         print("Validation error:", e)
#         return False, "Invalid image."





# # ----------------------------------------------------
# # PREDICTION FUNCTION (FIXED)
# # ----------------------------------------------------
# def predict_trait_pure(img):
#     # 🔴 MATCH TRAINING DISTRIBUTION
#     img = crop_text_region(img)
#     img = clean_image_for_model(img)

#     img = img.resize(IMG_SIZE)
#     img = img.convert("RGB")

#     arr = keras_image.img_to_array(img)
#     arr = np.expand_dims(arr, axis=0)
#     arr = preprocess_input(arr)

#     preds = model.predict(arr, verbose=0)[0]
#     preds = preds / preds.sum()

#     best_idx = int(np.argmax(preds))
#     best_trait = class_names[best_idx]

#     return best_trait, preds

# # ----------------------------------------------------
# # API ROUTES
# # ----------------------------------------------------
# @app.route("/")
# def home():
#     return "HandwritingAI Backend Running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file uploaded."}), 400

#         img = Image.open(request.files["file"].stream)

#         # VALIDATION
#         valid, msg = validate_handwriting(img)
#         if not valid:
#             return jsonify({"error": msg}), 400

#         # PREDICTION
#         trait, preds = predict_trait_pure(img)

#         scores = {
#             class_names[i]: float(round(preds[i] * 100, 2))
#             for i in range(len(class_names))
#         }

#         return jsonify({
#             "trait": trait,
#             "summary": trait_summary[trait],
#             "scores": scores
#         })

#     except Exception as e:
#         print("❌ Prediction Error:", e)
#         return jsonify({
#             "error": "Prediction failed",
#             "detail": str(e)
#         }), 500

# # ----------------------------------------------------
# # RUN SERVER
# # ----------------------------------------------------
# if __name__ == "__main__":
#     print("🔥 Server running at http://127.0.0.1:5000")
#     app.run(debug=False, use_reloader=False)



from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------
# INIT
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)

print("🚀 HandwritingAI Backend Started")

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness"
]

TRAIT_SUMMARY = {
    "Agreeableness": "Cooperative, kind, empathetic nature.",
    "Conscientiousness": "Organized, responsible, disciplined.",
    "Extraversion": "Energetic, talkative, socially expressive.",
    "Neuroticism": "Emotionally sensitive, easily stressed.",
    "Openness": "Creative, imaginative, open to new ideas."
}

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
HANDWRITING_MODEL_PATH = "models/handwriting_model.h5"
PERSONALITY_MODEL_PATH = "models/mobilenetv2_handwriting_personality_optimized.keras"

handwriting_model = tf.keras.models.load_model(
    HANDWRITING_MODEL_PATH, compile=False
)
personality_model = tf.keras.models.load_model(
    PERSONALITY_MODEL_PATH, compile=False
)

print("✅ Handwriting model loaded")
print("✅ Personality model loaded")

# ----------------------------------------------------
# STAGE 1 — HANDWRITING DETECTION
# IMPORTANT:
# Model was trained with:
# handwritten_images -> 0
# non_handwritten_images -> 1
# So sigmoid output = probability of NON-HANDWRITTEN
# ----------------------------------------------------
def predict_handwriting(img):
    img = img.convert("RGB").resize(IMG_SIZE)

    arr = keras_image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob_non_hw = handwriting_model.predict(arr, verbose=0)[0][0]

    # Correct interpretation
    is_handwriting = prob_non_hw < 0.5
    handwriting_confidence = 1.0 - prob_non_hw

    return is_handwriting, handwriting_confidence

# ----------------------------------------------------
# STAGE 2 — PERSONALITY PREPROCESS
# ----------------------------------------------------
def preprocess_for_personality(img):
    img = img.convert("RGB").resize(IMG_SIZE)

    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    return arr

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def home():
    return "✅ HandwritingAI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(request.files["file"].stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    # ------------------------------------------------
    # STAGE 1 — HANDWRITING CHECK
    # ------------------------------------------------
    is_hw, hw_conf = predict_handwriting(img)

    if not is_hw:
        return jsonify({
            "handwriting_detected": False,
            "handwriting_confidence": round(hw_conf * 100, 2),
            "error": "Uploaded image is NOT handwritten."
        }), 400

    # ------------------------------------------------
    # STAGE 2 — PERSONALITY PREDICTION
    # ------------------------------------------------
    arr = preprocess_for_personality(img)
    preds = personality_model.predict(arr, verbose=0)[0]

    best_idx = int(np.argmax(preds))
    trait = CLASS_NAMES[best_idx]
    confidence = float(preds[best_idx])

    scores = {
        CLASS_NAMES[i]: round(float(preds[i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }

    return jsonify({
        "handwriting_detected": True,
        "handwriting_confidence": round(hw_conf * 100, 2),
        "trait": trait,
        "summary": TRAIT_SUMMARY[trait],
        "confidence": round(confidence * 100, 2),
        "scores": scores
    })

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
