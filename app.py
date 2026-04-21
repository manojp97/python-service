from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.plateDetector import detect_number_plate
import os

app = Flask(__name__)
CORS(app)  # allow requests

# ✅ health check
@app.route("/")
def home():
    return "OCR Service Running 🚀"

# ✅ MAIN API (IMPORTANT: /ocr)
@app.route("/ocr", methods=["POST"])
def detect_plate():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        image_bytes = file.read()

        result = detect_number_plate(image_bytes)

        return jsonify({
            "plate_number": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)