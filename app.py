from flask import Flask, render_template, request, jsonify
from utils import predict_email
# ✅ IMPROVEMENT: Import MODEL_PATHS to get the model list automatically
from models import MODEL_PATHS

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("UI.html")

# Analyze with single model
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        email_text = data.get("email_text", "")
        model_choice = data.get("model", "distilbert")

        if not email_text.strip():
            return jsonify({"error": "No email text provided"}), 400

        result = predict_email(model_choice, email_text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Complete Analysis - all 4 models
@app.route("/analyze_all", methods=["POST"])
def analyze_all():
    try:
        data = request.json
        email_text = data.get("email_text", "")

        if not email_text.strip():
            return jsonify({"error": "No email text provided"}), 400

        # ✅ IMPROVEMENT: Get model names directly from MODEL_PATHS keys
        models = list(MODEL_PATHS.keys())
        results = {}

        for model in models:
            results[model] = predict_email(model, email_text)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)