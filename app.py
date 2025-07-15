from flask import Flask, request, jsonify  # ✅ Import Flask before using it
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model and column structure
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({"churn": int(prediction), "probability": round(probability, 2)})

if __name__ == "__main__":
    app.run(debug=True)

