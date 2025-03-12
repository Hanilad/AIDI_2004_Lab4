from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load saved model, scaler, and label encoder
model = joblib.load("fish_species_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Preprocess input data
        features_scaled = scaler.transform(features)

        # Predict species
        prediction = model.predict(features_scaled)
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"Predicted Species": predicted_species})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)