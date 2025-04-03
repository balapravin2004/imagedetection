from flask import Flask, request, jsonify
import cv2
import numpy as np
import mahotas
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load trained model
clf = joblib.load("fake_image_detector.pkl")

# Function to extract texture features
def extract_features(image_array):
    try:
        image = cv2.imdecode(np.frombuffer(image_array, np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        features = mahotas.features.haralick(image).mean(axis=0)
        return features
    except Exception as e:
        return str(e)  # Return error if processing fails

# API to handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()  # Read image as bytes
        features = extract_features(image_bytes)
        
        if isinstance(features, str):
            return jsonify({'error': f'Feature extraction failed: {features}'}), 500

        prediction = clf.predict([features])[0]
        result = "Fake" if prediction == 1 else "Real"

        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
