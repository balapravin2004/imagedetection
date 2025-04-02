from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mahotas
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
clf = joblib.load("fake_image_detector.pkl")

# Function to extract texture features
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    features = mahotas.features.haralick(image).mean(axis=0)
    return features

# Route to render UI
@app.route('/')
def index():
    return render_template('index.html')

# API to handle image upload and classification
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    
    # Process and classify image
    features = extract_features(filepath)
    prediction = clf.predict([features])[0]
    result = "Fake" if prediction == 1 else "Real"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
