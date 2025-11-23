from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image

# Handle imports
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess
except ImportError:
    from keras.applications.efficientnet import preprocess_input as efn_preprocess

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MODEL_PATH = "final_model_ekher_version.keras"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
print(f"\n{'=' * 60}")
print(f"🔄 Loading model: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
print(f"{'=' * 60}\n")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simple_preprocess(img):
    """Simple auto-crop - matches training preprocessing"""
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
    return img


def preprocess_image(image_path):
    """Preprocess image exactly like training data"""
    try:
        # Read image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image")

        # Apply simple preprocessing (crop)
        img = simple_preprocess(img)

        # Resize to model input size
        img = cv2.resize(img, (512, 512))

        # Convert grayscale to RGB (3 channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert to float and add batch dimension
        img_array = img_rgb.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply EfficientNet preprocessing
        img_array = efn_preprocess(img_array)

        return img_array

    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def predict_image(image_path):
    """Make prediction on preprocessed image"""
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Preprocess
        img_array = preprocess_image(image_path)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Interpret results
        probability = float(prediction)
        is_malignant = probability > 0.5
        confidence = probability if is_malignant else (1 - probability)

        result = {
            "prediction": "MALIGNANT" if is_malignant else "BENIGN",
            "probability_malignant": round(probability * 100, 2),
            "probability_benign": round((1 - probability) * 100, 2),
            "confidence": round(confidence * 100, 2)
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, DCM"}), 400

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = predict_image(filepath)

        # Clean up
        os.remove(filepath)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)