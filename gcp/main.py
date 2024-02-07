from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage 
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = 'rk-skindisease-tf-models'

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Acne", "Eczema"]

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000","http://localhost"], allow_headers=['*'])  # Enable CORS for all routes

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

@app.route('/predict', methods=['POST'])
def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/skinDisease.h5",
            "/tmp/skinDisease.h5",
        )
        model = tf.keras.models.load_model("/tmp/skinDisease.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))  # image resizing
    )

    # image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)

    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    response = jsonify({"class": predicted_class, "confidence": confidence})
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

if __name__ == '__main__':
    app.run(debug=True)
