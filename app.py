from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

basedir = os.path.dirname(__file__)
model = load_model(os.path.join(basedir, 'modelo_entrenado_vgg16_inception.keras'))

def predict_image(image_data):
    try:
        print("Received image data")
        image = Image.open(image_data)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict([image_array, image_array])
        print("La predicción es " + str(prediction))
        return prediction
    except Exception as e:
        print("Error processing image:", e)
        raise e

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected successfully'})

@socketio.on('image')
def handle_image(image_data):
    try:
        print("Length of image data:", len(image_data))
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Crear un objeto BytesIO para convertir los bytes en un objeto de tipo file-like
        image_stream = BytesIO(image_bytes)

        # Abrir la imagen usando Pillow
        image = Image.open(image_stream)

        # Redimensionar la imagen a 75x75 píxeles
        image = image.resize((75, 75))

        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict([image_array, image_array])

        # Obtener el índice de la clase con la probabilidad más alta
        class_index = np.argmax(prediction)

        response = {
            'class_index': int(class_index),
            'class_confidence': float(prediction[0][class_index])
        }

        emit('prediction', response)
    except Exception as e:
        print("Error processing image:", e)
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
