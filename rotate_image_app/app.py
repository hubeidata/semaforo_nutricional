from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
ROTATED_FOLDER = 'static/rotated/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROTATED_FOLDER'] = ROTATED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'jfif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Función para preprocesar la imagen
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalizar la imagen
    return image

# Realizar la predicción
def prediccion(image_path):
    # Ruta al modelo entrenado y a la carpeta de predicciones
    model_path = 'static/model/food20_classification_experimento_model.h5'

    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Diccionario de clases (debes ajustar esto según tus clases)
    class_indices = {'Ceviche': 0, 'Chicharron': 1, 'Soltero de Queso': 2}  # Ajusta según tus clases
    class_labels = {v: k for k, v in class_indices.items()}  # Invertir el diccionario

    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Mostrar la imagen con la etiqueta predicha
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Escribir el texto en la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (5, 20)  # Posición del texto (x, y)
    font_scale = 0.5  # Escala del texto
    font_color = (0, 0, 0)  # Color del texto en BGR (aquí es rojo)
    line_type = 1  # Grosor de la línea del texto

    cv2.putText(img, f"El plato es...{predicted_label}", 
        text_position, 
        font, 
        font_scale, 
        font_color, 
        line_type)
    
    img = cv2.resize(img(200, 200))
    return img  

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(ROTATED_FOLDER):
    os.makedirs(ROTATED_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return render_template('index.html', original_image=file.filename)
    return redirect(request.url)

@app.route('/rotated/<filename>')
def send_rotated_file(filename):
    return send_from_directory(app.config['ROTATED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Vista para manejar la carga de archivos y la predicción
@app.route('/predict/<filename>', methods=['POST'])
def predict_image(filename):
    plato = "chicharrón"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(file_path)
    predict_image = prediccion(file_path)
    predict_image_PIL = Image.fromarray(predict_image.astype('uint8'), 'RGB')
    rotated_path = os.path.join(app.config['ROTATED_FOLDER'], filename)
    predict_image_PIL.save(rotated_path)
    return render_template('index.html', original_image=filename, predict_image=filename, plato=plato)

# Vista para mostrar el formulario de carga de archivos
@app.route('/', methods=['GET'])
def show_form():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)