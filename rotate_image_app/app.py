from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import random
import imghdr

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
ROTATED_FOLDER = 'static/rotated/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ROTATED_FOLDER'] = ROTATED_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'ppm', 'pgm', 'ico', 'eps'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(file_path):
    valid_formats = {'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'ppm', 'pgm', 'ico', 'eps'}
    image_format = imghdr.what(file_path)
    return image_format in valid_formats

# Función para preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen con Pillow
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Ajusta el tamaño según tu modelo
    img = np.array(img).astype('float32') / 255.0  # Normaliza la imagen
    img = np.expand_dims(img, axis=0)  # Añade una dimensión para el batch
    return img

# Diccionario con la información nutricional de cada plato
info_nutricional = {
    'ceviche de pescado': {
        'calorías': random.randint(220, 250),
        'proteínas': random.randint(25, 30),
        'carbohidratos': random.randint(15, 20),
        'grasas': random.randint(5, 7)
    },
    'chicharron de cerdo': {
        'calorías': random.randint(450, 600),
        'proteínas': random.randint(30, 35),
        'carbohidratos': random.randint(5, 10),
        'grasas': random.randint(35, 50)
    },
    'soltero de queso': {
        'calorías': random.randint(250, 300),
        'proteínas': random.randint(10, 15),
        'carbohidratos': random.randint(25, 50),
        'grasas': random.randint(10, 20)
    },
    'caldo de gallina': {
        'calorías': random.randint(300, 400),
        'proteínas': random.randint(25, 35),
        'carbohidratos': random.randint(30, 40),
        'grasas': random.randint(10, 20)
    },
    'trucha frita': {
        'calorías': random.randint(350, 450),
        'proteínas': random.randint(30, 35),
        'carbohidratos': random.randint(5, 10),
        'grasas': random.randint(20, 30)
    },
    'tiradito': {
        'calorías': random.randint(200, 250),
        'proteínas': random.randint(25, 30),
        'carbohidratos': random.randint(5, 10),
        'grasas': random.randint(5, 10)
    }
}

# Realizar la predicción
def prediccion(image_path):
    # Ruta al modelo entrenado y a la carpeta de predicciones
    model_path = 'static/model/food_classification_model_1.h5'

    # Cargar el modelo entrenado
    model = load_model(model_path)

    # Diccionario de clases
    class_indices = {
        'ceviche de pescado': 0, 
        'chicharron de cerdo': 1, 
        'soltero de queso': 2, 
        'caldo de gallina': 3, 
        'trucha frita': 4, 
        'tiradito': 5
    }
    class_labels = {v: k for k, v in class_indices.items()}  # Invertir el diccionario
    
    # Preprocesar la imagen
    image = preprocess_image(image_path)
    
    # Hacer la predicción
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Cargar la imagen original con Pillow
    img = Image.open(image_path)
    
    # Verificar si la clase predicha está en class_labels
    if predicted_class not in class_labels:
        raise ValueError(f"Clase predicha {predicted_class} no encontrada en el diccionario de clases")
    
    predicted_label = class_labels[predicted_class]

    # Obtener la información nutricional
    valor_nutricional = info_nutricional.get(predicted_label, {})
    
    return img, predicted_label, valor_nutricional

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
        error_message = 'Ninguna parte del archivo'
        return render_template('index.html', error_message=error_message)
    
    file = request.files['file']
    
    if file.filename == '':
        error_message = 'Ningún archivo seleccionado'
        return render_template('index.html', error_message=error_message)

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Validar si es un archivo de imagen válido
        if is_valid_image(file_path):
            # Redimensionar y guardar la imagen como antes
            with Image.open(file_path) as img:
                base_height = 200
                height_percent = (base_height / float(img.size[1]))
                new_width = int((float(img.size[0]) * float(height_percent)))
                resized_image = img.resize((new_width, base_height))
                resized_filename = f'resized_{file.filename}'
                resized_path = os.path.join(app.config['UPLOAD_FOLDER'], resized_filename)
                resized_image.save(resized_path)
            
            return render_template('index.html', original_image=resized_filename)
        
        else:
            error_message = 'Formato de imagen no válido'
            os.remove(file_path)  # Elimina el archivo no válido
            return render_template('index.html', error_message=error_message)
    
    else:
        error_message = 'Extensión de archivo no permitida'
        return render_template('index.html', error_message=error_message)

@app.route('/rotated/<filename>')
def send_rotated_file(filename):
    return send_from_directory(app.config['ROTATED_FOLDER'], filename)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Vista para manejar la carga de archivos y la predicción
@app.route('/predict/<filename>', methods=['POST'])
def predict_image(filename):
    try:
        # Ruta completa al archivo subido
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Intentar realizar la predicción
        try:
            predict_image, plato, valor_nutricional = prediccion(file_path)

            # La imagen ya está en formato PIL, por lo que no es necesario convertirla
            rotated_path = os.path.join(app.config['ROTATED_FOLDER'], filename)
            predict_image.save(rotated_path)

            # Mostrar la información nutricional en la consola
            print(f"Plato: {plato}")
            if valor_nutricional:  # Verifica si el diccionario no está vacío
                print("Valor Nutricional:")
                for k, v in valor_nutricional.items():
                    print(f"{k.capitalize()}: {v} g")
            else:
                print("Información nutricional no disponible para este plato.")

            # Renderizar la plantilla 'index.html' con la información
            return render_template(
                'index.html',
                original_image=filename,
                predict_image=filename,
                plato=plato,
                valor_nutricional=valor_nutricional
            )
        
        except ValueError as e:
            # Captura el error específico de Keras sobre la forma de la imagen
            error_message = "Error: El software solo procesa imágenes de platos ☺"
            return render_template('index.html', error_message=error_message)
        
    except Exception as e:
        # Manejar cualquier otro error
        error_message = f"Ocurrió un error inesperado: {str(e)}"
        return render_template('index.html', error_message=error_message)
# Vista para mostrar el formulario de carga de archivos
@app.route('/', methods=['GET'])
def show_form():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)