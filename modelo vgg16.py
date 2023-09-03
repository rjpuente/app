import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

# Definir la ruta a la carpeta que contiene las subcarpetas "Train" y "Test"
dataset_root = os.path.dirname(__file__) + "\\Dataset\\"

# Definir el tamaño de las imágenes y el número de clases
image_size = (64, 64)
num_classes = 8

# Utilizar ImageDataGenerator para cargar y preprocesar imágenes
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Cargar imágenes de entrenamiento
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_root, "Train"),
    target_size=image_size,
    batch_size=30,
    class_mode="sparse",
    shuffle=True,
    seed=42
)

# Cargar imágenes de prueba
test_generator = datagen.flow_from_directory(
    os.path.join(dataset_root, "Test"),
    target_size=image_size,
    batch_size=30,
    class_mode="sparse",
    shuffle=False
)

# Cargar el modelo VGG16 preentrenado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


# Congelar las capas del modelo VGG16 para no entrenarlas nuevamente
for layer in base_model.layers:
    layer.trainable = False

# Agregar capas personalizadas encima del modelo base
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
num_epochs = 10
model.fit(train_generator, epochs=num_epochs)

# Guardar el modelo en un archivo
model.save('modelo_entrenado.keras')
print("Modelo guardado correctamente.")