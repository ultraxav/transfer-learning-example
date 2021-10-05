# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reentrenamiento de Red
#
# Este script carga VGG16, reemplaza la última capa de predicción, y reentrena para clasificar imágenes de 10 categorías de flores.
#
# ## Integrantes:
#
# * Del Villar, Javier
# * Pistoya, Haydeé Soledad
# * Sorza, Andrés
#
# ## Carga de Librerías

# %%
import h5py
import numpy as np
import sklearn.preprocessing
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

print(f'Usando Tensorflow version {tf.__version__}')

if tf.test.gpu_device_name():
    print(f'Usando GPU: {tf.test.gpu_device_name()}')
else:
    print('Usando CPU')

# %% [markdown]
# ## Parámetros

# %%
# Algunos parámetros del entrenamiento
batch_size = 32
epochs = 13
epochs_to_stop_after_no_improvement = 3
num_cores = 4

# Parámetros del descenso de gradiente
learning_rate = 0.001
learning_rate_decay = 1e-6
learning_rate_momentum = 0.7


# %% [markdown]
# ## Funciones de Soporte

# %%
def preprocesar_imagen_como_caffe(image: np.ndarray) -> np.ndarray:
    """
    Transforma las imágenes al formato con el que fue entrenado el modelo de VGG16 que estamos usando.
    :param image: Una imagen representada como una matriz de (largo en pixeles, alto en pixeles, 3 canales)
    :return La imagen transformada.
    """
    # Pasar imagen de 'RGB'->'BGR', porque el modelo ya entrenado de VGG16 que estamos usando
    # proviene de Caffe, y fue entrenado en ese orden de channels
    image = image[:, :, ::-1]

    # Centrar el valor de los pixeles alrededor del valor medio de cada canal en el conj. de
    # entrenamiento, esto se calcula simplemente promediando todos los valores de cada canal en
    # todas las imágenes de entrenamiento en imagenet.
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    return image


def rescalar_imagenes(flower_images: np.ndarray) -> np.ndarray:
    """
    Cambia el tamaño de las imágenes de flores al tamaño con el que esta entrenada VGG16: 224x224 pixeles
    :param flower_images: Una matriz de (nro_imagenes, ancho en pixeles, alto en pixeles, 3 canales); cada valor de la matriz esta entre 0 y 255.
    :return: Otra matriz de las mismas dimensiones de 'flowers' pero con todos los valores entre 0 y 1.
    """
    rescaled_images = np.empty(
        (flower_images.shape[0], 224, 224, flower_images.shape[3]),
        dtype=flower_images.dtype,
    )
    for i in range(0, flower_images.shape[0]):
        rescaled_images[i, 0:224, 0:224,] = (
            resize(flower_images[i] / 255.0, (224, 224), anti_aliasing=True) * 255.0
        )
    return rescaled_images


def encode_onehot_labels(labels: np.ndarray) -> np.ndarray:
    """
    Cambia la codificación de las categorías de flores a one-hot encoding, que es lo que necesitamos para entrenar la NN.
    :param labels: Una lista o arreglo de strings, el i-avo string es la categoría de la i-ava imagen de entrenamiento.
    :return una matriz de tamaño (cantidad de ejemplos en labels, cantidad de categorías en 'labels') donde cada celda es 1 o 0. Hay 1 solo 1 por fila.
    """
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(labels) + 1))
    return label_binarizer.transform(labels)


# %% [markdown]
# ## Definición de la Red Neuronal

# %%
def VGG_16():
    """
    Crea una red para clasificar imágenes con la arquitectura VGG16, para poder reusar los pesos de VGG16 entrenado con imagenet.
    """
    # Tamaño de imágenes es 224x224 y 3 canales de colores
    img_input = Input(shape=(224, 224, 3))

    # Block 1
    output = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input
    )
    output = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(
        output
    )
    output = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(output)

    # Block 2
    output = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1'
    )(output)
    output = Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2'
    )(output)
    output = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(output)

    # Block 3
    output = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1'
    )(output)
    output = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2'
    )(output)
    output = Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3'
    )(output)
    output = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(output)

    # Block 4
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1'
    )(output)
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2'
    )(output)
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3'
    )(output)
    output = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(output)

    # Block 5
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1'
    )(output)
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2'
    )(output)
    output = Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3'
    )(output)
    output = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(output)

    # Capa de clasificación
    output = Flatten(name='flatten')(output)
    output = Dense(4096, activation='relu', name='fc1')(output)
    output = Dense(4096, activation='relu', name='fc2')(output)

    # La salida son las 1000 categorías de imagenet
    output = Dense(1000, activation='softmax', name='predictions')(output)

    return Model(inputs=img_input, outputs=output, name='vgg16')


# %% [markdown]
# ## Carga de Datos
#
# Leer el dataset de 210 imagenes de flores junto con sus etiquetas

# %%
dataset = h5py.File(
    '../data/imagenet_dataset/10FlowerColorImages.h5',
    'r',
)
# dataset = h5py.File(
#     '/content/drive/MyDrive/collab/transfer_learning/imagenet_dataset/10FlowerColorImages.h5',
#     'r',
# )

# El [()] indica que lea el dataset completo a memoria, en vez de leerlo bajo demanda
images = dataset['images'][()]
image_labels = dataset['labels'][()]
images = rescalar_imagenes(images)
onehot_labels = encode_onehot_labels(image_labels)

# %% [markdown]
# ## Armado de la arquitectura de la Red Neuronal

# %%
pretrained_vgg16 = VGG_16()

# %% [markdown]
# ## Leer los pesos de VGG16 ya entrenada con imagenet
#
# El modelo fue entrenado con los canales de cada imagen en el orden 'BGR' que utiliza la biblioteca "Caffe", y c/pixel centrado sobre la media del dataset imagenet = [103.939, 116.779, 123.68]
#
# Las imágenes nuevas deben tener exactamente esta transformación.

# %%
pretrained_vgg16.load_weights('../data/imagenet_dataset/vgg16_weights.h5')
# pretrained_vgg16.load_weights(
#     '/content/drive/MyDrive/collab/transfer_learning/imagenet_dataset/vgg16_weights.h5'
# )

# %% [markdown]
# ## Modificación de la Red Neuronal

# %%
# Setear a todas las capas, excepto la última de clasificación
# como "no entrenable" (los pesos no se actualizarán)
for layer in pretrained_vgg16.layers[:-1]:
    layer.trainable = False

# Descartamos la última capa de vgg16 y creamos una capa nueva, cuya entrada es la anteúltima capa
# ("fc2") de vgg16.
# Necesitamos hacer esto porque el número de categorías en el nuevo dataset es diferente al número
# de categorias en imagenet.
new_layer = Dense(10, activation='softmax', name='predict_10flowers')(
    pretrained_vgg16.get_layer('fc2').output
)

# crear un nuevo modelo cuya salida es la nueva capa
new_model = Model(inputs=pretrained_vgg16.input, outputs=new_layer)


# %%
# Compilar el modelo con SGD/momentum optimizer y un learning rate muy lento
sgd = SGD(
    learning_rate=learning_rate,
    decay=learning_rate_decay,
    momentum=learning_rate_momentum,
    nesterov=True,
)
# Regla en Keras:
# Si loss = categorical_crossentropy => metrics = categorical_accuracy
# Si loss = binary_crossentropy y más de 2 clases => metrics = categorical_accuracy,
# Sino metrics=binary_accuracy
new_model.compile(
    optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy']
)

# %% [markdown]
# ## Creación de datasets de entrenamiento

# %%
X_train, X_test, y_train, y_test = train_test_split(
    images,
    onehot_labels,
    train_size=0.7,
    test_size=0.3,
    shuffle=True,
    stratify=image_labels,
)

# Aumentar la cantidad de ejemplos de entrenamiento, inventando variaciones de las imágenes
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=[0.8, 1.4],
    horizontal_flip=True,
    rotation_range=10,
    preprocessing_function=preprocesar_imagen_como_caffe,
)

# Testear con los casos de prueba sin rotar, cambiar tamaño ni nada, solo acomodados a la manera en
# que fue entrenada la NN
test_datagen = ImageDataGenerator(preprocessing_function=preprocesar_imagen_como_caffe)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size,
)

validation_generator = test_datagen.flow(X_test, y_test)

# %% [markdown]
# # Entrenamiento

# %%
new_model.fit(
    train_generator,
    steps_per_epoch=len(y_train) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(y_test) // batch_size,
    # use_multiprocessing=True,
    # workers=num_cores,
    callbacks=[
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=epochs_to_stop_after_no_improvement,
            verbose=1,
        ),
        ModelCheckpoint(
            '../data/vgg16_retrained_10flowers.h5',
            # '/content/drive/MyDrive/collab/transfer_learning/vgg16_retrained_10flowers.h5',
            verbose=1,
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='auto',
        ),
    ],
)

print('Entrenamiento finalizado!')
