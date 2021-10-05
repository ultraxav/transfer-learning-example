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

# %% id="njEh8Cemo3zb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632436518790, "user_tz": 180, "elapsed": 11154, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}} outputId="095da976-a451-4f60-9236-3dc82b2f63de"
# Antes de ejecutar: Activar TPUs como sigue:
# menu "Entorno de Ejecucion" -> "Cambiar tipo de entorno de ejecucion" -> "Acelerador de Hardware" = "TPU"

# %tensorflow_version 2.x
import tensorflow as tf

print("Usandor Tensorflow version " + tf.__version__)


if tf.test.gpu_device_name():
    print('Usando GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print("Usando CPU.")

# %% id="IFNd6dlwv7T1" executionInfo={"status": "ok", "timestamp": 1632436520218, "user_tz": 180, "elapsed": 1435, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
##################################################################
# Este script carga VGG16, reemplaza la ultima capa de prediccion,
# y reentrena para clasificar imagenes de 10 categorias de flores.
##################################################################

import h5py
from skimage.transform import resize
import numpy as np

from keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


# %% id="fm-hHB0NyDux" executionInfo={"status": "ok", "timestamp": 1632436520219, "user_tz": 180, "elapsed": 24, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# algunos parametros del entrenamiento
batch_size = 32  # cada batch son 32 imagenes
epochs = 13  # entrenamos hasta 13 epochs (pasadas sobre el dataset de entrenamiento), a menos que paremos antes por early stopping
epochs_to_stop_after_no_improvement = 3  # cuantas epochs consecutivas no tienen que tener mejora para aplicar early stopping
num_cores = 4  # cambiar este numero a la cantidad de cores de su cpu

# parametros del descenso de gradiente
learning_rate = 0.001
learning_rate_decay = 1e-6
learning_rate_momentum = 0.7


# %% id="iI_H5qI_yF9w" executionInfo={"status": "ok", "timestamp": 1632436520219, "user_tz": 180, "elapsed": 23, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
def preprocesar_imagen_como_caffe(image: np.ndarray) -> np.ndarray:
    """
    Transforma las imagenes al formato con el que fue entrenado el modelo de VGG16 que estamos usando.
    :param image: Una imagen representada como una matriz de (largo en pixels, alto en pixels, 3 canales)
    :return La imagen transformada.
    """
    # pasar imagen de  'RGB'->'BGR', porque el modelo ya entrenado de VGG16 que estamos usando proviene de Caffe, y fue entrenado en ese orden de channels
    image = image[:, :, ::-1]
    # central valor de los pixels alrededor del valor medio de cada canal en el conj. de entrenamiento,
    # esto se calcula simplemente promediando todos los valores de cada canal en todas las imagenes de entrenamiento en imagenet.
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    return image


def rescalar_imagenes(flower_images: np.ndarray) -> np.ndarray:
    """
    Cambia el tamaño de las imagenes de flores al tamaño con el que esta entrenada VGG16: 224x224 pixels
    :param flower_images: Una matriz de (nro_imagenes, ancho en pixels, alto en pixels, 3 canales); cada valor de la matriz esta entre 0 y 255.
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
    Cambia la codificacion de las categorias de flores a one-hot encoding, que es lo que necesitamos para entrenar la NN.
    :param labels: Una lista o arreglo de strings, el i-avo string es la categoria de la i-ava imagen de entrenamiento.
    :return una matriz de tamaño (cantidad de ejemplos en labels, cantidad de categorias en 'labels') donde cada celda es 1 o 0. Hay 1 solo 1 por fila.
    """
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(labels) + 1))
    return label_binarizer.transform(labels)


# %% id="Emb12Xn9yLhM" executionInfo={"status": "ok", "timestamp": 1632436520220, "user_tz": 180, "elapsed": 23, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
def VGG_16():
    img_input = Input(shape=(224, 224, 3))  # tamaño de imagenes y 3 canales de colores

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

    # capa de clasificacion
    output = Flatten(name='flatten')(output)
    output = Dense(4096, activation='relu', name='fc1')(output)
    output = Dense(4096, activation='relu', name='fc2')(output)
    output = Dense(1000, activation='softmax', name='predictions')(output)

    return Model(inputs=img_input, outputs=output, name='vgg16')


# %% id="Tv7VlazpyORO" executionInfo={"status": "ok", "timestamp": 1632436523376, "user_tz": 180, "elapsed": 3178, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# leer el dataset de 210 imagenes de flores junto con y sus etiquetas
dataset = h5py.File(
    "/content/drive/MyDrive/collab/transfer_learning/imagenet_dataset/10FlowerColorImages.h5",
    'r',
)
# El [()] indica que lea el dataset completo a memoria, en vez de leerlo bajo demanda
images = dataset['images'][()]
image_labels = dataset['labels'][()]
images = rescalar_imagenes(images)
onehot_labels = encode_onehot_labels(image_labels)

# %% id="ZXfOOjk9y557" executionInfo={"status": "ok", "timestamp": 1632436523381, "user_tz": 180, "elapsed": 13, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# armar la arquitectura de la red neuronal
pretrained_vgg16 = VGG_16()

# %% id="AJdWYYMhy8NE" executionInfo={"status": "ok", "timestamp": 1632436525002, "user_tz": 180, "elapsed": 1631, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# El modelo tiene los pesos del entrenamiento con Caffe, donde el dataset tiene los pixels en orden 'BGR'
#  y c/pixel centrado sobre la media del dataset imagenet = [103.939, 116.779, 123.68]
# Las imagenes nuevas tienen que tener exactamente esta transformacion
pretrained_vgg16.load_weights(
    '/content/drive/MyDrive/collab/transfer_learning/imagenet_dataset/vgg16_weights.h5'
)

# %% id="JvblPEJ0i67t" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1632436525003, "user_tz": 180, "elapsed": 26, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}} outputId="045c488f-54e3-49ad-bc36-58307eff9130"
# Aqui eliminamos el block5 COMPLETO de VGG 16 (el ultimo bloque de convoluciones antes de clasificar),
# y conectamos la salida del block 4 a las 2 capas densas de clasificación.
# Lo eliminamos creando una nueva capa "replaced_flatten" cuya entrada es la salida de "block4_pool".
# De ahi para abajo repetimos el esquema de 3 capas de clasificación.
# Pueden ver en "summary" que el block 5 no está mas.
new_layer = Flatten(name='replaced_flatten')(
    pretrained_vgg16.get_layer("block4_pool").output
)
new_layer = Dense(4096, activation='relu', name='replaced_fc1')(new_layer)
new_layer = Dense(4096, activation='relu', name='replaced_fc2')(new_layer)
new_layer = Dense(10, activation="softmax", name="predict_10flowers")(new_layer)
pretrained_vgg16 = Model(pretrained_vgg16.input, new_layer)
pretrained_vgg16.summary()

# %% id="3096j1L7yVfx" executionInfo={"status": "ok", "timestamp": 1632436525005, "user_tz": 180, "elapsed": 23, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# setear a todas las capas, excepto las 3 ultimas capas de clasificacion
# como "no entrenable" (los pesos no se actualizaran)
for layer in pretrained_vgg16.layers[:-3]:
    layer.trainable = False

# crear un nuevo modelo cuya salida es la nueva capa
new_model = Model(inputs=pretrained_vgg16.input, outputs=new_layer)

# %% id="5q_Jonghyesi" executionInfo={"status": "ok", "timestamp": 1632436525006, "user_tz": 180, "elapsed": 24, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# compilar el modelo con SGD/momentum optimizer
# y un learning rate muuuuy lento
sgd = SGD(
    learning_rate=learning_rate,
    decay=learning_rate_decay,
    momentum=learning_rate_momentum,
    nesterov=True,
)
# regla en Keras:
# si loss=categorical_crossentropy => metrics=categorical_accuracy
# si loss=binary_crossentropy y mas de 2 clases => metrics=categorical_accuracy, sino metrics=binary_accuracy
new_model.compile(
    optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy']
)

# %% id="2iSzocdpyjei" executionInfo={"status": "ok", "timestamp": 1632436525007, "user_tz": 180, "elapsed": 23, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}}
# didivir el dataset en 70% entrenamiento y 30% test
X_train, X_test, y_train, y_test = train_test_split(
    images,
    onehot_labels,
    train_size=0.7,
    test_size=0.3,
    shuffle=True,
    stratify=image_labels,
)

# aumentar la cantidad de ejemplos de entrenamiento, inventando variaciones de las imagenes
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=[0.8, 1.4],
    horizontal_flip=True,
    rotation_range=10,
    preprocessing_function=preprocesar_imagen_como_caffe,
)

# testear con los casos de prueba sin rotar, cambiar tamaño ni nada, solo acomodados a la manera en que fue entrenada la NN
test_datagen = ImageDataGenerator(preprocessing_function=preprocesar_imagen_como_caffe)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size,
)

validation_generator = test_datagen.flow(X_test, y_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="nFgMzQNoymQm" executionInfo={"status": "ok", "timestamp": 1632436682940, "user_tz": 180, "elapsed": 157955, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}} outputId="b3a3340e-a00e-4ae8-922e-ce82c753f70d"
# entrenar y guarda el mejor resultado
new_model.fit(
    train_generator,
    steps_per_epoch=len(y_train) // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(y_test) // batch_size,
    use_multiprocessing=True,
    workers=num_cores,
    # parar el entrenamiento si no mejora en 3 epochs consecutivas; guardar el mejor modelo hasta ese momento al final de cada epoch
    callbacks=[
        EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=epochs_to_stop_after_no_improvement,
            verbose=1,
        ),
        ModelCheckpoint(
            '/content/drive/MyDrive/collab/transfer_learning/vgg16_retrained_10flowers_v2.h5',
            verbose=1,
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='auto',
        ),
    ],
)

print("Entrenamiento finalizado.")
