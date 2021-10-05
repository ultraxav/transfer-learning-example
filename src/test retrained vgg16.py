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

# %% id="KiLC1RMMFNRU" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631836546894, "user_tz": 180, "elapsed": 3475, "user": {"displayName": "Fernando Das Neves", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "06324247755041763216"}} outputId="72ec99fa-25c7-4fdd-acb8-eed5583affbf"
# Este notebook lee un modelo de keras ya entrenado, y lo testea contra el dataset de holdout the "10 flowers".

# Antes de ejecutar: Activar GPUs como sigue:
# menu "Entorno de Ejecucion" -> "Cambiar tipo de entorno de ejecucion" -> "Acelerador de Hardware" = "GPU"

# %tensorflow_version 2.x
import tensorflow as tf

print("Usandor Tensorflow version " + tf.__version__)


if tf.test.gpu_device_name():
    print('Usando GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print("Usando CPU.")

# %% id="sn9HyNB-Usq6"
from keras.models import load_model
import h5py
import numpy as np

from typing import List
from skimage.transform import resize
from keras.preprocessing import image
import numpy as np
from os.path import join
from typing import List, Tuple


def read_target_names(target_names_file: str) -> List[str]:
    """
    Lee el archivo con los nombres de categoria. Asume que es un archivo de texto
     donde la primera linea es el nombre de la categoria 0, la 2da linea el nombre de la categoria 1, etc.
    :return Una lista de nombres de categorias.
    """
    target_names = []
    with open(target_names_file, "rt") as f:
        for target_name in f:
            target_name = target_name.strip()
            if len(target_name) > 0:
                target_names.append(target_name)
    return target_names


def preprocesar_imagen_como_caffe(image: np.ndarray) -> np.ndarray:
    """
    Transforma las imagenes, aplicando las mismas transformaciones con las que fue entrenado el modelo de VGG16 que estamos usando.
    :param image: Una imagen representada como una matriz de (largo en pixels, alto en pixels, 3 canales)
    :return La imagen transformada.
    """
    # pasar imagen de  'RGB'->'BGR', porque el modelo ya entrenado de VGG16 que estamos usando proviene de Caffe, y fue entrenado en ese orden de channels
    image = image[:, :, ::-1]
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    # central valor de los pixels alrededor del valor medio de cada canal en el conj. de entrenamiento,
    # esto se calcula simplemente promediando todos los valores de cada canal en todas las imagenes de entrenamiento en imagenet.
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    return image


def decode_predictions(preds, labels, top=1) -> List[List[Tuple[str, float]]]:
    """
    Transforma las predicciones en nombres de categoria, y retorna las 'top' mejores.
    :param preds: Un arreglo de numpy con el valor predicho por la NN para cada categoria.
    :param labels: Los nombres de cata categoria: debe haber tantos elementos en 'labels' como en 'preds'.
    :return Una lista de pares (nombre de la categoria, peso retornado por la NN para esa categoria), conteniendo los 'top' mejores resultados.
    """
    if len(preds.shape) != 2 or preds.shape[1] != len(labels):
        raise ValueError(
            'Debe haber el mismo numero de categoerias que predicciones del modelo. '
            + 'El modelo retornó {} predicciones, pero hay {} categorias'.format(
                preds.shape[1], len(labels)
            )
        )
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:]
        result = [(labels[i], pred[i]) for i in top_indices]
        results.append(result)
    return results


def get_predictions_for_image(
    keras_trained_model, img: np.ndarray, labels: List[str], top_n=1
) -> List[Tuple[str, float]]:
    """
    Clasifica una imagen y retorna las 'top' mejores categorias, segun el modelo entrenado.
    :param keras_trained_model: Un modelo de Keras ya entrenado.
    :param img: Un arreglo de numpy conteniendo la imagen a clasificar.
    :param labels: Una lista conteniendo el nombre de cada categoria que puede predecir el modelo.
    :param top: Cuantos de los mejores resultados de clasificación retornar.
    :return Una lista de pares (nombre de la categoria, peso retornado por la NN para esa categoria), conteniendo los 'top' mejores resultados.
    """
    # agregar una dimension a img, porque por como fue entrenado el modelo,
    # la entrada que espera el modelo es siempre  de 4 dimensiones: (nro_imagenes, alto, largo, 3)
    img = np.expand_dims(img, axis=0)
    return decode_predictions(keras_trained_model.predict(img), labels, top_n)


# %% id="s28lpqWLFPA6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1626914833359, "user_tz": 180, "elapsed": 17608, "user": {"displayName": "Fernando Das Neves", "photoUrl": "", "userId": "06324247755041763216"}} outputId="4f5b64ea-6e38-4b49-8f96-f6df62145edf"


# asume que el archivo con el modelo entrenado esta almacenado en gdrive.
# Si el archivo no esta esta en grdive, subalo.
model_file = (
    "/content/drive/MyDrive/collab/transfer_learning/vgg16_retrained_10flowers.h5"
)
cnn_model = load_model(model_file)
# los nombres de clase estan en google drive
target_names = read_target_names(
    "/content/drive/MyDrive/collab/transfer_learning/flowers_dataset/flower_class_index.txt"
)

# leer el dataset de test con imagenes de flores junto con y sus clases (de 0 a 9) desde google drive
dataset = h5py.File(
    "/content/drive/MyDrive/collab/transfer_learning/flowers_dataset/data/test/flowers_holdout.h5",
    'r',
)

# El [()] hace que se lea el dataset completo a memoria, en vez de leerlo bajo demanda
images = dataset['images'][()]
image_labels = dataset['labels'][()]
image_filenames = dataset['filenames'][()]
tp = 0
for i in range(0, images.shape[0]):
    image = preprocesar_imagen_como_caffe(images[i])
    clase_real = target_names[image_labels[i, 0]]
    clase_predicha: Tuple[str, float] = get_predictions_for_image(
        cnn_model, image, target_names
    )[0]
    if clase_real == clase_predicha[0][0]:
        tp += 1
    print(
        "'{}' clase real:'{}', clase predicha:{}".format(
            image_filenames[i][0].decode('utf-8'), clase_real, clase_predicha
        )
    )
print("Accuracy =", float(tp) / images.shape[0])
