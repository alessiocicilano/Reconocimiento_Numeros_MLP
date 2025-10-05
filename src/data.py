"""
Utilidades para datos MNIST: carga, normalización, one-hot, aplanamiento opcional.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def load_mnist(*, flatten: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Carga y preprocesa el dataset MNIST.
    
    Args:
        flatten: Si True, aplana las imágenes de (28,28) a (784,)
    Returns:
        Tupla con:
        - trainX: Datos de entrenamiento
        - trainY: Etiquetas de entrenamiento (one-hot)
        - testX: Datos de prueba
        - testY: Etiquetas de prueba (one-hot)
        - dim_input: Dimensión de entrada
    """
    # Importar tensorflow solo cuando se llama esta función
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    
    # Carga el dataset MNIST
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    # Normaliza los píxeles de [0, 255] a [0, 1]
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    if flatten:
        # Aplana las imágenes de (28, 28) a (784,)
        dim_input = trainX.shape[1] * trainX.shape[2]
        trainX = trainX.reshape(trainX.shape[0], dim_input)
        testX = testX.reshape(testX.shape[0], dim_input)
    else:
        dim_input = trainX.shape[1] * trainX.shape[2]

    # Convierte etiquetas a formato one-hot
    trainY_c = to_categorical(trainY, num_classes=10)
    testY_c = to_categorical(testY, num_classes=10)

    return trainX, trainY_c, testX, testY_c, dim_input


