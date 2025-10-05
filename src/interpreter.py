"""
Intérprete de arquitecturas → Constructor de modelos Keras.

Sintaxis soportada (activaciones sin distinción de mayúsculas; espacios flexibles):
    Dense(unidades,activacion) -> Dense(unidades,activacion) -> ...

Ejemplo:
    arch = "Dense(784,relu) -> Dense(64,relu) -> Dense(10,softmax)"
    model = compile_model(arch, input_dim=784)
"""

from __future__ import annotations

import re
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow import keras

# Diccionario de activaciones soportadas
_ACTS = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "softmax": "softmax",
    "linear": "linear",
}

# Expresión regular para parsear capas Dense
_DENSE_RE = re.compile(r"\s*Dense\s*\(\s*(\d+)\s*,\s*([a-zA-Z_]+)\s*\)\s*")


def _parse_dense(token: str) -> Tuple[int, str]:
    """
    Parsea un token de capa Dense.
    
    Args:
        token: String con formato "Dense(unidades, activacion)"
    Returns:
        Tupla (unidades, activacion)
    Raises:
        ValueError: Si la sintaxis es inválida
    """
    m = _DENSE_RE.fullmatch(token)
    if not m:
        raise ValueError(f"Sintaxis inválida en capa: {token}")
    units = int(m.group(1))
    act_key = m.group(2).lower()
    if act_key not in _ACTS:
        raise ValueError(f"Activación no soportada: {act_key}")
    return units, _ACTS[act_key]


def compile_model(architecture_string: str, input_dim: int):
    """
    Compila un modelo Keras a partir de una descripción textual.
    
    Args:
        architecture_string: String con la arquitectura (ej. "Dense(784,relu) -> Dense(10,softmax)")
        input_dim: Dimensión de entrada
    Returns:
        Modelo Keras Sequential compilado
    Raises:
        ValueError: Si la arquitectura es inválida o está vacía
    """
    # Importar tensorflow solo cuando se llama esta función
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Divide por "->" y elimina espacios
    parts: List[str] = [p.strip() for p in architecture_string.split("->") if p.strip()]
    if not parts:
        raise ValueError("La arquitectura no puede estar vacía")

    # Construye el modelo secuencialmente
    model = keras.Sequential()
    first = True
    for token in parts:
        units, activation = _parse_dense(token)
        if first:
            # Primera capa: especifica input_shape
            model.add(layers.Dense(units, activation=activation, input_shape=(input_dim,)))
            first = False
        else:
            # Capas siguientes
            model.add(layers.Dense(units, activation=activation))
    return model


