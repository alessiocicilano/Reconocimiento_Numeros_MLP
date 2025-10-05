"""
Funciones de activación sin estado para MLP con NumPy.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def sigmoid(z: Array) -> Array:
    """Función de activación Sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: Array) -> Array:
    """Función de activación ReLU (Rectified Linear Unit)."""
    return np.maximum(0.0, z)


