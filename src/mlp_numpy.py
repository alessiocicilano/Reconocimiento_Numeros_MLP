"""
Componentes de MLP basados en NumPy (solo propagación hacia adelante):
- Activaciones: en src/activations.py
- Helper para neurona forward
- Layer: capa completamente conectada con activación
- MLP: stack secuencial de capas con predict()

Ejemplo de uso:
    from src.mlp_numpy import Layer, MLP
    from src.activations import relu, sigmoid
    import numpy as np
    x = np.random.rand(4, 3)
    mlp = MLP([
        Layer(3, 5, relu),
        Layer(5, 2, sigmoid),
    ])
    y = mlp.predict(x)
    print(y.shape)  # (4, 2)
"""

from __future__ import annotations

from typing import Callable, List
import numpy as np

Array = np.ndarray
Activation = Callable[[Array], Array]

from src.activations import sigmoid, relu


def neuron_forward(x: Array, w: Array, b: Array, activation: Activation) -> Array:
    """Propagación hacia adelante de una sola neurona densa (por lotes).

    Args:
        x: forma (batch_size, input_dim) o (input_dim,)
        w: forma (input_dim,) pesos para la neurona
        b: forma (,) sesgo (escalar en forma de array) o float
        activation: función de activación
    Returns:
        Salida activada, forma (batch_size,) o () según corresponda
    """
    if x.ndim == 1:
        z = float(x @ w) + float(b)
        return activation(np.array(z))  # type: ignore[arg-type]
    z = x @ w + b  # broadcasting b
    return activation(z)


class Layer:
    """Capa densa con función de activación."""
    
    def __init__(self, input_dim: int, output_dim: int, activation: Activation, *, seed: int | None = 42):
        """
        Inicializa una capa densa.
        
        Args:
            input_dim: Dimensión de entrada
            output_dim: Dimensión de salida (número de neuronas)
            activation: Función de activación
            seed: Semilla para reproducibilidad
        """
        rng = np.random.default_rng(seed)
        self.weights: Array = rng.normal(0.0, 0.1, size=(input_dim, output_dim))
        self.biases: Array = np.zeros((output_dim,), dtype=float)
        self.activation: Activation = activation

    def forward(self, x_batch: Array) -> Array:
        """
        Propagación hacia adelante a través de la capa.
        
        Args:
            x_batch: Datos de entrada (batch_size, input_dim)
        Returns:
            Salida activada (batch_size, output_dim)
        """
        z: Array = x_batch @ self.weights + self.biases
        return self.activation(z)


class MLP:
    """Perceptrón Multicapa - stack secuencial de capas."""
    
    def __init__(self, layers: List[Layer]):
        """
        Inicializa el MLP.
        
        Args:
            layers: Lista de capas Layer
        """
        self.layers = layers

    def predict(self, x: Array) -> Array:
        """
        Realiza la predicción (propagación hacia adelante completa).
        
        Args:
            x: Datos de entrada
        Returns:
            Salida de la última capa
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


