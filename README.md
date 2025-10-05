# 🧠 Intérprete MLP - Reconocimiento de Números MNIST

## 📝 ¿Qué hace este proyecto?

Una aplicación web que:
1. **Entrena una red neuronal** para reconocer números escritos a mano (0-9)
2. **Permite cargar una imagen** de un número
3. **Indica qué número es** con el porcentaje de confianza

---

## 🚀 Cómo iniciarlo (PRIMERA VEZ)

### 🪟 Windows (CMD):
```cmd
cd ruta\del\proyecto
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

### 🍎 macOS / 🐧 Linux (Terminal):
```bash
cd ruta/del/proyecto
pip install -r requirements.txt
python3 -m streamlit run main.py
```

---

## 💡 Cómo usarlo

1. La aplicación se abre en el navegador
2. **Carga una imagen** de un número (0-9)
3. **Resultado** → Te dice qué número es con la confianza

**Nota**: El primer inicio puede tardar ~1 minuto (entrenamiento del modelo)

---

## 📁 Estructura del proyecto

```
├── main.py                    ← Aplicación principal
├── requirements.txt           ← Dependencias
├── src/
│   ├── activations.py         ← Funciones de activación (Fase 1)
│   ├── mlp_numpy.py           ← Red neuronal NumPy (Fase 1)
│   ├── interpreter.py         ← Parser de arquitecturas (Fase 2)
│   └── data.py                ← Dataset MNIST (Fase 3)
└── models/modelo_mnist.h5     ← Modelo guardado
```

---

## 🎯 Fases del proyecto

Este proyecto implementa las **3 fases requeridas**:

### **Fase 1: MLP desde Cero con NumPy**
- ✅ Funciones de activación: `sigmoid`, `relu`
- ✅ Clase `Layer`: capa densa con activación
- ✅ Clase `MLP`: stack de capas con método `predict()`
- 📄 Archivos: `src/activations.py`, `src/mlp_numpy.py`

### **Fase 2: Intérprete de Arquitecturas**
- ✅ Parser que lee sintaxis: `Dense(units, activation) -> Dense(...)`
- ✅ Función `compile_model()` que genera modelo Keras dinámicamente
- 📄 Archivo: `src/interpreter.py`

**Ejemplo de sintaxis:**
```python
arch = "Dense(784,relu) -> Dense(128,relu) -> Dense(10,softmax)"
model = compile_model(arch, input_dim=784)
```

### **Fase 3: Entrenamiento en MNIST**
- ✅ Carga y preprocesamiento del dataset MNIST (60.000 imágenes)
- ✅ Normalización y one-hot encoding
- ✅ Entrenamiento con TensorFlow/Keras
- ✅ Evaluación del modelo
- ✅ **Extra**: Interfaz web para reconocer números de imágenes
- 📄 Archivos: `src/data.py`, `main.py`

---

## 📚 Librerías utilizadas

- `numpy` - Cálculos matemáticos y operaciones matriciales
- `tensorflow/keras` - Framework de deep learning
- `matplotlib` - Visualización de gráficos
- `scikit-learn` - Métricas de evaluación
- `streamlit` - Interfaz web interactiva
- `pillow` - Procesamiento de imágenes

Todas basadas en los notebooks proporcionados por el profesor.

---

## 🎓 Notas para el profesor

Este proyecto sigue completamente la estructura y conceptos de los notebooks:
- `(MLP)Ejercicio1_classify_numbers (2).ipynb`
- `Sesion17_Bloque3_Convolutional_Neural_Networks_BigData (1).ipynb`

**Implementaciones basadas en el material del curso:**
- Estructura de clases `Layer` y `MLP` 
- Preprocesamiento MNIST y normalización 
- Técnicas de entrenamiento con Keras 

**Contribución adicional (requerida por la actividad):**
- Intérprete de arquitecturas (Fase 2)
- Interfaz web para demostración práctica
