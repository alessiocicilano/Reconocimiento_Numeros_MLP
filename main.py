#!/usr/bin/env python3
"""
Reconocimiento de Números MNIST - Aplicación Simple

Carga una imagen de un número (0-9) y obtén la predicción.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Configuración para evitar problemas en macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NO importar tensorflow aquí - lo cargaremos solo cuando sea necesario

def cargar_modelo():
    """Carga o entrena el modelo (solo la primera vez)."""
    # Usar session_state para cachear el modelo
    if 'model' in st.session_state:
        return st.session_state.model
    
    # Mostrar spinner mientras carga
    with st.spinner('🔄 Cargando modelo... (primera vez puede tardar ~30 segundos)'):
        # Importar tensorflow solo cuando se necesita
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')  # Desactivar GPU en macOS
        from tensorflow import keras
        from src.data import load_mnist
        from src.interpreter import compile_model
        
        try:
            model = keras.models.load_model('models/modelo_mnist.h5', compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
            # Entrena el modelo
            trainX, trainY, testX, testY, input_dim = load_mnist(flatten=True)
            
            arch = "Dense(784,relu) -> Dense(128,relu) -> Dense(10,softmax)"
            model = compile_model(arch, input_dim=input_dim)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Entrenamiento (más épocas = mejor precisión)
            model.fit(trainX, trainY, validation_data=(testX, testY), 
                     epochs=10, batch_size=128, verbose=0)
            
            # Guarda en la carpeta models/
            os.makedirs('models', exist_ok=True)
            model.save('models/modelo_mnist.h5')
    
    # Guardar en session_state para no recargar
    st.session_state.model = model
    return model


def preprocesar_imagen(image: Image.Image) -> np.ndarray:
    """Prepara la imagen para la predicción."""
    # Convierte a escala de grises
    img = image.convert('L')
    
    # Redimensiona a 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Invierte colores si es necesario (MNIST: fondo negro, número blanco)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normaliza
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape para el modelo
    img_array = img_array.reshape(1, 784)
    
    return img_array


# ==================== INTERFAZ ====================
st.set_page_config(
    page_title="Reconocimiento de Números", 
    page_icon="🔢",
    menu_items={}  # Elimina el menú de desarrollador
)

# Oculta el menú de Streamlit completamente
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🔢 Reconocimiento de Números")
st.markdown("¡Carga una imagen de un número (0-9) y te diré qué número es!")

# Upload imagen
uploaded_file = st.file_uploader("📤 Carga una imagen", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file is not None:
    # Carga el modelo solo cuando se sube una imagen
    model = cargar_modelo()
    
    image = Image.open(uploaded_file)
    
    # Muestra imagen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tu imagen")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Procesada (28x28)")
        img_small = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        st.image(img_small, use_container_width=True)
    
    # Predicción
    img_preprocessed = preprocesar_imagen(image)
    predictions = model.predict(img_preprocessed, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100
    
    # Resultado
    st.markdown("---")
    st.markdown(f"# 🎯 El número es: **{predicted_digit}**")
    st.markdown(f"### Confianza: **{confidence:.1f}%**")
    
    # Barra de confianza
    st.progress(confidence / 100)

else:
    st.info("👆 Carga una imagen para empezar")

# Pie de página
st.markdown("---")
st.caption("💡 Sugerencia: usa números escritos sobre fondo claro para mejores resultados")
