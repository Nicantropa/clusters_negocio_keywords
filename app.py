# app.py
import streamlit as st

# --- Configuración de la Página ---
# Esto debe ser el primer comando de Streamlit en tu script.
st.set_page_config(
    page_title="Pipeline de Clustering de Negocio",
    page_icon="📊",
    layout="wide"
)

# --- Página Principal ---

# Título principal de la aplicación
st.title("Bienvenido al Pipeline de Clustering 🚀")
st.write("---")

# Mensaje en la barra lateral para guiar al usuario
st.sidebar.success("Selecciona un paso del pipeline para comenzar.")

# Sección de bienvenida y explicación
st.header("¿Qué hace esta herramienta?")
st.markdown(
    """
    Esta aplicación te permite ejecutar un pipeline completo de Machine Learning para segmentar clientes o datos de negocio.
    Cada paso del proceso está separado en una página diferente para mayor claridad.
    
    **El flujo de trabajo es el siguiente:**
    1.  **Consolidación y Limpieza:** Carga y prepara tus datos iniciales.
    2.  **Ingeniería de Características:** Crea nuevas variables relevantes para el modelo.
    3.  **Exploración de Datos (EDA):** Visualiza y entiende tus datos.
    4.  **Modelo de Clustering:** Ejecuta el modelo de segmentación y analiza los resultados.
    """
)

st.write("---")

# Instrucción clara para el siguiente paso
st.info(
    "👈 **Para empezar, selecciona '1_consolidacion_limpieza' en la barra lateral.**",
    icon="💡"
)
