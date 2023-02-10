import numpy as np
import joblib
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'models/activities.pkl'

# Se reciben los valores y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1,-1)
    preds = model.predict(x)
    return preds


def main():
    model = ''

    # Se carga el modelo
    if model == '':
        with open(MODEL_PATH, 'rb') as file:
            model = joblib.load(file)

    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Predicción de Calorías: </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    actividad = st.text_input("Tipo de Actividad:")
    distancia = st.text_input("Distancia:")
    tiempo = st.text_input("Tiempo:")
    fcm = st.text_input("Frecuencia cardiaca media:")
    te = st.text_input("TE aeróbico:")
    ccm = st.text_input("Cadencia de carrera media:")
    ritmo = st.text_input("Ritmo medio:")
    lmz = st.text_input("Longitud media de zancada:")
    temperatura = st.text_input("Temperatura:")
    altura = st.text_input("Altura:")

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        x_in = [
            np.float_(actividad.title()),
            np.float_(distancia.title()),
            np.float_(tiempo.title()),
            np.float_(fcm.title()),
            np.float_(te.title()),
            np.float_(ccm.title()),
            np.float_(ritmo.title()),
            np.float_(lmz.title()),
            np.float_(temperatura.title()),
            np.float_(altura.title())
        ]
        predictS = model_prediction(x_in, model)
        st.success('La predicción de calorías es: {}'.format(predictS[0]))

if __name__ == '__main__':
    main()
