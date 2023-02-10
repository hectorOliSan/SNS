import numpy as np
import joblib
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'models/pinguinos.pkl'

# Tipo de especies
ESPECIE = ['Adelie', 'Chinstrap', 'Gentoo']

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
    <h1 style="color:#181082;text-align:center;">Especies de Pingüino: </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    bill_length_mm = st.text_input("Valor de bill_length_mm:")
    bill_depth_mm = st.text_input("Valor de bill_depth_mm:")
    flipper_length_mm = st.text_input("Valor de flipper_length_mm:")
    body_mass_g = st.text_input("Valor de body_mass_g:")

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        x_in = [np.float_(bill_length_mm.title()),
            np.float_(bill_depth_mm.title()),
            np.float_(flipper_length_mm.title()),
            np.float_(body_mass_g.title())]
        predictS = model_prediction(x_in, model)
        st.success('La predicción de especie es: {}'.
            format(ESPECIE[predictS[0]]).upper())

if __name__ == '__main__':
    main()
