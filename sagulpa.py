import numpy as np
import joblib
import streamlit as st

# Path de los modelos preentrenados
MODEL_PATH_CHECKIN = 'models/modelo_ci.pkl'
MODEL_PATH_CHECKOUT = 'models/modelo_co.pkl'

# Path del escalador
SCALER_PATH = 'models/scaler.pkl'

# Nombres de Parkings
PARKINGS = ['ELDER', 'MATA', 'METROPOL', 'RINCÓN', 'SANAPÚ', 'VEGUETA']

# Nombres de Días Semana
DAYWEEKS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']


# Se reciben los valores y el modelo, se escalan los datos y devuelve la predicción
def model_prediction(x_in, model):

    # Carga del objeto scaler guardado
    with open(SCALER_PATH, 'rb') as file:
        scaler = joblib.load(file)

    # Escalamiento de los datos de entrada
    x_in = np.array(x_in).reshape(1, -1)
    x_in = scaler.transform(x_in)

    st.info('Datos escalados: {}'.format(x_in[0]))
    x = np.asarray(x_in).reshape(1,-1)
    return int(model.predict(x))


def main():
    model_ci = ''
    model_co = ''

    # Se cargan los modelos
    if model_ci == '':
        with open(MODEL_PATH_CHECKIN, 'rb') as file:
            model_ci = joblib.load(file)

    if model_co == '':
        with open(MODEL_PATH_CHECKOUT, 'rb') as file:
            model_co = joblib.load(file)

    # Título
    html_temp = """
    <h1 style="color:#333b4b;text-align:center;">Predicción de Entradas/Salidas: </h1>
    <hr>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    prediction = st.selectbox(
        "Predicción para (CheckIn/CheckOut):",
        ('Entradas', 'Salidas')
    )
    parking = st.selectbox(
        "Nombre del Parking:",
        ('ELDER', 'MATA', 'METROPOL', 'RINCÓN', 'SANAPÚ', 'VEGUETA')
    )
    day = st.number_input("Día:", min_value=1, max_value=31)
    dayweek = st.selectbox(
        "Día de la Semana:",
        ('Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo')
    )
    timeslot = st.number_input("Hora:", min_value=0, max_value=23)
    holiday = st.checkbox("Festivo (S/N)")
    schoolday = st.checkbox("Lectivo (S/N)")
    temperature = st.number_input("Temperatura (ºC):", min_value=15, max_value=25)
    humidity = st.number_input("Humedad (%):", min_value=0.0, max_value=1.0, step=0.01)

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        x_in = [
            np.float_(PARKINGS.index(parking.title().upper())),
            np.float_(day),
            np.float_(DAYWEEKS.index(dayweek.title())+1),
            np.float_(timeslot),
            np.float_(holiday),
            np.float_(schoolday),
            np.float_(temperature),
            np.float_(humidity)
        ]
        st.info('Datos seleccionados: {}'.format(x_in))

        if prediction.title() == 'Entradas':
            predictS = model_prediction(x_in, model_ci)

        if prediction.title() == 'Salidas':
            predictS = model_prediction(x_in, model_co)

        st.success('La predicción de ' + prediction.title() + ' es: {}'.format(predictS))

if __name__ == '__main__':
    main()
