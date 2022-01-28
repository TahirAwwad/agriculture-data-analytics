from tensorflow import keras
import numpy
import pickle
import streamlit

READ_BINARY = "rb"

directory: str = "./artifacts/milk-production-models/"
features_scaler_filepath: str = f'{directory}milk-features-scaler.pickle'
target_scaler_filepath: str = f'{directory}milk-target-scaler.pickle'
model_filepath: str = f'{directory}milk-ann-model.h5'


def show_predict_page():

    model = keras.models.load_model(model_filepath)

    with open(features_scaler_filepath, READ_BINARY) as file:
        features_scaler = pickle.load(file)

    with open(target_scaler_filepath, READ_BINARY) as file:
        target_scaler = pickle.load(file)

    streamlit.sidebar.write(
        """### Change the sliders to best represent the expected costs of each Expense category""")

    taxes = streamlit.sidebar.slider("Taxes", 0, 50000, 1000)
    energy = streamlit.sidebar.slider("Energy", 0, 50000, 100)
    landrental = streamlit.sidebar.slider("Land Rental", 0, 50000, 1000)
    fertilizers = streamlit.sidebar.slider("Fertilizers", 0, 50000, 1000)
    feeding = streamlit.sidebar.slider("Feeding stuff", 0, 50000, 1000)

    ok = streamlit.sidebar.button("Calculate Milk production Value")

    if ok:
        values = [taxes, energy, landrental, fertilizers, feeding]
        feature_values = numpy.array([values]).astype(float)
        scaled_feature_values = features_scaler.transform(feature_values)

        scaled_predicted_value = model.predict(scaled_feature_values)
        predicted_values = target_scaler.inverse_transform(
            scaled_predicted_value)
        predicted_value = predicted_values[0][0].astype(float)

        streamlit.subheader("The estimated value of Milk production is ")
        streamlit.metric(label="", value=round(predicted_value, 2))
