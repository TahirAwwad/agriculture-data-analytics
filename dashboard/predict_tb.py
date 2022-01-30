from tensorflow import keras
from data_analytics.machine_learning import build_dummy_list
import numpy
import pickle
import streamlit

READ_BINARY = "rb"

directory: str = "./artifacts/county-bovine-tb-models/"
features_scaler_filepath: str = f'{directory}features-scaler.pickle'
target_scaler_filepath: str = f'{directory}target-scaler.pickle'
model_filepath: str = f'{directory}ann-model.h5'


def show_page():

    model = keras.models.load_model(model_filepath)

    with open(features_scaler_filepath, READ_BINARY) as file:
        features_scaler = pickle.load(file)

    with open(target_scaler_filepath, READ_BINARY) as file:
        target_scaler = pickle.load(file)

    streamlit.sidebar.write(
        """### Change the sliders to best represent the expected costs of each Expense category""")

    veterinary_office_values = ["Carlow", "Cavan", "Clare", "Cork North", "Cork South",
                                "Donegal", "Dublin", "Galway", "Kerry", "Kildare",
                                "Kilkenny", "Laois", "Leitrim", "Limerick", "Longford",
                                "Louth", "Mayo", "Meath", "Monaghan", "Offaly",
                                "Roscommon", "Sligo", "Tipperary North", "Tipperary South",
                                "Waterford", "Westmeath", "Wexford", "Wicklow E", "Wicklow W", ]

    veterinary_office = streamlit.sidebar.selectbox(
        'Veterinary Office', veterinary_office_values)
    animal_count = streamlit.sidebar.slider('Animal Count', 0, 1235702, 61786)
    restricted_herds_at_end_of_year = streamlit.sidebar.slider(
        'Restricted Herds at end of Year', 0, 526, 27)
    restricted_herds_at_start_of_year = streamlit.sidebar.slider(
        'Restricted Herds at start of Year', 0, 926, 47)
    herds_tested = streamlit.sidebar.slider('Herds Tested', 0, 23278, 1164)
    herds_count = streamlit.sidebar.slider('Herds Count', 0, 23592, 1180)
    reactors_per_1000_tests_a_p_t = streamlit.sidebar.slider(
        'Reactors per 1000 Tests A.P.T.', 0, 19, 1)
    reactors_to_date = streamlit.sidebar.slider(
        'Reactors to date', 0, 5998, 300)
    tests_on_animals = streamlit.sidebar.slider(
        'Tests on Animals', 0, 1759682, 87985)

    ok = streamlit.sidebar.button("Calculate Head TB Rate Value")

    if ok:
        values = [animal_count, restricted_herds_at_end_of_year, restricted_herds_at_start_of_year,
                  herds_tested, herds_count, reactors_per_1000_tests_a_p_t, reactors_to_date, tests_on_animals] + build_dummy_list(veterinary_office_values, veterinary_office)
        feature_values = numpy.array([values]).astype(float)
        scaled_feature_values = features_scaler.transform(feature_values)

        scaled_predicted_value = model.predict(scaled_feature_values)
        predicted_values = target_scaler.inverse_transform(
            scaled_predicted_value)
        predicted_value = predicted_values[0][0].astype(float)

        streamlit.subheader("The estimated Herd Incidence Rate is ")
        streamlit.metric(label="", value=round(predicted_value, 2))
