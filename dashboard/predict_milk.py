from sklearn.preprocessing import MinMaxScaler
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
        features_scaler: MinMaxScaler = pickle.load(file)

    with open(target_scaler_filepath, READ_BINARY) as file:
        target_scaler: MinMaxScaler = pickle.load(file)

    streamlit.sidebar.write(
        """### Change the sliders to best represent the expected costs of each Expense category""")

    taxes_on_products = streamlit.sidebar.slider(
        'Taxes on Products', 0, 252, 13)
    subsidies_on_products = 382.0193548387096
    compensation_of_employees = 429.62258064516146
    contract_work = 287.2451612903226
    entrepreneurial_income = 1950.2645161290322
    factor_income = 2754.2419354838707
    fixed_capital_consumption___farm_buildings = streamlit.sidebar.slider(
        'Farm Buildings', 0, 976, 49)
    fixed_capital_consumption___machinery__equipment__etc = streamlit.sidebar.slider(
        'Machinery, Equipment, etc', 0, 1014, 51)
    interest_less_fisim = 196.26451612903233
    operating_surplus = 2324.6225806451607
    livestock___cattle = streamlit.sidebar.slider(
        'Livestock - Cattle', 0, 4724, 237)
    livestock___sheep = streamlit.sidebar.slider(
        'Livestock - Sheep', 0, 607, 31)
    land_rental = streamlit.sidebar.slider('Land Rental', 0, 640, 32)
    intermediate_consumption___contract_work = 287.2451612903226
    intermediate_consumption___crop_protection_products = streamlit.sidebar.slider(
        'Intermediate Consumption - Crop Protection Products', 0, 165, 9)
    intermediate_consumption___energy_and_lubricants = streamlit.sidebar.slider(
        'Intermediate Consumption - Energy and Lubricants', 0, 910, 46)
    intermediate_consumption___feeding_stuffs = streamlit.sidebar.slider(
        'Intermediate Consumption - Feeding Stuffs', 0, 3349, 168)
    intermediate_consumption___fertilisers = streamlit.sidebar.slider(
        'Intermediate Consumption - Fertilisers', 0, 1228, 62)
    intermediate_consumption___financial_intermediation_services_indirect = 81.35483870967741
    intermediate_consumption___forage_plants = 822.2774193548387
    intermediate_consumption___maintenance_and_repairs = 351.4258064516129
    intermediate_consumption___veterinary_expenses = streamlit.sidebar.slider(
        'Intermediate Consumption - Veterinary Expenses', 0, 652, 33)
    intermediate_consumption___other_goods_and_services = 354.74193548387103

    ok = streamlit.sidebar.button("Calculate Milk production Value")

    if ok:
        values = [taxes_on_products, subsidies_on_products, compensation_of_employees, contract_work, entrepreneurial_income, factor_income,
                  fixed_capital_consumption___farm_buildings, fixed_capital_consumption___machinery__equipment__etc, interest_less_fisim, operating_surplus,
                  livestock___cattle, livestock___sheep, land_rental, intermediate_consumption___contract_work, intermediate_consumption___crop_protection_products,
                  intermediate_consumption___energy_and_lubricants, intermediate_consumption___feeding_stuffs, intermediate_consumption___fertilisers,
                  intermediate_consumption___financial_intermediation_services_indirect, intermediate_consumption___forage_plants,
                  intermediate_consumption___maintenance_and_repairs, intermediate_consumption___veterinary_expenses,
                  intermediate_consumption___other_goods_and_services]
        feature_values = numpy.array([values]).astype(float)
        scaled_feature_values = features_scaler.transform(feature_values)

        scaled_predicted_value = model.predict(scaled_feature_values)
        predicted_values = target_scaler.inverse_transform(
            scaled_predicted_value)
        predicted_value = predicted_values[0][0].astype(float)

        streamlit.subheader("The estimated value of Milk production (Millions) is ")
        streamlit.metric(label="", value=round(predicted_value, 2))
