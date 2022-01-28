from tensorflow import keras
import streamlit as st
import pickle
import numpy as np

READ_BINARY = "rb"

directory: str = "./artifacts/milk-production-models/"
features_scaler_filepath: str = f'{directory}milk-features-scaler.pickle'
target_scaler_filepath: str = f'{directory}milk-target-scaler.pickle'
model_filepath: str = f'{directory}milk-ann-model.h5'


model = keras.models.load_model(model_filepath)

with open(features_scaler_filepath, READ_BINARY) as file:
    pkl_scaler_x = pickle.load(file)

with open(target_scaler_filepath, READ_BINARY) as file:
    pkl_scaler_y = pickle.load(file)


def show_predict_page():

    #st.title("Milk production forcast")
    st.sidebar.write(
        """### Change the sliders to best represent the expected costs of each Expence catagory""")

    #countries = ('Ireland','Germany','United Kingdom','Poland','Spain')

    #country = st.sidebar.selectbox('Countries',countries)
    taxes = st.sidebar.slider("Taxes", 0, 50000, 1000)
    Energy = st.sidebar.slider("Energy", 0, 50000, 100)
    #Sheep = st.sidebar.slider("Livestock Sheep",0,50000,1000)
    landrental = st.sidebar.slider("Land Rental", 0, 50000, 1000)
    Fertilisers = st.sidebar.slider("Fertilisers", 0, 50000, 1000)
    feeding = st.sidebar.slider("Feeding stuff", 0, 50000, 1000)
    #vetexp = st.slider("Veterinary Expenses",0,50000,100)
    #Subsidies = st.slider("Subsidies",0,50000,100)
    #Compensation = st.slider("Compensation of Employees",0,50000,100)
    #Contract = st.slider("Contract Work",0,50000,100)
    #Entrepreneurial = st.slider("Entrepreneurial Income",0,50000,100)
    #Factor = st.slider("Factor Income",0,50000,100)
    #Buildings = st.slider("Farm Buildings",0,50000,100)
    #Equipment = st.slider("Farm Equipment",0,50000,100)
    #FISIM = st.slider("FISIM",0,50000,100)
    #Surplus = st.slider("Surplus",0,50000,100)
    #Cattle = st.slider("Cattle",0,50000,100)
    #CropProtection = st.slider("Crop Protection",0,50000,100)

    #Financial = st.slider("Financial Intermediation Services",0,50000,100)
    #Forage = st.slider("Forage Plants",0,50000,100)
    #Maintenance = st.slider("Maintenance",0,50000,100)
    #Seeds = st.slider("Seeds",0,50000,100)
    #OtherGoods = st.slider("Other Goods",0,50000,100)

    ok = st.sidebar.button("Calculate Milk production Value")

    if ok:
        newdata = np.array([[taxes, Energy, landrental, Fertilisers, feeding  # ,vetexp ,Subsidies ,Compensation,Contract,Entrepreneurial,Factor,Buildings,Equipment,FISIM , Surplus ,Cattle ,CropProtection ,Energy,Financial ,Forage ,Maintenance ,Seeds,OtherGoods
                             ]])
        newdata = newdata.astype(float)
        # scale the input
        newdata_scaled = pkl_scaler_x.transform(newdata.reshape(-1, 1))
        predicted_scaled = model.predict(newdata_scaled)
        # descale the predicted value back to original format
        predicted_transformed = pkl_scaler_y.inverse_transform(
            predicted_scaled).astype(float)
        st.subheader("The estimated value of Milk production is ")
        st.metric(label="", value=round(predicted_transformed[0][0], 2))
        # st.metric(value=predicted_transformed)