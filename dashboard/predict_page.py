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
        pkl_scaler_x = pickle.load(file)

    with open(target_scaler_filepath, READ_BINARY) as file:
        pkl_scaler_y = pickle.load(file)

    #streamlit.title("Milk production forcast")
    streamlit.sidebar.write(
        """### Change the sliders to best represent the expected costs of each Expence catagory""")

    #countries = ('Ireland','Germany','United Kingdom','Poland','Spain')

    #country = streamlit.sidebar.selectbox('Countries',countries)
    taxes = streamlit.sidebar.slider("Taxes", 0, 50000, 1000)
    Energy = streamlit.sidebar.slider("Energy", 0, 50000, 100)
    #Sheep = streamlit.sidebar.slider("Livestock Sheep",0,50000,1000)
    landrental = streamlit.sidebar.slider("Land Rental", 0, 50000, 1000)
    Fertilisers = streamlit.sidebar.slider("Fertilisers", 0, 50000, 1000)
    feeding = streamlit.sidebar.slider("Feeding stuff", 0, 50000, 1000)
    #vetexp = streamlit.slider("Veterinary Expenses",0,50000,100)
    #Subsidies = streamlit.slider("Subsidies",0,50000,100)
    #Compensation = streamlit.slider("Compensation of Employees",0,50000,100)
    #Contract = streamlit.slider("Contract Work",0,50000,100)
    #Entrepreneurial = streamlit.slider("Entrepreneurial Income",0,50000,100)
    #Factor = streamlit.slider("Factor Income",0,50000,100)
    #Buildings = streamlit.slider("Farm Buildings",0,50000,100)
    #Equipment = streamlit.slider("Farm Equipment",0,50000,100)
    #FISIM = streamlit.slider("FISIM",0,50000,100)
    #Surplus = streamlit.slider("Surplus",0,50000,100)
    #Cattle = streamlit.slider("Cattle",0,50000,100)
    #CropProtection = streamlit.slider("Crop Protection",0,50000,100)

    #Financial = streamlit.slider("Financial Intermediation Services",0,50000,100)
    #Forage = streamlit.slider("Forage Plants",0,50000,100)
    #Maintenance = streamlit.slider("Maintenance",0,50000,100)
    #Seeds = streamlit.slider("Seeds",0,50000,100)
    #OtherGoods = streamlit.slider("Other Goods",0,50000,100)

    ok = streamlit.sidebar.button("Calculate Milk production Value")

    if ok:
        newdata = numpy.array([[taxes, Energy, landrental, Fertilisers, feeding  # ,vetexp ,Subsidies ,Compensation,Contract,Entrepreneurial,Factor,Buildings,Equipment,FISIM , Surplus ,Cattle ,CropProtection ,Energy,Financial ,Forage ,Maintenance ,Seeds,OtherGoods
                                ]])
        newdata = newdata.astype(float)

        newdata_scaled = pkl_scaler_x.transform(newdata)
        predicted_scaled = model.predict(newdata_scaled)
        # descale the predicted value back to original format
        predicted_transformed = pkl_scaler_y.inverse_transform(
            predicted_scaled).astype(float)
        streamlit.subheader("The estimated value of Milk production is ")
        streamlit.metric(label="", value=round(predicted_transformed[0][0], 2))
        # streamlit.metric(value=predicted_transformed)