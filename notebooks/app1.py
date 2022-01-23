import streamlit as st
from predict_page import show_predict_page


st.sidebar.selectbox("Choose Page",("Predction","EDA","Stats"))



def load_model():
    pkl_model = pickle.load(open('pkl_ann_milk','rb'))
    return pkl_model

model = load_model()

pkl_scaler_y = pickle.load(open('tahir-api/notebooks/pkl_scaler_y','rb'))
 

pkl_scaler_x = pickle.load(open('tahir-api/notebooks/pkl_scaler_x','rb'))

show_predict_page()
