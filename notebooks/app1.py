import streamlit as st
from predict_page import show_predict_page


st.sidebar.selectbox("Choose Page",("Predction","EDA","Stats"))
show_predict_page()
