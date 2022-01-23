import streamlit as st
import pickle
import numpy as np
from predict_page import show_predict_page


st.sidebar.selectbox("Choose Page",("Predction","EDA","Stats"))


show_predict_page()
