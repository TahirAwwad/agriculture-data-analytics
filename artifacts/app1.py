import streamlit as st
import pickle
import numpy as np
from predict_page import show_predict_page
from explore import show_explore_page









page = st.sidebar.selectbox("Choose Page",("Predction","Summary Stats"))

if page=='Summary Stats':
    show_explore_page()
else:
    show_predict_page()
    
