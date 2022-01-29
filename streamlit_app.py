import streamlit
from dashboard import explore, predict_milk, predict_tb

page = streamlit.sidebar.selectbox(
    "Choose Page", ["Summary Stats", "Milk Prediction", "TB Prediction"])

if page == 'Summary Stats':
    explore.show_explore_page()
elif page == 'Milk Prediction':
    predict_milk.show_predict_page()
else:
    predict_tb.show_page()
