import streamlit
from dashboard import explore, predict_page, predict_tb

page = streamlit.sidebar.selectbox(
    "Choose Page", ["Summary Stats", "Milk Prediction", "TB Prediction"])

if page == 'Summary Stats':
    explore.show_explore_page()
elif page == 'Milk Prediction':
    predict_page.show_predict_page()
else:
    predict_tb.show_page()
