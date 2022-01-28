import streamlit
from dashboard import explore, predict_page

page = streamlit.sidebar.selectbox(
    "Choose Page", ["Predction", "Summary Stats"])

if page == 'Summary Stats':
    explore.show_explore_page()
else:
    predict_page.show_predict_page()
