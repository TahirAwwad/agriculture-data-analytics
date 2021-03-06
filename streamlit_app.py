import streamlit
from dashboard import explore, predict_milk, predict_tb, jupyter_notebooks, sentiment_analysis

streamlit.set_page_config(
    page_title='Agriculture Data Analytics Dashboard', layout='wide')

pages = ["Summary Stats", "Milk Prediction",
         "TB Prediction", "Sentiment Analysis", "Jupyter Notebooks"]

page = streamlit.sidebar.selectbox("Navigation", pages)

if page == 'Summary Stats':
    explore.show_page()
elif page == 'Sentiment Analysis':
    sentiment_analysis.show_page()
elif page == 'Milk Prediction':
    predict_milk.show_page()
elif page == 'Jupyter Notebooks':
    jupyter_notebooks.show_page()
else:
    predict_tb.show_page()
