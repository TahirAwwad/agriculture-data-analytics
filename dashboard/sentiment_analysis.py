from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot
import streamlit
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt


def load_data():
    filepath: str = "./artifacts/ifa-ie-articles.csv"
    dataframe: DataFrame = read_csv(filepath)

    return dataframe


def show_page():

    dataframe: DataFrame = load_data()

    streamlit.header("Sentiment Analysis")
    streamlit.write("The sentiment score varied between –1.0 to 1.0, with –1.0 as the most negative text and 1.0 as the most positive text. As seen below, most articles had a sentiment score above 0.5 which would indicate a positive outlook of the text data. We can also see that dairy articles contained less negative sentiment when compared to cattle, nevertheless the positive sentiment density is higher in the cattle related articles. ")

    figure, axis = plt.subplots()
    sns.distplot(dataframe[dataframe.Trend == 'cattle']
                 ['compound'], label='cattle')
    sns.distplot(dataframe[dataframe.Trend == 'dairy']
                 ['compound'], label='dairy')
    plt.legend()
    plt.show()

    streamlit.plotly_chart(figure)

    streamlit.write("We have proven in our null hypothesis test that the sentiment is the same for the dairy and cattle sector. As seen on the wordclouds the sentiment is similar for both sectors. The word tokens for show positives attitude. The predominant tokens are ‘increase’, ‘price’, ‘demand’, ‘production’. This could indicate the dairy and cattle sectors are experiencing growth. By looking at the wordclouds we can see an overlap of the word tokens, this fact could give another proof the sentiments for both sectors are similar")

    image = Image.open('./artifacts/sentiment-analysis/cattle-word-cloud.png')
    streamlit.image(image, caption='Cattle/Dairy word cloud comparison')
