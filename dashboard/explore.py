from pandas import read_csv, DataFrame, to_datetime
from matplotlib import pyplot
import streamlit


def load_data():
    filepath: str = "./artifacts/irish-milk-production-eda-output.csv"
    dataframe: DataFrame = read_csv(filepath)

    milk_columns = ['Year',
                    'All Livestock Products - Milk',
                    'Taxes on Products',
                    'Subsidies on Products',
                    'Compensation of Employees',
                    'Contract Work',
                    'Entrepreneurial Income',
                    'Factor Income',
                    ]

    dataframe = dataframe[milk_columns]
    dataframe.set_index('Year', drop=True, inplace=True)
    dataframe.index = to_datetime(dataframe.index, format='%Y')
    return dataframe


def show_explore_page():

    dataframe: DataFrame = load_data()

    streamlit.header("Milk Production - Summary Statistics")

    streamlit.subheader("Data")
    streamlit.dataframe(dataframe.head(5))

    streamlit.subheader("Stats")
    streamlit.dataframe(dataframe.describe())

    streamlit.subheader("Graph")
    streamlit.line_chart(dataframe, use_container_width=True)

    streamlit.subheader("Histogram")
    figure, axis = pyplot.subplots()
    axis.hist(dataframe.iloc[:, 0], bins=20)
    streamlit.pyplot(figure)
