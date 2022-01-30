from pandas import DataFrame, read_csv
import streamlit
import streamlit.components.v1 as components


directory: str = "./artifacts/jupyter-notebooks/"


def show_page():

    jupyter_notebooks_filepath = f'{directory}notebooks.csv'
    jupyter_notebooks_dataframe: DataFrame = read_csv(
        jupyter_notebooks_filepath)

    values = jupyter_notebooks_dataframe['Title'].tolist()
    options = jupyter_notebooks_dataframe['Filename'].tolist()

    dic = dict(zip(options, values))

    selected_notebook = directory + \
        streamlit.sidebar.radio(
            'Jupyter Notebooks', options, format_func=lambda x: dic[x])

    with open(selected_notebook, 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
        components.html(source_code, height=800, width=1000, scrolling=True)
