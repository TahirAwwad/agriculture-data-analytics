#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)


app = Dash(__name__)


# -- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")

#df = pd.read_csv("../data/share-of-land-area-used-for-agriculture.csv")
#df.rename(columns={'Agricultural land (% of land area)': 'Percent Pasture'}, inplace=True)

df = pd.read_csv("../data/area-meadows-and-pastures.csv")
df.rename(columns={'Land use indicators - Land under perm. meadows and pastures - 6655 - Share in Land area - 7209 - %': 'Percent Pasture'}, inplace=True)



df.reset_index(inplace=True)


print(df[:5])


# App layout
app.layout = html.Div([

    html.H1("Percentage Land Devoted to Pasture", style={'text-align': 'center'}),


 
dcc.Slider(
        id='slct_year',
        min=1961,
        max=2017,
        step=1,
        value=1961,
    marks={
        1961:  {'label': '1961'},
        1965: {'label': '1965'},
        1970: {'label': '1970'},
        1975: {'label': '1975'},
        1980: {'label': '1980'},
        1985: {'label': '1985'},
        1990: {'label': '1990'},
        1995: {'label': '1995'},
        2000: {'label': '2000'},
        2005: {'label': '2005'},
        2010: {'label': '2010'},
        2015: {'label': '2015'},
        2017: {'label': '2017'}
    }
    
        ), 
    
#    dcc.Dropdown(id="slct_year",
#                 options=[
#                     {"label": "2015", "value": 2015},
#                     {"label": "2016", "value": 2016},
#                     {"label": "2017", "value": 2017}],
#                 multi=False,
#                 value=2015,
#                 style={'width': "40%"}
#                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='Percentage Pasture', figure={})

])

# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='Percentage Pasture', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    # dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locations='Code',
        color="Percent Pasture",
        hover_data=['Entity', 'Percent Pasture'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Percent Pasture': '% Land devoted to pasture'},
        template='plotly_dark'
    )
    return container, fig

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)








get_ipython().run_line_magic('tb', '')




