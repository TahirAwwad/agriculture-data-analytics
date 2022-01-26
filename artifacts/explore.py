import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



def laod_data():
    df = pd.read_csv("TA_inputoutputvalue_1990_2021_CSO.csv")
    ## Extract milk production dataset
    # drop redundunt columns
    df = df.drop('Unnamed: 0',axis = 1)

    # extract milk dataset
    df_milk = df[['Year',
    #              'UNIT',
                'All Livestock Products - Milk',
                'Taxes on Products',
                'Subsidies on Products',
                'Compensation of Employees',
                'Contract Work',
                'Entrepreneurial Income',
                'Factor Income',
                #'Fixed Capital Consumption - Farm Buildings',
                #'Fixed Capital Consumption - Machinery, Equipment, etc',
                #'Interest less FISIM',
                #'Operating Surplus',
                #'Livestock - Cattle',
                #'Livestock - Sheep',
                #'Land Rental',
                #'Intermediate Consumption - Contract Work',
                #'Intermediate Consumption - Crop Protection Products',
                #'Intermediate Consumption - Energy and Lubricants',
                #'Intermediate Consumption - Feeding Stuffs',
                #'Intermediate Consumption - Fertilisers',
                #'Intermediate Consumption - Financial Intermediation Services Indirect',
                #'Intermediate Consumption - Forage Plants',
                #'Intermediate Consumption - Maintenance and Repairs',
                #'Intermediate Consumption - Seeds',
                #'Intermediate Consumption - Services',
                #'Intermediate Consumption - Veterinary Expenses',
                #'Intermediate Consumption - Other Goods (Detergents, Small Tools, etc)',
                #'Intermediate Consumption - Other Goods and Services'
                
                ]]
    # Assign year as index
    df_milk.set_index('Year',drop=True,inplace=True)
    df_milk.index = pd.to_datetime(df_milk.index, format='%Y')
    return df_milk



def show_explore_page():
    st.header("Summary Statistics and Visual ")
    data = laod_data()
    st.line_chart(data,use_container_width=True)
    data.describe()
    fig, ax = plt.subplots()
    ax.hist(data.iloc[:,1], bins=20)
    st.pyplot(fig)




