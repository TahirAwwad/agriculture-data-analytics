#!/usr/bin/env python
# coding: utf-8

from data_analytics.graphs import display_caption
import data_analytics.github as github
import data_analytics.exploratory_data_analysis_reports as eda_reports
import pandas
import matplotlib.pyplot as pyplot
import seaborn


# Perennial Ryegrass (95% of Irish Grassland)
# 
# 
# - It may not survive very cold winters (minus 6oC or less; Cool et al., 2004).
# - Its optimum growth temperature is 18-20oC (Mitchell, 1956)
# - L. perenne is most productive in spring and autumn (Waller and Sale, 2001)
# 
# 
# Rainfall - https://data.cso.ie/table/MTM01 : Ideal crops wet or dry conditions, as well of production to feed livestock.  
# Temperature - https://data.cso.ie/table/MTM02 : Viability of specific crops due to humidity (E.g. 20-25 Celsius max temp.)  
# Sunshine - https://data.cso.ie/table/MTM02 : Minimum levels of sunshine for crops to synthesise and grow.  
# Fertiliser - https://data.cso.ie/table/AJM05 :  
# Area Farmed in June - https://data.cso.ie/table/AQA06 : Percentage of grassland compared to farmland - other crops. Could also look into reduction of farmland and globalisation growth. Globalization. Opportunity to produce locally shortening supply chains.   
# CLC Land Cover Change - https://data.cso.ie/table/GCA02 : Changes between grass and cropland. 

area_farmed_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-area-farmed-june-aqa06.csv')
fert_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-fertilizers-ajm05.csv')
land_cover_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-clc-land-cover-gca02.csv')


rainfall_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-rainfall-mtm01.csv')
sunshine_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-sunshine-mtm02-filtered.csv')
temperature_dataframe = pandas.read_csv('./../assets/cso-2022-01Jan-10-temperature-mtm02-filtered.csv')


eda_reports.print_dataframe_analysis_report(rainfall_dataframe, "rainfall_dataframe")
eda_reports.print_dataframe_analysis_report(sunshine_dataframe, "sunshine_dataframe")
eda_reports.print_dataframe_analysis_report(temperature_dataframe, "temperature_dataframe")


# 
# Not all stations have presented solid results in terms of continuous data. Some have operated for certing periods but not the entire timeline seen on the dataset. Also, for a fair comparisson, the goal is to collect the maximum number of Meteorological Stations with published data in the last five years.
# 
# Criteria:
# 1. To have published the data in all: Rainfall, Sunshine and Temperature datasets.
# 2. For the data to be complete with no null values in the last five years.
# 
# 
# The only Stations that meet the criteria are listed in the dictionary below:

stations_to_keep = ['Casement' , 'Cork airport', 'Dublin airport', 'Shannon airport']


area_farmed_dataframe.head()


area_farmed_dataframe = area_farmed_dataframe.rename(columns={'Type of Land Use':'Land_Type'})


land_cover_dataframe


fert_dataframe


fert_dataframe = fert_dataframe.rename(columns={'Type of Fertilizer':'Fertilizer_Type'})


# # Functions

# The two functions below will allow month and year values to be separate. Improving analysis around seasonality and allowing for plots to be creating aknoleging the fluctuation throughout the year.

#Extract the year from Month column
def create_year(text):
  return int(text[:4])


#Extract the month from Month column
def create_month(text):
  return int(text[-2:])


# 
# 
# The three functions below aim to create a comparative visualization between the last five years, allowing for the observations of trends in:
# 
# 
# *   Rainfall
# *   Sunshine
# *   Lowest Temperature
# *   Highest Temperature
# 
# 

def plot_rainfall(rainfall_dataframe, initial_year, location):
    df_last = rainfall_dataframe.loc[rainfall_dataframe.Year >= initial_year]  #Filter by Year
    df_last = df_last.sort_values(by=["Year","month"]) #Sort by Year > Month
    df_last = df_last.loc[(df_last.Met_Station == location)] #Filter by location (Meteorological Station)
    df_last.reset_index(inplace=True,drop=True) #Reset Indexes

    caption:str = f"Total rainfall for {location}"
    pivot = df_last.pivot("month","Year" ,"VALUE") #Create Pivot Table Month x Year x Value
    seaborn.set(rc = {'figure.figsize':(25,12)}) #Set figure size
    seaborn.lineplot(data=pivot).set_title(caption) #Plot
    pyplot.show()
    display_caption(caption)


def plot_sunshine(df, initial_year, location):
    df_last = df.loc[df.Year >= initial_year]  #Filter by Year
    df_last = df_last.sort_values(by=["Year","month"]) #Sort by Year > Month
    df_last = df_last.loc[(df_last.Met_Station == location)] #Filter by location (Meteorological Station)
    df_last.reset_index(inplace=True,drop=True) #Reset Indexes

    caption:str = f"Total sunshine for {location}"

    pivot = df_last.pivot("month","Year" ,"VALUE") #Create Pivot Table Month x Year x Value
    seaborn.set(rc = {'figure.figsize':(25,12)}) #Set figure size
    seaborn.lineplot(data=pivot).set_title(caption) #Plot
    pyplot.show()
    display_caption(caption)


def plot_temp(df, initial_year, temp_type,location):
    temp_last = df.loc[df.Year >= initial_year] #Filter by Year #CHANGED varaibel to local#
    temp_last = temp_last.sort_values(by=["Year","month"]) #Sort by Year > Month

    #Filter by temperature type
    temp_last_final = None
    if temp_type == "Min":
        temp_last_final = temp_last.loc[(temp_last.Met_Station == location) & (temp_last.Statistic == "Lowest Temperature")]
    elif temp_type == "Max":
        temp_last_final = temp_last.loc[(temp_last.Met_Station == location) & (temp_last.Statistic == "Highest Temperature")]

    temp_last_final.reset_index(inplace=True, drop=True) #Reset indexes

    caption:str = f"Location: {location} - Temperature Type: {temp_type}"
    pivot = temp_last_final.pivot("month","Year" ,"VALUE") #Create Pivot Table Month x Year x Value
    seaborn.set(rc = {'figure.figsize':(25,12)}) #Set figure size
    seaborn.lineplot(data=pivot).set_title(caption)
    pyplot.show()
    display_caption(caption)


# Creating an average of our last 5 years should provide relevant indicators of when to plant, grow and harvest. The last five years have been chosen due to rising global temperature, considering old data could corrupt the data with unnacurate representations of our current seasons.

def get_average_rain_sun(dataframe, met_station, min_year):
    temporary = dataframe.loc[(dataframe.Met_Station == met_station) & (dataframe.Year >= min_year)]
    temporary.reset_index(inplace=True, drop=True)
    return list(temporary.groupby(by="month").mean()["VALUE"])

def compare_rain_sun(rainfall_dataframe, sunshine_dataframe, met_station, min_year):
    temp_df = pandas.DataFrame()
    temp_df["month"] = range(1,13)
    temp_df["rain"] = get_average_rain_sun(rainfall_dataframe, met_station, min_year)
    temp_df["sun"] =  get_average_rain_sun(sunshine_dataframe, met_station, min_year)

    caption:str = f"Rainfall x Sunshine in {met_station}."

    seaborn.lineplot(x='month', y='value', hue='variable', data=pandas.melt(temp_df, ['month'])).set_title(caption)
    pyplot.show()
    display_caption(caption)


def get_average_temperature(dataframe, met_station, min_year):
    temp_max = dataframe.loc[(dataframe.Met_Station == met_station) & 
                             (dataframe.Year >= min_year) & (dataframe.Statistic == "Highest Temperature")]
    temp_min = dataframe.loc[(dataframe.Met_Station == met_station) & 
                             (dataframe.Year >= min_year) & (dataframe.Statistic == "Lowest Temperature")]
    temp_min.reset_index(inplace=True, drop=True)
    temp_max.reset_index(inplace=True, drop=True)
    return list(temp_min.groupby(by="month").mean()["VALUE"]), list(temp_max.groupby(by="month").mean()["VALUE"])

def compare_temp(temp, met_station, min_year):
    temp_df = pandas.DataFrame()
    temp_df["month"] = range(1,13)
    temp_df["temp_min"], temp_df["temp_max"] = get_average_temperature(temp, met_station, min_year)
    
    caption:str = f"High v Low Temperatures from {met_station}."
    seaborn.lineplot(x='month', y='value', hue='variable',
             data=pandas.melt(temp_df, ['month'])).set_title(caption)
    pyplot.show()
    display_caption(caption)


# # Rain
# ---

rainfall_dataframe = rainfall_dataframe.rename(columns={'Meteorological Weather Station':'Met_Station'})
rainfall_dataframe.head()


rainfall_dataframe["Year"] = rainfall_dataframe.Month.apply(create_year)


rainfall_dataframe["month"] = rainfall_dataframe.Month.apply(create_month)


rainfall_dataframe.drop(["Month"],axis=1,inplace=True)


rainfall_dataframe = rainfall_dataframe.loc[rainfall_dataframe.Met_Station.isin(stations_to_keep)]
rainfall_dataframe = rainfall_dataframe.loc[rainfall_dataframe.Statistic == 'Total Rainfall']
rainfall_dataframe.reset_index(inplace=True,drop=True)
rainfall_dataframe.head()


for station in stations_to_keep:
    plot_rainfall(rainfall_dataframe, 2017,station)


# # Sunshine

sunshine_dataframe = sunshine_dataframe.rename(columns={'Meteorological Weather Station':'Met_Station'})


sunshine_dataframe = sunshine_dataframe.loc[sunshine_dataframe.Met_Station.isin(stations_to_keep)]
sunshine_dataframe = sunshine_dataframe.loc[sunshine_dataframe.Statistic == 'Total Sunshine Hours']
sunshine_dataframe.reset_index(inplace=True, drop=True)
sunshine_dataframe.head()


sunshine_dataframe["Year"]  = sunshine_dataframe.Month.apply(create_year)
sunshine_dataframe["month"] = sunshine_dataframe.Month.apply(create_month)
sunshine_dataframe.drop(["Month"],axis=1, inplace=True)
sunshine_dataframe.head()


sunshine_dataframe.loc[(sunshine_dataframe.Met_Station == "Claremorris") & (sunshine_dataframe.Year > 2017)]


for station in stations_to_keep:
    plot_sunshine(sunshine_dataframe, 2017,station)


# # Temperature

temperature_dataframe = temperature_dataframe.rename(columns={'Meteorological Weather Station':'Met_Station'})


temperature_dataframe = temperature_dataframe.loc[temperature_dataframe.Met_Station.isin(stations_to_keep)]
temperature_dataframe.reset_index(inplace=True,drop=True)
temperature_dataframe.head()


temperature_dataframe["Year"] = temperature_dataframe.Month.apply(create_year)
temperature_dataframe["month"] = temperature_dataframe.Month.apply(create_month)
temperature_dataframe.drop(["Month"],axis=1,inplace=True)
temperature_dataframe.head()


for station in stations_to_keep:
    plot_temp(temperature_dataframe, 2017, "Min", station)


# # Rainfall x Sunshine Plots

for station in stations_to_keep:
    compare_rain_sun(rainfall_dataframe, sunshine_dataframe, station,2017)


# # Highest x Lowest Temperature Plots

# Comparative visualization using previous function.

for station in stations_to_keep:
    compare_temp(temperature_dataframe,station,2017)

