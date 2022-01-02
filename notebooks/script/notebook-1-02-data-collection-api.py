#!/usr/bin/env python
# coding: utf-8

# ## Fetch the proposed datasets understudy

# ### Setup
#     Install required libraries

#!pip install jsonstat.py
get_ipython().system('pip install eurostat')
import eurostat
import io
import pandas
import requests
#import jsonstat


# ### Source 1: Eurostat data
#     - Agriculture price indecies of product

# ### 2015 base year of Product Prices

# price indecies of product 2015 base year
price_idx_products_annual_code = 'apri_pi15_outa'
price_idx_products_annual_dataframe = eurostat.get_data_df(price_idx_products_annual_code, flags=False)
price_idx_products_annual_dataframe.sample(5)

# rename column
price_idx_products_annual2015_dataframe = (price_idx_products_annual_dataframe
                                           .rename(columns={price_idx_products_annual_dataframe
                                                            .columns[3]:'geotime'})
                                          )

# transform years columns to a Series
price_idx_products_annual2015_dataframe =( price_idx_products_annual2015_dataframe
     .melt(
         id_vars = ["p_adj",
                    "unit",
                    "geotime",
                    "product"],
         var_name = "year",
         value_name="priceIDX")
    )
price_idx_products_annual2015_dataframe.sample(5)


# ### ### 2010 base year of Product Prices

# price indecies of product 2010 base year
price_idx_products_annual2010_code = 'apri_pi10_outa'
price_idx_products_annual2010_dataframe = eurostat.get_data_df(price_idx_products_annual2010_code, flags=False)
price_idx_products_annual2010_dataframe = (price_idx_products_annual2010_dataframe
                                           .rename(columns={price_idx_products_annual2010_dataframe
                                                            .columns[3]:'geotime'})
                                          )

# transform years columns to a Series
price_idx_products_annual2010_dataframe =( price_idx_products_annual2010_dataframe
     .melt(
         id_vars = ["p_adj",
                    "unit",
                    "geotime",
                    "product"],
         var_name = "year",
         value_name="priceIDX")
    )
price_idx_products_annual2010_dataframe.sample(5)

# write to csv
price_idx_products_annual2010_dataframe.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_priceIDX_2000_2017_eurostat.csv')


# ## CSO
# ## Agriculture Area Used and Crop Yield
#     AQA03 - Crop Yield (1985 - 2007)
#     

# Note of where the URL comes from: https://data.cso.ie/table/AQA03

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AQA03%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

crop_yield8507_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

#for key, value in crop_yield8507_df.iloc[0:,0:3].iteritems():
#    print(key, value.unique())
#    print("\n")

crop_yield8507_df = ( crop_yield8507_df
 .pivot_table( 
     columns = "Statistic"
     ,index = ['Year',
               'Type of Crop',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
crop_yield8507_df.sample(5)

# write to csv

crop_yield8507_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_cropyield_1985_2007_CSO.csv')


#     AQA04 - Crop Yield (2008 - 2020)

# Note of where the URL comes from: https://data.cso.ie/table/AQA04

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AQA04%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

crop_yield0820_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

# pivot
crop_yield0820_df =( crop_yield0820_df
 .pivot_table( 
     columns = "Statistic"
     ,index = ['Year',
               'Type of Crop',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
 .rename(columns={"Crop Production":"Crop Yield"})                   
   )

# write to csv
crop_yield0820_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_cropyield_2008_2020_CSO.csv')


#     Join Crop Yields from 1985 to 2020 into 1 dataframe

# append crop yield from 1985 tp 2020 
crop_yield_ie_df = crop_yield8507_df.append(crop_yield0820_df)
crop_yield_ie_df.head()

# write to csv
crop_yield_ie_df.to_csv('~/Documents/GitHub/agriculture-data-analytics/assets/TA_cropyield_1985_2020_CSO.csv')


# ## Agricultural Input and Output Price Indices
#     1995 - 2010
#     2005 - 2017
#     2014 - 2020

# Note of where the URL comes from: https://data.cso.ie/table/AQA04

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AHA01%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

prc_idx_9510_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

for key, value in prc_idx_9510_df.iloc[0:,0:-1].iteritems():
    print(key, value.unique())
    print("\n")

print(prc_idx_9510_df.head())

# pivot
prc_idx_9510_df =( prc_idx_9510_df
 .pivot_table( 
     columns = "Agricultural Product"
     ,index = ['Year',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
prc_idx_9510_df.sample(5)

# write to csv
prc_idx_9510_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_inputoutputpriceIDX_1995_2010_CSO.csv')



# Note of where the URL comes from: https://data.cso.ie/table/AQA04

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AHA03%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

prc_idx_0517_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

#for key, value in prc_idx_0517_df.iloc[0:,0:-1].iteritems():
#    print(key, value.unique())
#    print("\n")

#print(prc_idx_0517_df.head())

# pivot
prc_idx_0517_df =( prc_idx_0517_df
 .pivot_table( 
     columns = "Agricultural Product"
     ,index = ['Year',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
prc_idx_0517_df.sample(10)

# write to csv
prc_idx_0517_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_inputoutputpriceIDX_2005_2017_CSO.csv')



# Note of where the URL comes from: https://data.cso.ie/table/AQA04

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AHA04%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

prc_idx_1420_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

#for key, value in prc_idx_1420_df.iloc[0:,0:-1].iteritems():
#    print(key, value.unique())
#    print("\n")

#print(prc_idx_0517_df.head())

# pivot
prc_idx_1420_df =( prc_idx_1420_df
 .pivot_table( 
     columns = "Agricultural Product"
     ,index = ['Year',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
prc_idx_1420_df.head()

# write to csv
prc_idx_1420_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_inputoutputpriceIDX_2014_2020_CSO.csv')


# ## Value at Current Prices for Output, Input and Income in Agriculture


# Note of where the URL comes from: https://data.cso.ie/table/AEA01

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AEA01%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

prc_9021df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

#for key, value in prc_9021df.iteritems():
#    print(key, value.unique())
#    print("\n")

#print(prc_idx_0517_df.head())

# pivot
prc_9021df =( prc_9021df
 .pivot_table( 
     columns = "Statistic"
     ,index = ['Year',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
prc_9021df.head()

# write to csv
prc_9021df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_inputoutputvalue_1990_2021_CSO.csv')


# ## Value at Current Prices for Subsidies on Products

# Note of where the URL comes from: https://data.cso.ie/table/AEA05

data_csv_url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22,%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22,%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false,%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22CSV%22,%22version%22:%221.0%22%7D,%22matrix%22:%22AEA05%22%7D,%22version%22:%222.0%22%7D%7D"

response_json_rpc = requests.get(data_csv_url).json()

subsidies_df = pandas.read_csv(io.StringIO(response_json_rpc["result"]), sep=",")

# pivot
subsidies_df =( subsidies_df
 .pivot_table( 
     columns = "Statistic"
     ,index = ['Year',
               'UNIT'
              ]
     ,values='VALUE'
     ,dropna = True
             )
 .reset_index()
   )
subsidies_df.head()

subsidies_df.to_csv('/Users/admin/Documents/GitHub/agriculture-data-analytics/assets/TA_subsidies_1990_2020_CSO.csv')


# # Next : EDA
