#!/usr/bin/env python
# coding: utf-8

# ## Fetch the proposed datasets understudy

# ### Setup
#     Install required libraries

import eurostat
import io
import pandas
import requests
import jsonstat


# ### Eurostat data
#     - Agriculture price indecies of product

# price indecies of product 2015 base year
price_idx_products_annual_code = 'apri_pi15_outa'
price_idx_products_annual_dataframe = eurostat.get_data_df(price_idx_products_annual_code, flags=False)
price_idx_products_annual_dataframe.sample(5)


# price indecies of product 2010 base year
price_idx_products_annual2010_code = 'apri_pi10_outa'
price_idx_products_annual2010_dataframe = eurostat.get_data_df(price_idx_products_annual2010_code, flags=False)
price_idx_products_annual2010_dataframe = (price_idx_products_annual2010_dataframe
                                           .rename(columns={price_idx_products_annual2010_dataframe
                                                            .columns[3]:'geotime'})
                                          )
price_idx_products_annual2010_dataframe.sample(5)


price_idx_products_annual2010_dataframe["geotime"].value_counts()


price_idx_products_annual2010_dataframe.query("geotime=='IE'")




