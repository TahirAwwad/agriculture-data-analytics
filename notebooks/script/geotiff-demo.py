#!/usr/bin/env python
# coding: utf-8

# ## GeoTiff Demo

# ### Importing Geotiff 

# !pip install rasterio
import rasterio

import warnings
warnings.filterwarnings('ignore')


from rasterio.plot import show


fp = r'../maps/CropAllocationFoodFeedFuel_Geotiff/FoodProdAreaFrac.tif'
img1 = rasterio.open(fp)
fp = r'../maps/CropAllocationFoodFeedFuel_Geotiff/DeliveredkcalFraction.tif'
img2 = rasterio.open(fp)
#fp = r'u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif'
#img3 = rasterio.open(fp)


print("Image has", img1.count, "band.")
img1.bounds


show(img1)


from matplotlib import pyplot





pyplot.imshow(img1.read(1), cmap='Greys_r')


from rasterio.plot import show_hist
show_hist(img1, bins=50, lw=0.0, stacked=False, alpha=0.3,
    histtype='stepfilled', title="Histogram")


#pip install ipywidgets


import ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual 


def select_image(Dataset):
#    show_hist(Dataset, bins=50, lw=0.0, stacked=False, alpha=0.3,
#        histtype='stepfilled', title="Histogram");
    show(Dataset);
    print(Dataset.name)
    return


interact(select_image, Dataset=[("Food Production Area Fraction", img1) ,("Delivered kCal Fraction",img2)]);




