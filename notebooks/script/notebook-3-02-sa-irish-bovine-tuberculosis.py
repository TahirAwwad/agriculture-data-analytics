#!/usr/bin/env python
# coding: utf-8

# <style>
# *
# {
# 	text-align: justify;
# 	line-height: 1.5;
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# }
# 
# h2, h3, h4, h5, h6
# {
# 	font-family: "Arial", sans-serif;
# 	font-size: 12px;
# 	font-weight: bold;
# }
# h2
# {
# 	font-size: 14px;
# }
# h1
# {
# 	font-family: "Wingdings", sans-serif;
# 	font-size: 16px;
# }
# </style>

# ## Statistical Analysis of Bovine Tuberculosis

# <!--
# import data_analytics.github as github
# print(github.create_jupyter_notebook_header("markcrowe-com", "agriculture-data-analytics", "notebooks/notebook-2-02-eda-bovine-tuberculosis.ipynb", "master"))
# -->
# <table style="margin: auto;"><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/agriculture-data-analytics/master?filepath=notebooks/notebook-2-02-eda-bovine-tuberculosis.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/agriculture-data-analytics/blob/master/notebooks/notebook-2-02-eda-bovine-tuberculosis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

# ### Objective

# The objective is to provide a Statistical Analysis of the 'bovine-tuberculosis-eda-output.csv' artifact</a>.  

# ### Academic Objective

# - Analyse the variables in your dataset(s) and use appropriate inferential statistics to gain insights on possible population values (e.g., if you were working with international commerce, identify the average/variance exportation yearly/quarterly in the appropriate currency).  
# 
# - Undertake research to find similarities between some countries against Ireland and apply parametric and non-parametric inferential statistics techniques to compare them (e.g., analysis of variance, t-test, Wilcoxon test, among others).  
# 
# - Use the outcome of your analysis to deepen your research. Indicate the challenges you faced in the process.  
# 
# - You are expected to use at least 5 different inferential statistics techniques.  
# 
# - Use of descriptive statistics and appropriate visualizations are required to explain the scenario and justify the chosen models performed.  
# 
# - All your calculations and reasoning behind your models must be documented in the report and/or the appendix.  

# ### Setup

# Import required third party Python libraries, import supporting functions and sets up data source filepaths.

# Local
#!pip install -r script/requirements.txt --quiet --user
# Remote option
#!pip install -r "https://github.com/markcrowe-com/data-analytics-project-template/blob/master/notebooks/script/requirements.txt?raw=true" --quiet --user


from agriculture_data_analytics.project_manager import *
from agriculture_data_analytics.dataframe_labels import *
from data_analytics.graphs import display_caption
from matplotlib import rcParams
from pandas import DataFrame
import data_analytics.github as github
import data_analytics.exploratory_data_analysis as eda
import data_analytics.graphs as eda_graphs
import data_analytics.exploratory_data_analysis_reports as eda_reports
import numpy
import os
import pandas
import matplotlib.pyplot as pyplot
import scipy
import seaborn


artifact_manager = ProjectArtifactManager()
asset_manager = ProjectAssetManager()


artifact_manager: ProjectArtifactManager = ProjectArtifactManager()
asset_manager: ProjectAssetManager = ProjectAssetManager()
artifact_manager.is_remote = asset_manager.is_remote = True
github.display_jupyter_notebook_data_sources(
    [artifact_manager.get_county_bovine_tuberculosis_eda_filepath()])
artifact_manager.is_remote = asset_manager.is_remote = False


# #### Create Data Frames

filepath: str = artifact_manager.get_county_bovine_tuberculosis_eda_filepath()
county_bovine_tuberculosis_dataframe: DataFrame = pandas.read_csv(filepath)


county_bovine_tuberculosis_dataframe.head()


# Statistics of the columns

county_bovine_tuberculosis_dataframe.describe()


# ### Correlation

# Year is our interval. We drop the Year for correlation.

# Correlation size | Interpretation
# -|-
# &plusmn; 1.00 to 1.00 | Perfect correlation
# &plusmn; 0.90 to 0.99 | Very high correlation
# &plusmn; 0.70 to 0.90 | High correlation
# &plusmn; 0.50 to 0.70 | Moderate correlation
# &plusmn; 0.30 to 0.50 | Low correlation
# &plusmn; 0.00 to 0.30 | Negligible correlation
# 
# <p class="Caption">Correlation Interpretation Table</p>
# 
# This table suggests the interpretation of correlation size at different absolute values. These cut-offs are arbitrary and should be used judiciously while interpreting the dataset.

correlation_dataframe = county_bovine_tuberculosis_dataframe.drop(columns=[YEAR])


eda_graphs.display_correlation_matrix_pyramid_heatmap(correlation_dataframe.corr());


# <p class="Caption">Bovine TB Correlation Matrix Heat Map Pyramid</p>

# #### Correlations suggesting investigation
# Consider correlation Threshold &GreaterEqual; 0.85.  
# 
# Feature one  | Feature two |  Correlation size
# :-|:-|-
# Animal Count | Tests on Animals | 0.99
# Restricted Herds at end of Year | Reactors to date | 0.86
# Restricted Herds at end of Year | Restricted Herds at start of Year | 0.97
# Restricted Herds at start of Year | Reactors to date | 0.85
# Herds Tested | Herds Count | 1

# ### Central Tendency

rcParams['figure.figsize'] = 13, 6  # Set the graph size


axis = seaborn.barplot(data=county_bovine_tuberculosis_dataframe,
                       color="g",
                       x=YEAR,
                       y=HERD_INCIDENCE_RATE)
axis.set_xticklabels(labels=axis.get_xticklabels(), rotation=90)
pyplot.show()
display_caption(f"Bar plot of {HERD_INCIDENCE_RATE} against Year")

axis = seaborn.barplot(data=county_bovine_tuberculosis_dataframe,
                       color="b",
                       x=YEAR,
                       y=REACTORS_PER_1000_TESTS_APT)
axis.set_xticklabels(labels=axis.get_xticklabels(), rotation=90)
pyplot.show()
display_caption(f"Bar plot of {REACTORS_PER_1000_TESTS_APT} against Year")


# ### Measures of Variability

# #### Inter-quartile Range

box = county_bovine_tuberculosis_dataframe.boxplot([HERD_INCIDENCE_RATE],
                                                   showmeans=True,
                                                   whis=99)
pyplot.title(f"Average {HERD_INCIDENCE_RATE}")
pyplot.ylabel("Years")
pyplot.show()
display_caption(f"{HERD_INCIDENCE_RATE} Box plot")


box = county_bovine_tuberculosis_dataframe.boxplot([HERD_INCIDENCE_RATE],
                                                   showmeans=True,
                                                   whis=99)
pyplot.title(f"Average {REACTORS_PER_1000_TESTS_APT}")
pyplot.ylabel("Years")
pyplot.show()
display_caption(f"{REACTORS_PER_1000_TESTS_APT} Box plot")


# ### Standard Deviation

county_bovine_tuberculosis_dataframe[[
    HERD_INCIDENCE_RATE, HERDS_TESTED, REACTORS_PER_1000_TESTS_APT
]].std()


pyplot.errorbar(
    county_bovine_tuberculosis_dataframe[YEAR],
    county_bovine_tuberculosis_dataframe[HERD_INCIDENCE_RATE],
    county_bovine_tuberculosis_dataframe[HERD_INCIDENCE_RATE].std(),
    linestyle='None',
    marker='^',
    capsize=3)
title = f"Standard Deviation of {HERD_INCIDENCE_RATE}"
pyplot.title(title)
pyplot.show()
display_caption(title)


pyplot.errorbar(
    county_bovine_tuberculosis_dataframe[YEAR],
    county_bovine_tuberculosis_dataframe[REACTORS_PER_1000_TESTS_APT],
    county_bovine_tuberculosis_dataframe[REACTORS_PER_1000_TESTS_APT].std(),
    linestyle='None',
    marker='^',
    capsize=3)
title = f"Standard Deviation of {REACTORS_PER_1000_TESTS_APT}"
pyplot.title(title)
pyplot.show()
display_caption(title)


# ### Distribution

_ = seaborn.histplot(
    data=county_bovine_tuberculosis_dataframe[HERD_INCIDENCE_RATE], kde=True)
pyplot.show()
display_caption(f"{HERD_INCIDENCE_RATE} Distribution")


# ### Normal Distribution

data = county_bovine_tuberculosis_dataframe[HERD_INCIDENCE_RATE]
axis = seaborn.histplot(data=data,
                        kde=True,
                        line_kws={
                            'linestyle': ':',
                            'linewidth': 3
                        })

median = data.median()
x = numpy.arange(data.min(), data.max(), 0.1)
y = scipy.stats.norm.pdf(x, scale=data.std(), loc=median)
y = [probability * median**2 for probability in y]  #scaling
axis.plot(x, y, c='r')
pyplot.legend(labels=["Kernel Density", "Probability Density"])
pyplot.show()
display_caption(f"{HERD_INCIDENCE_RATE} Normal Distribution")


# ### After removing outliers

eda_reports.report_outliers_columns(county_bovine_tuberculosis_dataframe)


expected_herd_incidence_dataframe = eda.get_expected_range_dataframe(
    county_bovine_tuberculosis_dataframe,
    HERD_INCIDENCE_RATE)[[YEAR, HERD_INCIDENCE_RATE]]
print(expected_herd_incidence_dataframe.shape[0], "expected values found")
expected_herd_incidence_dataframe[HERD_INCIDENCE_RATE].describe()


expected_reactors_per_1000_dataframe = eda.get_expected_range_dataframe(
    county_bovine_tuberculosis_dataframe,
    REACTORS_PER_1000_TESTS_APT)[[YEAR, REACTORS_PER_1000_TESTS_APT]]
print(expected_reactors_per_1000_dataframe.shape[0], "expected values found")
expected_reactors_per_1000_dataframe[REACTORS_PER_1000_TESTS_APT].describe()


axis = seaborn.barplot(data=expected_herd_incidence_dataframe,
                       color="g",
                       x=YEAR,
                       y=HERD_INCIDENCE_RATE)
axis.set_xticklabels(labels=axis.get_xticklabels(), rotation=90)
pyplot.show()
display_caption(f"Bar plot of {HERD_INCIDENCE_RATE} against Year")

axis = seaborn.barplot(data=expected_reactors_per_1000_dataframe,
                       color="b",
                       x=YEAR,
                       y=REACTORS_PER_1000_TESTS_APT)
axis.set_xticklabels(labels=axis.get_xticklabels(), rotation=90)
pyplot.show()
display_caption(f"Bar plot of {REACTORS_PER_1000_TESTS_APT} against Year")


# ### Distribution

_ = seaborn.histplot(
    data=expected_herd_incidence_dataframe[HERD_INCIDENCE_RATE], kde=True)
pyplot.show()
display_caption(f"{HERD_INCIDENCE_RATE} Distribution")


data = expected_herd_incidence_dataframe[HERD_INCIDENCE_RATE]
axis = seaborn.histplot(data=data,
                        kde=True,
                        line_kws={
                            'linestyle': ':',
                            'linewidth': 3
                        })

median = data.median()
x = numpy.arange(data.min(), data.max(), 0.1)
y = scipy.stats.norm.pdf(x, scale=data.std(), loc=median)
y = [probability * median**2 for probability in y]  #scaling
axis.plot(x, y, c='r')
pyplot.legend(labels=["Kernel Density", "Probability Density"])
pyplot.show()
display_caption(f"{HERD_INCIDENCE_RATE} Normal Distribution")

