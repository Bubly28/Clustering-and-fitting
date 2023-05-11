# -*- coding: utf-8 -*-
"""
Created on Thu May 11 20:46:23 2023

@author: hp
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline
from sklearn import cluster
from sklearn.metrics import silhouette_score


def read_data(filename):
    '''
    Function for reading the data 
    '''
    df = pd.read_excel(filename, skiprows = 3)
    return df

def filter_data(df, col, value, yr, ind):
    '''
    Function for filtering the data
    
    Input:
        df: Data
        col: Column Name
        value: Value in the Country Name
        yr: Years
        ind: Indicator

    '''
    df3 = df.groupby(col, group_keys = True)
    df3 = df3.get_group(value)
    df3 = df3.reset_index()
    df3.set_index('Indicator Name', inplace=True)
    df3 = df3.loc[:, yr]
    df3 = df3.transpose()
    df3 = df3.loc[:,ind ]
    df3 = df3.dropna(axis = 1)
    return df3

#Caling the read function
data1 =  read_data("climate_change.xlsx")
warnings.filterwarnings("ignore")
start = 1960
end = 2015
year = [str(i) for i in range(start, end+1)]
#Indicator for fitting
Indicator = ['CO2 intensity (kg per kg of oil equivalent energy use)','Population, total']
#Indicator for clustering
Indicator1 = ['CO2 intensity (kg per kg of oil equivalent energy use)','Population, total', 'CO2 emissions from liquid fuel consumption (kt)', 'Energy use (kg of oil equivalent per capita)', 'Electricity production from renewable sources, excluding hydroelectric (% of total)']
#data for fitting
data = filter_data(data1,'Country Name', 'Australia', year , Indicator)
data = data.rename_axis('Year').reset_index()
data['Year'] = data['Year'].astype('int')
data.dtypes
#Data for clustering
data_clus = filter_data(data1,'Country Name', 'Australia', year , Indicator1)
data_clus = data_clus.rename(columns={
    'CO2 intensity (kg per kg of oil equivalent energy use)': 'CO2 intensity',
    'Population, total':'Population',
    'CO2 emissions from liquid fuel consumption (kt)':'CO2 emission',
    'Energy use (kg of oil equivalent per capita)':'Energy use',
    'Electricity production from renewable sources, excluding hydroelectric (% of total)':'Electricity production'})
print(data_clus.head())
print(data_clus.describe())
data_clus.corr()
