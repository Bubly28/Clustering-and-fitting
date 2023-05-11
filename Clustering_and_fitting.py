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

def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
        
    The function does not have a plt.show() at the end so that the user 
    can savethe figure.
    """


    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    plt.title("Heatmap (Australia)")
    plt.savefig("Heatmap.png", dpi=300)
    # no plt.show() at the end
    
def scaler(df):
    """ 
    Expects a dataframe and normalises all 
    columnsto the 0-1 range. It also returns 
    dataframes with minimum and maximum for
    transforming the cluster centres
    """

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max    
    
def n_cluster(data_frame):
    '''
    Fuction to find the best number of cluster
    
    Input: 
        data_frame: DataFrame for clustering
        
    Output:
        k_rng: No. of clusters
        sse: Sum of squared error
    
    '''
    k_rng = range(1,10)
    sse=[]
    for k in k_rng:
      km = KMeans(n_clusters=k)
      km.fit_predict(data_frame)
      sse.append(km.inertia_)
    return k_rng,sse    

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

#Checking the correlation
corr = data_clus.corr()
print(corr)
map_corr(data_clus)
plt.show()

#Plotting the scatter matrix
pd.plotting.scatter_matrix(data_clus, figsize=(12, 12), s=5, alpha=0.8)
plt.savefig('Matrix.png', dpi=300)
plt.show()

df_ex = data_clus[['CO2 intensity','Population']] # extract the two columns for clustering
df_ex = df_ex.dropna() # entries with one nan are useless
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])
# reset_index() moved the old index into column index
# remove before clustering
df_ex = df_ex.drop("index", axis=1)
print(df_ex.iloc[0:15])


# normalise, store minimum and maximum
df_norm, df_min, df_max = scaler(df_ex)
print()

#No. of clusters, Sum of squared error
n,s = n_cluster(df_norm)
plt.xlabel=('no. of clusters')
plt.ylabel('sum of squared error')
plt.plot(n,s)
plt.title('No. of clusters')
plt.savefig("Elbow_method.png", dpi=300)
print(s)

#Clustering
ncluster = 6
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["CO2 intensity"], df_norm["Population"], 10, labels
,marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
#plt.xlabel('CO2 intensity')
plt.ylabel('Population')
plt.title('Cluster')
plt.savefig("Cluster.png", dpi=300)
plt.show()

#Scatter plot before fitting
plt.figure(figsize = (8,6))
plt.scatter(data["Year"], data["CO2 intensity (kg per kg of oil equivalent energy use)"])
plt.title('Scatter Plot between 1960-2020 before fitting')
plt.ylabel('CO2 intensity')
#plt.xlabel('Year')
plt.savefig("Scatter_fit.png", dpi=300)
plt.show()

#Fitting
popt, pcov = opt.curve_fit(Expo, data['Year'],data['CO2 intensity (kg per kg of oil equivalent energy use)'], p0=[1000, 0.02])
data["Pop"] = Expo(data['Year'], *popt)
sigma = np.sqrt(np.diag(pcov))
low, up = err_ranges(data["Year"],Expo,popt,sigma)
#Plotting the fitted and real data by showing confidence range
plt.figure()
plt.title("Plot After Fitting")
plt.plot(data["Year"], data['CO2 intensity (kg per kg of oil equivalent energy use)'], label="data")
plt.plot(data["Year"], data["Pop"], label="fit")
plt.fill_between(data["Year"], low, up, alpha=0.7)
plt.legend()
#plt.xlabel("year")
plt.savefig("Fitting_Graph.png", dpi=300)
plt.show()
