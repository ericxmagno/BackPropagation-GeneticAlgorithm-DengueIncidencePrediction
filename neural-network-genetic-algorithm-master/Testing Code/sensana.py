#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tabulate import tabulate

import pandas as pd
import math
import keras as k

import datetime
import matplotlib

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras import backend as B

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Seed for reproducibility
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)

early_stopper = EarlyStopping(monitor = 'val_loss', patience=10)

# Sensitivity Analysis

def split_dataset(fgroup, model, dataset: np.ndarray, test: np.ndarray):
    """Retrieve the dengue dataset and process the data."""
    # Set defaults.
    batch_size = 32 # not used

    train = dataset

    # Split training and test sets to x and y.
    train_x, train_y = train[:,:-1], train[:, -1]
    test_x, test_y = test[:,:-1], test[:, -1]
    

    fgroup = str(fgroup)
    model = str(model)
    # Scale test set.
    filenamex = r"C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup" + fgroup + r"\fgroup" + fgroup + "-scalerx" + model + ".save"
    scalerx = joblib.load(filenamex)
    test_xsc = scalerx.transform(test_x)

    # reshape outputs 
    test_y = test_y.reshape(-1, 1)

    # Scaled test set.
    filenamey = r"C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup" + fgroup + r"\fgroup" + fgroup + "-scalery" + model + ".save"
    scalery = joblib.load(filenamey)
    test_ysc = scalery.transform(test_y)

    return (test_x, test_xsc, test_y, test_ysc, scalerx, scalery)

def load_dataset(datasource: str, fgroup: int, week: int) -> (np.ndarray):
    """
    The function loads dataset from given file name
    :param datasource: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """
    # load the dataset
    dataframe = pd.read_csv(datasource)
    i = week
    if fgroup == 1: # model id 1
        dataset = dataframe[['cases'+str(i),'cases'+str(i+1),'cases'+str(i+2),'cases'+str(i+3),'rainfall11','rainfall13',
        'tmax11','tmax12','tmin9','tmin11','tmean11','tmean12', 'rh5' if i==4  else 'rh3','rh4','wind_speed4','wind_speed3','target']]
    elif fgroup == 2: # model id 4
        dataset = dataframe[['cases'+str(i),'cases'+str(i+1),'cases'+str(i+2),'rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]
    elif fgroup == 3: # model id 5
        dataset = dataframe[['cases'+str(i),'cases'+str(i+1),'rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]
    elif fgroup == 4: # model id 6
        dataset = dataframe[['cases'+str(i+1),'rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]
    dataset = dataset.values
    dataset = dataset.astype('float32')


    return dataset

def main():
    fgroup = 1
    print("Fgroup " + str(fgroup))
    fgrouptbl = []

    model = 3
    week = 1
    
    print("Model " + str(model))
    
    datasource = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\csvs\full_dataset_extended.csv'
    dataset = load_dataset(datasource, fgroup, week)
    testdatasource = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\csvs\test_dataset_extended.csv'
    testdataset = load_dataset(testdatasource, fgroup, week)

    # split dataset
    x_test, x_scaled, y_test, y_scaled, scalerx, scalery = split_dataset(fgroup, model, dataset, testdataset)

    # sens analysis
    values = [0.80  , 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20]
    fgroup_name = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup' + str(fgroup) + r"\fgroup " + str(fgroup)
    model_name = fgroup_name + " - model" + str(model) + ".h5"
    model = load_model(model_name)
    
    table = []
    # print(x_test[:, 15])
    para = ""

    # iterate through values
    # og_in = x values, og_out = y values
    for i in range(len(values)): 
        og_in = x_test.copy() 
        og_out = scalery.inverse_transform(model.predict(x_scaled, verbose=0))
        #  [:, x] replace x with data column to be tested (e.g. 0 = cases1, 1 = cases2, etc.)
        og_in[:, 15] = og_in[:, 15] * values[i]
        new_in = scalerx.transform(og_in)
        new_out = scalery.inverse_transform(model.predict(new_in, verbose=0))
        table.append(percent_change(og_out,new_out))
    send = []
    send.append(table)
    drawtable("para", send)
    
    # Print basic evaluative data.
    # fgroup_name = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup' + str(fgroup) + r"\fgroup " + str(fgroup)
    # model_name = fgroup_name + " - model" + str(model) + ".h5"
    # model = load_model(model_name)
    # y = scalery.inverse_transform(model.predict(x_scaled, verbose=0))
    # print(str(np.mean(y)) + " " + str(np.std(y)) + " " + str(np.median(y)))


def percent_change(old, new):
    changes = []
    for i in range(len(old)):
        if new[i] == -1:
            continue
        if old[i] != 0:
            changes.append(float(new[i] - old[i]) / old[i] * 100)
    return np.mean(changes)
   
            
def drawtable(parameter: str, table: list):

    rows = [parameter]
    # columns = ("week", "rmse", "mae", "r2s")

    df = pd.DataFrame(table, index = rows)
    df.columns = ["0.80", "0.85", "0.90", "0.95", "1.05", "1.10", "1.15", "1.20"]
    df.index.names = ['para']
    df.columns.names = ['scale']
    h = [df.index.names[0] +'/'+ df.columns.names[0]] + list(df.columns)
    print(tabulate(df, headers= h, tablefmt= 'grid'))



main()


