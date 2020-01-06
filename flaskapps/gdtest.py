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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

early_stopper = EarlyStopping(monitor = 'val_loss', patience=10)

minval = 100

def split_dataset(dataset: np.ndarray, test: np.ndarray):
    """Retrieve the dengue dataset and process the data."""
    # Set defaults.
    batch_size = 16
    # train_size = int(len(dataset) * 0.80) # %split of training and validation set

    # Split data into training and validation sets, then to x and y.
    # train, valid  = dataset[0:train_size], dataset[train_size:]
    train = dataset

    # Split training and test sets to x and y.
    train_x, train_y = train[:,:-1], train[:, -1]
    # valid_x, valid_y = valid[:,:-1], valid[:, -1]
    test_x, test_y = test[:,:-1], test[:, -1]
    
    # Normalize training dataset and use this scaler on testing set.
    scalerx = MinMaxScaler(feature_range=(0, 1))
    train_x = scalerx.fit_transform(train_x)
    # valid = scaler.transform(valid)
    test_x = scalerx.transform(test_x)

    # reshape outputs 
    train_y = train_y.reshape(-1, 1)
    # valid_y = valid_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Normalize training dataset and use this scaler on testing set.
    scalery = MinMaxScaler(feature_range=(0, 1))
    train_y = scalery.fit_transform(train_y)
    # valid = scaler.transform(valid)
    test_y = scalery.transform(test_y)

    return (train_x, test_x, train_y, test_y, scalery)

def load_dataset(datasource: str) -> (np.ndarray):
    """
    The function loads dataset from given file name
    :param datasource: file name of data source
    :return: tuple of dataset and the used MinMaxScaler
    """
    # load the dataset
    dataframe = pd.read_csv(datasource)

    dataset = dataframe[['rainfall11','tmax12','tmin9','tmean11', 'rh4','wind_speed4','cases1','cases2','cases3','target']]
    
    dataset = dataset.values
    dataset = dataset.astype('float32')


    return dataset

def main():
    datasource = r'C:\Users\Eric\Documents\VSCode\flaskapps\csvs\full_dataset_extended.csv'
    dataset = load_dataset(datasource)
    testdatasource = r'C:\Users\Eric\Documents\VSCode\flaskapps\csvs\test_dataset_extended.csv'
    testdataset = load_dataset(testdatasource)

    # split dataset
    x_train, x_test, y_train, y_test, scaler = split_dataset(dataset, testdataset)

    week = []
    model_name = "adam-1.h5"
    model = load_model(model_name)
    print(scaler.inverse_transform(model.predict(x_test,verbose=0)).reshape(1,-1))
    print(scaler.inverse_transform(y_test).reshape(1,-1))

    B.clear_session()

main()