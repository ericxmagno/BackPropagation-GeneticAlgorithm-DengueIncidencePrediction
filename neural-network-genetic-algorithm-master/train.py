"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import backend as K

import logging
import math
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

# Seed for reproducibility
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)

# Helper: Early stopping.
early_stopper = EarlyStopping(monitor = 'val_loss', patience=10)

def split_dataset(dataset: np.ndarray, test: np.ndarray):
    """Retrieve the dengue dataset and process the data."""
    # Set defaults.
    batch_size = 32
    train = dataset
    
    # Split training and test sets to x and y.
    train_x, train_y = train[:,:-1], train[:, -1]
    test_x, test_y = test[:,:-1], test[:, -1]

    # Normalize training dataset and use this scaler on testing set.
    scalerx = MinMaxScaler(feature_range=(0, 1))
    train_x = scalerx.fit_transform(train_x)
    test_x = scalerx.transform(test_x)

    # reshape output
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    scalery = MinMaxScaler(feature_range=(0, 1))
    train_y = scalery.fit_transform(train_y)
    test_y = scalery.transform(test_y)

    # assign input shape to row 1 of training set
    input_shape = (train_x.shape[1], )

    return (batch_size, input_shape, train_x, test_x, train_y, test_y, scalerx, scalery)

def load_dataset(datasource: str, fgroup: int) -> (np.ndarray):
    """
    The function loads dataset from given file name
    :param datasource: file name of data source
    :return: tupdataframed the used MinMaxScaler
    """
    # load the dataset
    dataframe = pd.read_csv(datasource)
    
    if fgroup == 1: # model id 1
        dataset = dataframe[['cases1','cases2','cases3','cases4','rainfall11','rainfall13','tmax11','tmax12',
        'tmin9','tmin11','tmean11','tmean12','rh3','rh4','wind_speed4','wind_speed3','target']]
    elif fgroup == 2: # model id 4
        dataset = dataframe[['cases1','cases2','cases3','rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]
    elif fgroup == 3: # model id 5
        dataset = dataframe[['cases1','cases2','rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]
    elif fgroup == 4: # model id 6
        dataset = dataframe[['cases1','rainfall11','tmax12','tmin9','tmean11','rh4','wind_speed4','target']]

    dataset = dataset.values
    dataset = dataset.astype('float32')

    return dataset

def compile_model(network, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=(input_shape)))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

    return model

def train_and_score(network, datasource, fgroup):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    # reset Keras graphs
    K.clear_session()

    # load dataset  
    dataset = load_dataset(datasource, fgroup)
    testdatasource = 'csvs/test_dataset_extended.csv'
    testdataset = load_dataset(testdatasource, fgroup)

    # split dataset
    batch_size, input_shape, x_train, x_test, y_train, y_test, scalerx, scalery = split_dataset(dataset, testdataset)

    # k-fold cross validation (5)
    tsplit = TimeSeriesSplit(n_splits=5)
    score = 0.
    for train, valid in tsplit.split(x_train):
        model = compile_model(network, input_shape)
        model.fit(x_train[train], y_train[train],
                batch_size=batch_size,
                epochs=10000,  # using early stopping, so no real limit
                verbose=0,
                validation_data=(x_train[valid], y_train[valid]),
                callbacks=[early_stopper])
        eval = model.evaluate(x_test, y_test, verbose=0)
        score += math.sqrt(eval[1])
    
    score /= 5.0
    return score    # 1 is mse. 0 is loss.

def savethis(network, filenum, datasource, fgroup):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    K.clear_session()
    # load dataset  
    dataset = load_dataset(datasource, fgroup)
    testdatasource = 'csvs/test_dataset_extended.csv'
    testdataset = load_dataset(testdatasource, fgroup)

    # split dataset
    batch_size, input_shape, x_train, x_test, y_train, y_test, scalerx, scalery = split_dataset(dataset, testdataset)

    scalerx_filename = "fgroup" + str(fgroup) + "-scalerx" + str(filenum) + ".save"
    scalery_filename = "fgroup" + str(fgroup) + "-scalery" + str(filenum) + ".save"
    joblib.dump(scalerx, scalerx_filename)
    joblib.dump(scalery, scalery_filename)

    tsplit = TimeSeriesSplit(n_splits=5)
    score = 0.
    scoretr = 0.
    for train, valid in tsplit.split(x_train):
        model = compile_model(network, input_shape)
        model.fit(x_train[train], y_train[train],
                batch_size=batch_size,
                epochs=10000,  # using early stopping, so no real limit
                verbose=0,
                validation_data=(x_train[valid], y_train[valid]),
                callbacks=[early_stopper])
        treval = model.evaluate(x_train, y_train, verbose=0)
        eval = model.evaluate(x_test, y_test, verbose=0)
        scoretr += math.sqrt(treval[1])
        score += math.sqrt(eval[1])

    filename = "fgroup " + str(fgroup) + " - model" + str(filenum) + ".h5"
    logging.info("Network Training RMSE: %.6f" % (scoretr/5.0))
    print(scoretr/5.0)
    score /= 5.0
    print(score)
    model.save(filename)
    return score
