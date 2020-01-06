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

def split_dataset(fgroup, model, dataset: np.ndarray, test: np.ndarray):
    """Retrieve the dengue dataset and process the data."""
    # Set defaults.
    batch_size = 32
    # train_size = int(len(dataset) * 0.80) # %split of training and validation set

    # Split data into training and validation sets, then to x and y.
    # train, valid  = dataset[0:train_size], dataset[train_size:]
    train = dataset

    # Split training and test sets to x and y.
    train_x, train_y = train[:,:-1], train[:, -1]
    # valid_x, valid_y = valid[:,:-1], valid[:, -1]
    test_x, test_y = test[:,:-1], test[:, -1]
    

    fgroup = str(fgroup)
    model = str(model)
    # Normalize training dataset and use this scaler on testing set.
    filenamex = r"C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup" + fgroup + r"\fgroup" + fgroup + "-scalerx" + model + ".save"
    scalerx = joblib.load(filenamex)
    train_x = scalerx.transform(train_x)    
    # valid = scaler.transform(valid)
    test_x = scalerx.transform(test_x)

    # reshape outputs 
    train_y = train_y.reshape(-1, 1)
    # valid_y = valid_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Normalize training dataset and use this scaler on testing set.
    filenamey = r"C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup" + fgroup + r"\fgroup" + fgroup + "-scalery" + model + ".save"
    scalery = joblib.load(filenamey)
    train_y = scalery.transform(train_y)
    # valid = scaler.transform(valid)
    test_y = scalery.transform(test_y)

    return (train_x, test_x, train_y, test_y, scalery)

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
    # maintbl = []
    fgroup = 1
    print("Fgroup " + str(fgroup))
    fgrouptbl = []
    table = []
    for i in range(0,5):
        
        
        print("Model " + str(i))
        

        week = 1
            
        weektbl = []
        parentpath = ""

        datasource = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\csvs\full_dataset_extended.csv'
        dataset = load_dataset(datasource, fgroup, week)
        testdatasource = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\csvs\test_dataset_extended.csv'
        testdataset = load_dataset(testdatasource, fgroup, week)

        # split dataset
        x_train, x_test, y_train, y_test, scaler = split_dataset(fgroup, i, dataset, testdataset)

        
        fgroup_name = r'C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\fgroup' + str(fgroup) + r"\fgroup " + str(fgroup)
        model_name = fgroup_name + " - model" + str(i) + ".h5"
        model = load_model(parentpath + model_name)

        ytr_pred = model.predict(x_train, verbose=0)
        weektbl.append(math.sqrt(mean_squared_error(y_train,ytr_pred)))

        y_pred = model.predict(x_test, verbose=0)
        
        rmse = str(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = str(mean_absolute_error(y_test, y_pred))
        r2s = str(r2_score(y_test, y_pred))

        print("RMSE: " + rmse + "- MAE: " + mae +  " - R2 Score: " + r2s)
        y_train = model.predict(x_train, verbose = 0)
        y = scaler.inverse_transform(y_train)
        invy_train = scaler.inverse_transform(y_train)

        yt = scaler.inverse_transform(y_test)
        invy_pred = scaler.inverse_transform(y_pred)
        invrmse = str(np.sqrt(mean_squared_error(yt, invy_pred)))
        invmae = str(mean_absolute_error(yt, invy_pred))
        invr2s = str(r2_score(yt, invy_pred))
        # print(invy_pred)
        drawgraph(model_name, invy_train, invy_pred, y, yt)

        # weektbl.append(rmse)
        # # weektbl.append(mae)
        # weektbl.append(r2s)
        # weektbl.append(invrmse)
        # weektbl.append(invmae)
        # weektbl.append(invr2s)

        # table.append(weektbl)
        B.clear_session()

        # fgrouptbl.append(table)
    # drawtable(i, table)
    # fgrouptbl = np.array(fgrouptbl).astype(np.float)
    # temp = np.average(fgrouptbl, axis=0)
    # print(temp)
    # maintbl.append(temp)
    # drawbargraph(maintbl)


   
        
def drawgraph(model_name: str, invy_train: np.ndarray, invy_pred: np.ndarray, y: np.ndarray, yt: np.ndarray):
    dengue = pd.read_csv(r"C:\Users\Eric\Documents\VSCode\neural-network-genetic-algorithm-master\csvs\dengueweekly.csv")
    dates = dengue[dengue['YEAR']<2017]
    dates = dates[8:-43].reset_index(drop=True)
    dates = dates.drop(['CASES'],axis=1)
    dates['YEAR'] = dates['YEAR'].astype(str)
    dates['WEEK'] = dates['WEEK'].astype(str)
    dates['date'] = dates['YEAR'] + ' ' + dates['WEEK'].map(str)
    datesTr = []
    for i in dates['date']:
        datesTr.append(datetime.datetime.strptime(i + '-0', "%Y %W-%w"))
    dates = dengue[dengue['YEAR']>2015]
    dates = dates[9:-17].reset_index(drop=True)
    dates['YEAR'] = dates['YEAR'].astype(str)
    dates['WEEK'] = dates['WEEK'].astype(str)
    dates['date'] = dates['YEAR'] + ' ' + dates['WEEK'].map(str)
    datesTs = []
    for i in dates['date']:
        datesTs.append(datetime.datetime.strptime(i + '-0', "%Y %W-%w"))

    figure = plt.figure()
    sub = figure.add_subplot(111)
    plt.axvspan(datesTr[-7], datesTs[-1], facecolor='0.1', alpha=0.1, label="Test Set")
    sub.set_title("Model 2")
    sub.set_xlabel("Year")
    sub.set_ylabel("No. of Dengue Cases")
    
    #sub.plot(datesTr,y_scaled)
    #sub.plot(datesTr,predTr)
    #sub.plot(datesTs,yt_scaled)
    #sub.plot(datesTs,yt_predicted)
    sub.plot(datesTr,y, color='b', label='Actual')
    sub.plot(datesTr,invy_train, color='orange', label='Predicted')
    sub.plot(datesTs,yt, color='b')
    sub.plot(datesTs,invy_pred, color='orange')
    plt.legend()
    plt.show()
    
def drawtable(model: int, table: list):

    rows = ['Model %d' % x for x in (0, 1, 2, 3, 4)]
    # columns = ("week", "rmse", "mae", "r2s")

    df = pd.DataFrame(table, index = rows)
    df.columns = ["rmse", "rmse", "mae", "r2s","rmse", "mae", "r2s"]
    df.index.names = ['Weeks']
    df.columns.names = ['metric']
    h = [df.index.names[0] +'/'+ df.columns.names[0]] + list(df.columns)
    print(tabulate(df, headers= h, tablefmt= 'grid'))

def drawbargraph(table: list):

    n_groups = 4
    fgroup1 = np.array(table[0])
    fgroup2 = np.array(table[1])
    fgroup3 = np.array(table[2])
    fgroup4 = np.array(table[3])

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.10
    opacity = 0.8

    rects1 = plt.bar(index, fgroup1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='FS1')
    rects2 = plt.bar(index+bar_width, fgroup2, bar_width,
                 alpha=opacity,
                 color='g',
                 label='FS4')
    rects3 = plt.bar(index+bar_width*2, fgroup3, bar_width,
                 alpha=opacity,
                 color='m',
                 label='FS5')
    rects4 = plt.bar(index+bar_width*3, fgroup4, bar_width,
                 alpha=opacity,
                 color='c',
                 label='FS6')

    plt.xlabel('Week')
    plt.ylabel('R2 Score')
    plt.title('R2Score by Week')
    plt.xticks(index + bar_width, ('1', '2', '3', '4'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()


main()


