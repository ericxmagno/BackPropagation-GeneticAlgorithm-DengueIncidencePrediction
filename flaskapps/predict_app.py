import base64
import numpy as np
import io
import keras
import math
import pandas as pd
import tensorflow as tf
import pickle
from neupy import algorithms, estimators, environment
from keras import backend as keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from flask import request
from flask import jsonify
from flask import Flask
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from flask import render_template


import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from operator import itemgetter

# Seed for reproducibility
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)
early_stopper = EarlyStopping(monitor = 'val_loss', patience=10)
global graph
global modelgd1, modelgd2, modelgd3, modelgd4
global modelga1, modelga2, modelga3, modelga4
global modelgrnn1, modelgrnn2, modelgrnn3, modelgrnn4


app = Flask(__name__)

@app.route("/")
def mainPage():
    return render_template('predict.html')

def get_model():
    global graph
    global modelgd1, modelgd2, modelgd3, modelgd4
    global modelga1, modelga2, modelga3, modelga4
    global modelgrnn1, modelgrnn2, modelgrnn3, modelgrnn4


    # BP w/ GD
    modelgd1 = load_model('adam-1.h5')
    modelgd2 = load_model('adam-2.h5')
    modelgd3 = load_model('adam-3.h5')
    modelgd4 = load_model('adam-4.h5')
    print(" GD Models loaded!")

    # GA
    modelga1 = load_model('modelgaweek1.h5')
    modelga2 = load_model('modelgaweek2.h5')
    modelga3 = load_model('modelgaweek3.h5')
    modelga4 = load_model('modelgaweek4.h5')
    print(" GA Models loaded!")

    # GRNN
    with open('grnn-1.pickle', 'rb') as f:
        modelgrnn1 = pickle.load(f)
    with open('grnn-2.pickle', 'rb') as f:
        modelgrnn2 = pickle.load(f)
    with open('grnn-3.pickle', 'rb') as f:
        modelgrnn3 = pickle.load(f)
    with open('grnn-4.pickle', 'rb') as f:
        modelgrnn4 = pickle.load(f)
    print(" GRNN Models loaded!")
    graph = tf.get_default_graph()

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    data = message["data"]

    prediction_gd, y  = gd(data)
    prediction_ga = ga(data)
    prediction_grnn = grnn(data)
    
    
    for p in prediction_gd:
        print(p)
        

    response = {
        "prediction" : {
            'week1gd' : prediction_gd[0],
            'week2gd' : prediction_gd[1],
            'week3gd' : prediction_gd[2],
            'week4gd' : prediction_gd[3],

            'week1ga' : prediction_ga[0],
            'week2ga' : prediction_ga[1],
            'week3ga' : prediction_ga[2],
            'week4ga' : prediction_ga[3],

            'week1grnn' : prediction_grnn[0],
            'week2grnn' : prediction_grnn[1],
            'week3grnn' : prediction_grnn[2],
            'week4grnn' : prediction_grnn[3],

            'expected' : y
        }
    }

    return jsonify(response)

def ga(data):
    test = [data["cases1"], data["cases2"], data["cases3"], data["cases4"], data["rainfall11"], data["rainfall13"], data["tmax11"], data["tmax12"], 
            data["tmin9"], data["tmin11"], data["tmean11"], data["tmean12"], data["rh3"], data["rh4"], data["wind_speed4"], data["wind_speed3"]] # week1

    test2 = [data["cases2"], data["cases3"], data["cases4"], data["cases5"], data["rainfall11"], data["rainfall13"], data["tmax11"], data["tmax12"], 
            data["tmin9"], data["tmin11"], data["tmean11"], data["tmean12"], data["rh3"], data["rh4"], data["wind_speed4"], data["wind_speed3"]] # week2

    test3 = [data["cases3"], data["cases4"], data["cases5"], data["cases6"], data["rainfall11"], data["rainfall13"], data["tmax11"], data["tmax12"], 
            data["tmin9"], data["tmin11"], data["tmean11"], data["tmean12"], data["rh3"], data["rh4"], data["wind_speed4"], data["wind_speed3"]] # week3

    test4 = [data["cases4"], data["cases5"], data["cases6"], data["cases7"], data["rainfall11"], data["rainfall13"], data["tmax11"], data["tmax12"], 
            data["tmin9"], data["tmin11"], data["tmean11"], data["tmean12"], data["rh5"], data["rh4"], data["wind_speed4"], data["wind_speed5"]] # week4
    
    y = [{'target': data["target"]}]
    print(test)
    print(y)

    week1 = np.array(test)
    week2 = np.array(test2)
    week3 = np.array(test3)
    week4 = np.array(test4)
    y = pd.DataFrame.from_dict(y, orient='columns')
    y = y.values
    y = y.astype('float32')

    week1 = week1[~np.isnan(week1)]
    week2 = week2[~np.isnan(week2)]
    week3 = week3[~np.isnan(week3)]
    week4 = week4[~np.isnan(week4)]

    week1 = np.resize(week1, (16, 16))
    week2 = np.resize(week2, (16, 16))
    week3 = np.resize(week3, (16, 16))
    week4 = np.resize(week4, (16, 16))
    y = np.reshape(y, (-1,1))

    scalerx = joblib.load("scalerxga.save") 
    scalery = joblib.load("scaleryga.save")
    week1 = scalerx.transform(week1)[0, :]
    week1 = np.reshape(week1, (-1, 16))
    week2 = scalerx.transform(week2)[0, :]
    week2 = np.reshape(week2, (-1, 16))
    week3 = scalerx.transform(week3)[0, :]
    week3 = np.reshape(week3, (-1, 16))
    week4 = scalerx.transform(week4)[0, :]
    week4 = np.reshape(week4, (-1, 16))
    y = scalery.transform(y)

    prediction = []
    with graph.as_default():
        prediction.append(scalery.inverse_transform(modelga1.predict(week1, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelga2.predict(week2, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelga3.predict(week3, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelga4.predict(week4, verbose=0)).tolist())
    y =  str(scalery.inverse_transform(y))
    predict_flat = [item for p in prediction for item in p]
    prediction = [item for p in predict_flat for item in p]
    prediction = ["%.0f"%item for item in prediction]

    return prediction

def gd(data):
    test = [data["rainfall11"], data["tmax12"], data["tmin9"], data["tmean11"], data["rh4"], data["wind_speed4"],
        data["cases1"], data["cases2"], data["cases3"]] # week1

    test2 = [data["cases2"], data["cases3"], data["cases4"], data["rainfall11"], data["tmax12"], data["tmin9"], data["tmean11"], data["rh4"], data["wind_speed4"]] # week2

    test3 = [data["cases3"], data["cases4"], data["cases5"],data["rainfall11"], data["tmax12"], data["tmin9"], data["tmean11"], data["rh4"], data["wind_speed4"]] # week3

    test4 = [data["cases4"], data["cases5"], data["cases6"],data["rainfall11"], data["tmax12"], data["tmin9"], data["tmean11"], data["rh4"], data["wind_speed4"]]  # week4
    
    y = [{'target': data["target"]}]

    print(test)
    print(y)

    week1 = np.array(test)
    week2 = np.array(test2)
    week3 = np.array(test3)
    week4 = np.array(test4)
    y = pd.DataFrame.from_dict(y, orient='columns')
    y = y.values
    y = y.astype('float32')

    week1 = week1[~np.isnan(week1)]
    week2 = week2[~np.isnan(week2)]
    week3 = week3[~np.isnan(week3)]
    week4 = week4[~np.isnan(week4)]

    week1 = np.resize(week1, (9, 9))
    week2 = np.resize(week2, (9, 9))
    week3 = np.resize(week3, (9, 9))
    week4 = np.resize(week4, (9, 9))
    y = np.reshape(y, (-1,1))

    scalerx = joblib.load("scalerxgd.save") 
    scalery = joblib.load("scalerygd.save")
    week1 = scalerx.transform(week1)[0, :]
    week1 = np.reshape(week1, (-1, 9))
    week2 = scalerx.transform(week2)[0, :]
    week2 = np.reshape(week2, (-1, 9))
    week3 = scalerx.transform(week3)[0, :]
    week3 = np.reshape(week3, (-1, 9))
    week4 = scalerx.transform(week4)[0, :]
    week4 = np.reshape(week4, (-1, 9))
    y = scalery.transform(y)

    prediction = []
    with graph.as_default():
        prediction.append(scalery.inverse_transform(modelgd1.predict(week1, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelgd2.predict(week2, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelgd3.predict(week3, verbose=0)).tolist())
        prediction.append(scalery.inverse_transform(modelgd4.predict(week4, verbose=0)).tolist())
    y =  str(scalery.inverse_transform(y))
    predict_flat = [item for p in prediction for item in p]
    prediction = [item for p in predict_flat for item in p]
    prediction = ["%.0f"%item for item in prediction]

    return prediction, y

def grnn(data):
    test = [data["cases1"], data["cases2"], data["tmean11"], data["wind_speed4"]] # week1

    test2 = [data["cases2"], data["cases3"], data["tmean11"], data["wind_speed4"]] # week2

    test3 = [data["cases3"], data["cases4"], data["tmean11"], data["wind_speed4"]] # week3

    test4 = [data["cases4"], data["cases5"], data["tmean11"], data["wind_speed4"]]  # week4
    
    y = [{'target': data["target"]}]

    print(test)
    print(y)

    week1 = np.array(test)
    week2 = np.array(test2)
    week3 = np.array(test3)
    week4 = np.array(test4)
    y = pd.DataFrame.from_dict(y, orient='columns')
    y = y.values
    y = y.astype('float32')

    week1 = week1[~np.isnan(week1)]
    week2 = week2[~np.isnan(week2)]
    week3 = week3[~np.isnan(week3)]
    week4 = week4[~np.isnan(week4)]

    week1 = np.resize(week1, (4, 4))
    week2 = np.resize(week2, (4, 4))
    week3 = np.resize(week3, (4, 4))
    week4 = np.resize(week4, (4, 4))
    y = np.reshape(y, (-1,1))

    scalerx = joblib.load("scalerxgrnn.save") 
    scalery = joblib.load("scalerygrnn.save")
    week1 = scalerx.transform(week1)[0, :]
    week1 = np.reshape(week1, (-1, 4))
    week2 = scalerx.transform(week2)[0, :]
    week2 = np.reshape(week2, (-1, 4))
    week3 = scalerx.transform(week3)[0, :]
    week3 = np.reshape(week3, (-1, 4))
    week4 = scalerx.transform(week4)[0, :]
    week4 = np.reshape(week4, (-1, 4))
    y = scalery.transform(y)

    environment.reproducible()

    prediction = []
    with graph.as_default():
        prediction.append(scalery.inverse_transform(modelgrnn1.predict(week1).tolist()))
        prediction.append(scalery.inverse_transform(modelgrnn2.predict(week2).tolist()))
        prediction.append(scalery.inverse_transform(modelgrnn3.predict(week3).tolist()))
        prediction.append(scalery.inverse_transform(modelgrnn4.predict(week4).tolist()))
    y =  str(scalery.inverse_transform(y))
    predict_flat = [item for p in prediction for item in p]
    prediction = [item for p in predict_flat for item in p]
    prediction = ["%.0f"%item for item in prediction]

    return prediction
