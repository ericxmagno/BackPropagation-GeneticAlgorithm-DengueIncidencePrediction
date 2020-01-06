"""Class that represents the network to be evolved."""
import random
import logging
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from train import train_and_score
from train import savethis
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np

# Seed for reproducibility
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 10.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.generation = -1

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            # self.network[key] = random.choice(self.nn_param_choices[key])
            a = self.nn_param_choices[key]
            self.network[key] = a[np.random.choice(len(a))]

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, fgroup, gen):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.generation == -1:
            self.generation = gen
        if self.accuracy == 10.:
            self.accuracy = train_and_score(self.network, dataset, fgroup)


    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Generation %d" % (self.generation))
        logging.info("Network Test RMSE: %.6f" % (self.accuracy))

    def save(self, filename, datasource, fgroup):
        self.accuracy = savethis(self.network, filename, datasource, fgroup)
