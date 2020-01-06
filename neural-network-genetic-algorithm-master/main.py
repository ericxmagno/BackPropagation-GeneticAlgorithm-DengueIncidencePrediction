"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Model

# Code adapted from this project
# https://github.com/harvitronix/neural-network-genetic-algorithm


# Seed for reproducibility
from numpy.random import seed
seed(9)
from tensorflow import set_random_seed
set_random_seed(9)


# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset, fgroup, gen):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, fgroup, gen)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset, fgroup):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, fgroup, i+1)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation MSE average: %.6f" % (average_accuracy))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

    # Print out the top 5 networks.
    print_networks(networks[:5], dataset, fgroup)

def print_networks(networks, dataset, fgroup):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    filenum = 0
    print("---FGROUP %d---" % (fgroup))
    for network in networks:
        filename = str(filenum)
        network.save(filename, dataset,fgroup)
        filenum+=1
        network.print_network() 
        


def main():
    """Evolve a network."""
    generations = 20  # Number of times to evolve the population.
    population = 100  # Number of networks in each generation.
    dataset = 'csvs/full_dataset_extended.csv'


    nn_param_choices = { 
        'nb_neurons': [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256, 512, 1024],
        'nb_layers': [1, 2, 3],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'linear'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations wit. h population %d***" %
                 (generations, population))

    featuregroup = 4 # 3 of 4
    generate(generations, population, nn_param_choices, dataset, featuregroup)

if __name__ == '__main__':
    main()
