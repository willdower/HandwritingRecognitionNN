import gzip
import numpy as np
import matplotlib
import sys

def randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons):
    W_1 = np.random.normal(loc=0, scale=1, size=(n_input_neurons, n_hidden_neurons))
    W_2 = np.random.normal(loc=0, scale=1, size=(n_hidden_neurons, n_output_neurons))
    return W_1, W_2

def neural_network():
    # training_set_data = np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")
    training_set_data = np.array([[0.1, 0.1], [0.1, 0.2]])

    n_epochs = 1
    batch_size = 2
    learning_rate = 0.1

    n_input_neurons = sys.argv[1]
    n_hidden_neurons = sys.argv[2]
    n_output_neurons = sys.argv[3]
    bias_value = 1

    n_samples = training_set_data.shape[0]
    n_batches = round(n_samples / batch_size)

    batches = np.array_split(training_set_data, n_batches)
    W = randomly_initialize_W(n_input_neurons, n_hidden_neurons)
