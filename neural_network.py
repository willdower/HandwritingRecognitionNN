import gzip
import numpy as np
import matplotlib
import sys

def randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons):
    W_1 = np.random.normal(loc=0, scale=1, size=(n_input_neurons, n_hidden_neurons))
    W_2 = np.random.normal(loc=0, scale=1, size=(n_hidden_neurons, n_output_neurons))
    B_hidden = np.random.normal(loc=0, scale=1, size=(n_hidden_neurons, 1))
    B_output = np.random.normal(loc=0, scale=1, size=(n_output_neurons, 1))
    return W_1, W_2, B_hidden, B_output

def sigmoid(net):
    return 1 / (1 + np.exp(-net))

def forward_pass(W_output, W_hidden, B_hidden, B_output, x, y):
    net_hidden = np.dot(x, W_hidden) + B_hidden
    out_hidden = sigmoid(net_hidden)

    net_output = np.dot(W_output, out_hidden) + B_output
    out_output = sigmoid(net_output)

    cost = 0.5*(y - out_output)**2

    return cost, out_hidden, out_output

def backward_pass(x, y, out_hidden, out_output):
    Etotal_over_out = -(y - out_output)
    output_over_net = out_output*(1 - out_output)
    Etotal_over_net = np.multiply(Etotal_over_out, output_over_net)
    Etotal_over_net = Etotal_over_net.reshape(1, 2)
    out_hidden = out_hidden.reshape(1, 2)
    W_output_c = np.dot(np.transpose(out_hidden), Etotal_over_net)
    B_output_c = Etotal_over_net



    return W_output_c, B_output_c

def new_backward_pass(x, y, output, hidden_output, W_output, W_hidden, B_output, B_hidden):
    output_error = -(y - output)
    output_over_net = output*(1 - output)
    sigmoid_on_error = np.multiply(output_error, output_over_net)

    W_output = np.transpose(W_output)
    hidden_error = np.dot(sigmoid_on_error, W_output)
    hidden_over_net = hidden_output*(1 - hidden_output)
    sigmoid_on_hidden_error = np.multiply(hidden_error, hidden_over_net)

    x_transpose = x.reshape(-x.size-1, x.size-1)
    hidden_output_transpose = hidden_output.reshape(-hidden_output.size-1, hidden_output.size-1)
    sigmoid_on_hidden_error = sigmoid_on_hidden_error.reshape(1, sigmoid_on_hidden_error.size)
    sigmoid_on_error = sigmoid_on_error.reshape(1, sigmoid_on_error.size)
    W_hidden_c = np.dot(x_transpose, sigmoid_on_hidden_error)
    W_output_c = np.dot(hidden_output_transpose, sigmoid_on_error)

    B_hidden_c = sigmoid_on_hidden_error
    B_output_c = sigmoid_on_error

    print("test")

def neural_network():
    # training_set_data = np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")
    training_set_x = np.array([[0.1, 0.1], [0.1, 0.2]])
    training_set_y = np.array([[1, 0], [0, 1]])

    n_epochs = 1
    batch_size = 2
    learning_rate = 0.1

    # n_input_neurons = int(sys.argv[1])
    n_input_neurons = 2
    n_hidden_neurons = 2
    n_output_neurons = 2
    # n_hidden_neurons = int(sys.argv[2])
    # n_output_neurons = int(sys.argv[3])
    bias_value = 1

    n_samples = training_set_x.shape[0]
    n_batches = round(n_samples / batch_size)

    x_batches = np.array_split(training_set_x, n_batches)
    split_points = []
    for batch in x_batches:
        split_points.append(batch.size)

    y_batches = np.split(training_set_y, split_points)
    W_output, W_hidden, B_hidden, B_output = randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons)
    W_output = np.array([[0.1, 0.1], [0.1, 0.2]])
    W_hidden = np.array([[0.1, 0.2], [0.1, 0.1]])
    B_hidden = np.array([0.1, 0.1])
    B_output = np.array([0.1, 0.1])

    for epoch in range(n_epochs):
        for batch in range(n_batches):

            current_batch_x = x_batches[batch]
            current_batch_y = y_batches[batch]

            sum_Woutput_change = np.zeros(W_output.size)
            sum_Whidden_change = np.zeros(W_hidden.size)
            sum_Bhidden_change = np.zeros(B_hidden.size)
            sum_Boutput_change = np.zeros(B_output.size)


            for num in range(current_batch_x.size-1):
                cost, out_hidden, out_output = forward_pass(W_output, W_hidden, B_hidden, B_output, current_batch_x[num], current_batch_y[num])
                # W_output_c, B_output_c = backward_pass(current_batch_x[num], current_batch_y[num], out_hidden, out_output)
                new_backward_pass(current_batch_x[num], current_batch_y[num], out_output, out_hidden, W_output, W_hidden, B_output, B_hidden)


 # USE BATCH TRANSPOSE FOR X AND Y

neural_network()