import gzip
import numpy as np
import csv
import sys

# Randomly initializes the weight matrices according to the normal distribution
def randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons):
    W_1 = np.random.normal(loc=0, scale=1, size=(n_input_neurons, n_hidden_neurons))
    W_2 = np.random.normal(loc=0, scale=1, size=(n_hidden_neurons, n_output_neurons))
    B_hidden = np.random.normal(loc=0, scale=1, size=(1, n_hidden_neurons))
    B_output = np.random.normal(loc=0, scale=1, size=(1, n_output_neurons))
    return W_1, W_2, B_hidden, B_output

# The activation function
def sigmoid(net):
    return 1 / (1 + np.exp(-net))

# Forward pass to determine the outputs
def forward_pass(W_output, W_hidden, B_hidden, B_output, x):
    net_hidden = np.dot(x, W_hidden) + B_hidden
    out_hidden = sigmoid(net_hidden)

    net_output = np.dot(out_hidden, W_output) + B_output
    out_output = sigmoid(net_output)

    return out_hidden, out_output

# Backwards pass calculates how the weights need to change to minimise the cost (difference between output and target)
def backward_pass(x, y, output, hidden_output, W_output):
    output_error = -(y - output)  # Calculate error
    output_over_net = output*(1 - output)  # Derivative of sigmoid function
    sigmoid_on_error = np.multiply(output_error, output_over_net)  # Calculate the sigmoid function's affect on error

    W_output = np.transpose(W_output)
    hidden_error = np.dot(sigmoid_on_error, W_output)  # Calculate the affect of output weights on hidden weights' error
    hidden_over_net = hidden_output*(1 - hidden_output)  # Derivative of sigmoid function
    sigmoid_on_hidden_error = np.multiply(hidden_error, hidden_over_net)  # Calculate the sigmoid function's affect on error

    # Correctly arrange matrices for calculations
    x = np.atleast_2d(x)
    hidden_output = np.atleast_2d(hidden_output)
    x_transpose = np.transpose(x)
    hidden_output_transpose = np.transpose(hidden_output)
    sigmoid_on_hidden_error = sigmoid_on_hidden_error.reshape(1, sigmoid_on_hidden_error.size)
    sigmoid_on_error = sigmoid_on_error.reshape(1, sigmoid_on_error.size)

    # Calculate weight changes
    W_hidden_c = np.dot(x_transpose, sigmoid_on_hidden_error)
    W_output_c = np.dot(hidden_output_transpose, sigmoid_on_error)

    # Calculate bias changes
    B_hidden_c = sigmoid_on_hidden_error
    B_output_c = sigmoid_on_error

    return W_output_c, W_hidden_c, B_hidden_c, B_output_c

def neural_network(n_epochs, batch_size, learning_rate, training_set_x, training_set_y, testing_set_x):

    # Get the number of neurons in each layer
    n_input_neurons = int(sys.argv[1])
    n_hidden_neurons = int(sys.argv[2])
    n_output_neurons = int(sys.argv[3])

    # Calculate samples and batches
    n_samples = training_set_x.shape[0]
    n_batches = round(n_samples / batch_size)

    # Initialise weights
    W_hidden, W_output, B_hidden, B_output = randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons)

    print("Training begun for LR of " + str(learning_rate) + ", BS of " + str(batch_size) + ", and " + str(n_epochs) + " epochs")

    for epoch in range(n_epochs):

        # Ensure inputs and labels are shuffled the same way so they stay together (x_1 == y_1)
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(training_set_x)
        np.random.set_state(rng_state)
        np.random.shuffle(training_set_y)

        # Split into mini-batches
        x_batches = np.array_split(training_set_x, n_batches)
        split_points = []
        split_point = 0
        for batch in x_batches:
            split_point += batch.shape[0]
            split_points.append(split_point)

        y_batches = np.split(training_set_y, split_points)


        for batch in range(n_batches):

            # Get mini-batch values
            current_batch_x = x_batches[batch]
            current_batch_y = y_batches[batch]

            # Initialize sums to 0
            sum_Woutput_change = np.zeros((W_output.shape[0], W_output.shape[1]))
            sum_Whidden_change = np.zeros((W_hidden.shape[0], W_hidden.shape[1]))
            sum_Bhidden_change = np.zeros((1, B_hidden.size))
            sum_Boutput_change = np.zeros((1, B_output.size))

            # For every sample in mini-batch
            for num in range(current_batch_x.shape[0]):
                x = current_batch_x[num]
                x = x.reshape(1, x.size)
                y = np.zeros((1, n_output_neurons))
                y[0][int(current_batch_y[num])] = 1

                # Do forward pass
                out_hidden, out_output = forward_pass(W_output, W_hidden, B_hidden, B_output, x)

                #Do backward pass
                W_output_c, W_hidden_c, B_hidden_c, B_output_c = backward_pass(x, y, out_output, out_hidden, W_output)

                # Add to change sums
                sum_Woutput_change += W_output_c
                sum_Whidden_change += W_hidden_c
                sum_Bhidden_change += B_hidden_c
                sum_Boutput_change += B_output_c

            # Average across entire mini-batch
            sum_Woutput_change = sum_Woutput_change / current_batch_x.shape[0]
            sum_Whidden_change = sum_Whidden_change / current_batch_x.shape[0]
            sum_Bhidden_change = sum_Bhidden_change / current_batch_x.shape[0]
            sum_Boutput_change = sum_Boutput_change / current_batch_x.shape[0]

            # Apply weight changes
            W_output = W_output - (learning_rate * sum_Woutput_change)
            W_hidden = W_hidden - (learning_rate * sum_Whidden_change)
            B_hidden = B_hidden - (learning_rate * sum_Bhidden_change)
            B_output = B_output - (learning_rate * sum_Boutput_change)

    print("\nTraining complete")

    # Get predicts for input test file
    output_arr = []
    for num in range(testing_set_x.shape[0]):
        x = testing_set_x[num].reshape(1, testing_set_x[num].size)
        hidden_out, out = forward_pass(W_output, W_hidden, B_hidden, B_output, x)
        prediction = np.argmax(out)
        output_arr.append(prediction)

    # Write predictions to file
    output_filename = sys.argv[7]
    with gzip.open(output_filename, 'wt', newline="") as file:
        csv_file = csv.writer(file)
        for prediction in output_arr:
            csv_file.writerow([prediction])


np.seterr(over='ignore')

print("Begun loading files...")

# Load training set
training_set_x = np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")
training_set_y = np.genfromtxt(gzip.open(sys.argv[5], "rb"), delimiter=",")

# Load testing set
testing_set_x = np.genfromtxt(gzip.open(sys.argv[6], "rb"), delimiter=",")
print("Finished loading files.")

neural_network(30, 20, 3, training_set_x, training_set_y, testing_set_x)
