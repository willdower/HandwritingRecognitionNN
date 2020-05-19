import gzip
import numpy as np
import matplotlib.pyplot as plt
import sys

def randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons):
    W_1 = np.random.normal(loc=0, scale=1, size=(n_input_neurons, n_hidden_neurons))
    W_2 = np.random.normal(loc=0, scale=1, size=(n_hidden_neurons, n_output_neurons))
    B_hidden = np.random.normal(loc=0, scale=1, size=(1, n_hidden_neurons))
    B_output = np.random.normal(loc=0, scale=1, size=(1, n_output_neurons))
    return W_1, W_2, B_hidden, B_output

def sigmoid(net):
    return 1 / (1 + np.exp(-net))

def predict(W_output, W_hidden, B_hidden, B_output, x):
    net_hidden = np.dot(x, W_hidden) + B_hidden
    out_hidden = sigmoid(net_hidden)

    net_output = np.dot(out_hidden, W_output) + B_output
    out_output = sigmoid(net_output)

    return out_output

def forward_pass(W_output, W_hidden, B_hidden, B_output, x, y):
    net_hidden = np.dot(x, W_hidden) + B_hidden
    out_hidden = sigmoid(net_hidden)

    net_output = np.dot(out_hidden, W_output) + B_output
    out_output = sigmoid(net_output)

    cost = 0.5*(out_output - y)**2

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

def new_backward_pass(x, y, output, hidden_output, W_output):
    output_error = -(y - output)
    output_over_net = output*(1 - output)
    sigmoid_on_error = np.multiply(output_error, output_over_net)

    W_output = np.transpose(W_output)
    hidden_error = np.dot(sigmoid_on_error, W_output)
    hidden_over_net = hidden_output*(1 - hidden_output)
    sigmoid_on_hidden_error = np.multiply(hidden_error, hidden_over_net)

    x = np.atleast_2d(x)
    hidden_output = np.atleast_2d(hidden_output)
    x_transpose = np.transpose(x)
    hidden_output_transpose = np.transpose(hidden_output)
    sigmoid_on_hidden_error = sigmoid_on_hidden_error.reshape(1, sigmoid_on_hidden_error.size)
    sigmoid_on_error = sigmoid_on_error.reshape(1, sigmoid_on_error.size)
    W_hidden_c = np.dot(x_transpose, sigmoid_on_hidden_error)
    W_output_c = np.dot(hidden_output_transpose, sigmoid_on_error)

    B_hidden_c = sigmoid_on_hidden_error
    B_output_c = sigmoid_on_error

    return W_output_c, W_hidden_c, B_hidden_c, B_output_c

def neural_network(n_epochs, batch_size, learning_rate, changing, training_set_x, training_set_y, testing_set_x, testing_set_y):
    error_list = []
    max_accuracy = 0
    max_accuracy_epoch = 0

    n_input_neurons = int(sys.argv[1])
    n_hidden_neurons = int(sys.argv[2])
    n_output_neurons = int(sys.argv[3])

    n_samples = training_set_x.shape[0]
    n_batches = round(n_samples / batch_size)

    W_hidden, W_output, B_hidden, B_output = randomly_initialize_W(n_input_neurons, n_hidden_neurons, n_output_neurons)

    print("Training begun for LR of " + str(learning_rate) + ", BS of " + str(batch_size) + ", and " + str(n_epochs) + " epochs")

    for epoch in range(n_epochs):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(training_set_x)
        np.random.set_state(rng_state)
        np.random.shuffle(training_set_y)

        x_batches = np.array_split(training_set_x, n_batches)
        split_points = []
        split_point = 0
        for batch in x_batches:
            split_point += batch.shape[0]
            split_points.append(split_point)

        y_batches = np.split(training_set_y, split_points)


        for batch in range(n_batches):

            current_batch_x = x_batches[batch]
            current_batch_y = y_batches[batch]

            sum_Woutput_change = np.zeros((W_output.shape[0], W_output.shape[1]))
            sum_Whidden_change = np.zeros((W_hidden.shape[0], W_hidden.shape[1]))
            sum_Bhidden_change = np.zeros((1, B_hidden.size))
            sum_Boutput_change = np.zeros((1, B_output.size))


            for num in range(current_batch_x.shape[0]):
                x = current_batch_x[num]
                x = x.reshape(1, x.size)
                y = np.zeros((1, n_output_neurons))
                y[0][int(current_batch_y[num])] = 1

                cost, out_hidden, out_output = forward_pass(W_output, W_hidden, B_hidden, B_output, x, y)
                W_output_c, W_hidden_c, B_hidden_c, B_output_c = new_backward_pass(x, y, out_output, out_hidden, W_output)
                sum_Woutput_change += W_output_c
                sum_Whidden_change += W_hidden_c
                sum_Bhidden_change += B_hidden_c
                sum_Boutput_change += B_output_c

            sum_Woutput_change = sum_Woutput_change / current_batch_x.shape[0]
            sum_Whidden_change = sum_Whidden_change / current_batch_x.shape[0]
            sum_Bhidden_change = sum_Bhidden_change / current_batch_x.shape[0]
            sum_Boutput_change = sum_Boutput_change / current_batch_x.shape[0]

            W_output = W_output - (learning_rate * sum_Woutput_change)
            W_hidden = W_hidden - (learning_rate * sum_Whidden_change)
            B_hidden = B_hidden - (learning_rate * sum_Bhidden_change)
            B_output = B_output - (learning_rate * sum_Boutput_change)

        successful = 0
        failure = 0
        for num in range(testing_set_x.shape[0]):
            x = testing_set_x[num].reshape(1, testing_set_x[num].size)
            out = predict(W_output, W_hidden, B_hidden, B_output, x)
            prediction = np.argmax(out)
            if (prediction == testing_set_y[num]):
                successful += 1
            else:
                failure += 1
        # print("\nEpoch: " + str(epoch + 1))
        # print("Successful: " + str(successful) + " Failure: " + str(failure))
        percentage = (successful / (successful + failure)) * 100
        # print("Percentage Successful: " + str(percentage))
        error_list.append(percentage)

        if percentage > max_accuracy:
            max_accuracy = percentage
            max_accuracy_epoch = epoch + 1

    print("\nTraining complete, max accuracy on epoch " + str(max_accuracy_epoch) + " of " + str(max_accuracy) + "%")
    plot_x = range(n_epochs)

    if changing == 0:
        plt.plot(plot_x, error_list, label='%f Learning Rate' % learning_rate)
    elif changing == 1:
        plt.plot(plot_x, error_list, label='%d Mini-Batch Size' % batch_size)
    else:
        plt.plot(plot_x, error_list, label='%d Epochs' % n_epochs)


np.seterr(over='ignore')

print("Begun loading files...")
training_set_x = np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")
training_set_y = np.genfromtxt(gzip.open(sys.argv[5], "rb"), delimiter=",")

testing_set_x = np.genfromtxt(gzip.open(sys.argv[6], "rb"), delimiter=",")
testing_set_y = np.genfromtxt(gzip.open(sys.argv[7], "rb"), delimiter=",")
print("Finished loading files.")

batch_sizes = [1, 5, 10, 20, 100]
for batch_size in batch_sizes:
    neural_network(30, batch_size, 3, 1, training_set_x, training_set_y, testing_set_x, testing_set_y)

plt.ylim(0, 100)
plt.xlabel("Epochs")
plt.ylabel("Percentage Accuracy")
plt.title("Percentage Accuracy over Epochs for Different Batch Sizes")
plt.legend()
save_name = "Accuracy BS.png"
plt.savefig(save_name)
plt.clf()
plt.cla()

learning_rates = [0.001, 0.1, 1.0, 10, 100]
for learning_rate in learning_rates:
    neural_network(30, 20, learning_rate, 0, training_set_x, training_set_y, testing_set_x, testing_set_y)

plt.ylim(0, 100)
plt.xlabel("Epochs")
plt.ylabel("Percentage Accuracy")
plt.title("Percentage Accuracy over Epochs for Different Learning Rates")
plt.legend()
save_name = "Accuracy LR.png"
plt.savefig(save_name)
plt.clf()
plt.cla()

epochs = [1, 5, 10, 20, 30, 50]
for n_epochs in epochs:
    neural_network(n_epochs, 20, 3, 2, training_set_x, training_set_y, testing_set_x, testing_set_y)

plt.ylim(0, 100)
plt.xlabel("Epochs")
plt.ylabel("Percentage Accuracy")
plt.title("Percentage Accuracy over Epochs for Different Number of Epochs")
plt.legend()
save_name = "Accuracy Epoch.png"
plt.savefig(save_name)
plt.clf()
plt.cla()
