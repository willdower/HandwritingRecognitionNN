import gzip
import numpy as np
import matplotlib
import sys

training_set_data = np.loadtxt(gzip.open(sys.argv[5], "rb"), delimiter=",")

n_epochs = 30
batch_size = 20
learning_rate = 3

print(training_set_data)