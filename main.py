import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
import model
import plots

# General parameters
np.random.seed(123)
epochs = 5

# Dataset preparation
X_train_data = np.asarray(pickle.load(open('mfccs_train_batches.pickle', 'rb')), dtype=object)
y_train_data = np.asarray(pickle.load(open('phonems_train_batches.pickle', 'rb')), dtype=object)
indeces = np.random.choice(X_train_data.shape[0], size=1000, replace=False)
X_train_data = X_train_data[indeces]
y_train_data = y_train_data[indeces]

X_test_data = np.asarray(pickle.load(open('mfccs_test.pickle', 'rb')), dtype=object)
y_test_data = np.asarray(pickle.load(open('phonems_test.pickle', 'rb')), dtype=object)
indeces = np.random.choice(X_test_data.shape[0], size=1000, replace=False)
X_test_data = X_test_data[indeces]
y_test_data = y_test_data[indeces]

# Model
network = model.ALIFNetwork(n_in=13, n_rec=10, n_out=61, tau_m=2., tau_a=200., tau_out=3., thr=1.6, dt=1., n_refractory=2., beta=0.005)

all_loss = []
weights_in = []
weights_rec = []
weights_out = []
bias_out = []
for epoch in range(epochs):
    weights_rec.append(copy.deepcopy(network.w_rec))
    weights_in.append(copy.deepcopy(network.w_in))
    weights_out.append(copy.deepcopy(network.w_out))
    bias_out.append(copy.deepcopy(network.b_out))
    batch_loss = []

    # For each batch of size 32
    for X_batch, y_batch in zip(X_train_data, y_train_data):
        # Forward pass
        outputs, voltage, spikes, adaptation = network.forward(X_batch, y_batch, training=True)

        # Performance monitoring loss
        loss = network.cross_entropy(outputs, y_batch)
        batch_loss.append(loss)

    # plots.heatmaps(network, outputs, y_batch)
    network.weight_update()
    network.reset_gradients()
    
    print(f"Epoch {epoch}, Loss: {np.mean(batch_loss):.4f}")
    all_loss.append(np.mean(batch_loss))


# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 7), constrained_layout=True)

idx = 5
# input = X_batch
# x = range(len(input))

# v = np.asarray(voltage)
# s = np.asarray(spikes)
# a = np.asarray(adaptation)

# axs[0].plot(x, input)
# axs[0].set_title("Input")
# axs[0].set_xlabel("t")

# axs[1].plot(x, v[:, idx], color='blue')
# axs[1].set_title("Voltage")
# axs[1].set_xlabel("t")
# axs[1].set_ylabel("v")

# axs[2].plot(x, a[:, idx], color='red')
# axs[2].set_title("Adaptive Threshold")
# axs[2].set_xlabel("t")

# axs[3].plot(x, s[:, idx], color='red')
# axs[3].set_title("Spikes")
# axs[3].set_xlabel("t")












fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 7), constrained_layout=True)

w_in = np.asarray(weights_in)
w_rec = np.asarray(weights_rec)
w_out = np.asarray(weights_out)
b_out = np.asarray(bias_out)

axs[0].plot(range(epochs), w_rec[:, :, idx], color='blue')
axs[0].set_title("Recurrent weight")
axs[0].set_xlabel("epoch")

axs[1].plot(range(epochs), w_out[:, :, idx], color='blue')
axs[1].set_title("Output weight")
axs[1].set_xlabel("epoch")

axs[2].plot(range(epochs), b_out[:, :], color='red')
axs[2].set_title("Output bias")
axs[2].set_xlabel("epoch")

axs[3].plot(range(epochs), w_in[:, :, idx], color='blue')
axs[3].set_title("Input weight")
axs[3].set_xlabel("epoch")

plt.show()

# Testing the trained model
correct_predictions = 0
total_samples = 0

# Iterate through test data
for X, y in zip(X_test_data, y_test_data):
    # Forward pass
    outputs, _, _, _ = network.forward(np.asarray([X]), [y], training=False)
    predictions = np.argmax(outputs, axis=1)

    # always predicts the same, that's why accuracy is 0    
    correct_predictions += np.sum(predictions == y)
    total_samples += len(y_batch)

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f"Test Accuracy: {accuracy:.2f}%")