import pickle
import matplotlib.pyplot as plt
import numpy as np

import model

# General parameters
epochs = 1

# Dataset preparation
X_train_data = pickle.load(open('mfccs_train_batches.pickle', 'rb'))[:1000]
y_train_data = pickle.load(open('phonems_train_batches.pickle', 'rb'))[:1000]
X_test_data = pickle.load(open('mfccs_test.pickle', 'rb'))[:1000]
y_test_data = pickle.load(open('phonems_test.pickle', 'rb'))[:1000]


# Model
network = model.ALIFNetwork(n_in=13, n_rec=6, n_out=39, tau_m=20., tau_a=200., thr=1.6, dt=1., n_refractory=2., beta=0.184)

for epoch in range(epochs):
    batch_loss = []

    # For each batch of size 32
    for X_batch, y_batch in zip(X_train_data, y_train_data):
        # Forward pass
        outputs, voltage, spikes, adaptation, etrace, weights_rec = network.forward(X_batch, y_batch)

        # Performance monitoring loss
        loss = network.cross_entropy(outputs, y_batch)
        batch_loss.append(loss)
    
    print(f"Epoch {epoch}, Loss: {np.mean(batch_loss):.4f}")


fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 7), constrained_layout=True)

idx = 5
input = X_batch
x = range(len(input))

v = np.asarray(voltage)
s = np.asarray(spikes)
a = np.asarray(adaptation)
e = np.asarray(etrace)
w = np.asarray(weights_rec)

# Plot data in each subplot
axs[0].plot(x, input)
axs[0].set_title("Input")
axs[0].set_xlabel("t")

axs[1].plot(x, v[:, idx], color='blue')
axs[1].set_title("Voltage")
axs[1].set_xlabel("t")
axs[1].set_ylabel("v")

axs[2].plot(x, a[:, idx], color='red')
axs[2].set_title("Adaptive Threshold")
axs[2].set_xlabel("t")

axs[3].plot(x, s[:, idx], color='red')
axs[3].set_title("Spikes")
axs[3].set_xlabel("t")

axs[4].plot(x, e[:, idx], color='blue')
axs[4].set_title("Eligibility trace")
axs[4].set_xlabel("t")

axs[5].plot(x, w[:, :, idx], color='blue')
axs[5].set_title("Weight")
axs[5].set_xlabel("t")

plt.show()

# Testing the trained model
correct_predictions = 0
total_samples = 0

# Iterate through test data
for X, y in zip(X_test_data, y_test_data):
    # Forward pass
    outputs, _, _, _, _, _ = network.forward(np.asarray([X]), [y])
    predictions = np.argmax(outputs, axis=1)
    
    correct_predictions += np.sum(predictions == y)
    total_samples += len(y_batch)

# Calculate accuracy
accuracy = correct_predictions / total_samples * 100
print(f"Test Accuracy: {accuracy:.2f}%")