import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
import model
import torch
import torch.optim as optim
import torch.nn.functional as F

# General parameters
np.random.seed(123)
epochs = 80

# Dataset preparation
X_train_data = np.asarray(pickle.load(open('mfccs_train_batches.pickle', 'rb')), dtype=object)
y_train_data = np.asarray(pickle.load(open('phonems_train_batches.pickle', 'rb')), dtype=object)
indeces = np.random.choice(X_train_data.shape[0], size=3000, replace=False)
X_train_data = X_train_data[indeces]
y_train_data = y_train_data[indeces]

X_test_data = np.asarray(pickle.load(open('mfccs_test.pickle', 'rb')), dtype=object)
y_test_data = np.asarray(pickle.load(open('phonems_test.pickle', 'rb')), dtype=object)
indeces = np.random.choice(X_test_data.shape[0], size=3000, replace=False)
X_test_data = X_test_data[indeces]
y_test_data = y_test_data[indeces]

# Model
network = model.ALIFNetwork(n_in=13, n_rec=100, n_out=61, tau_m=2., tau_a=200., tau_out=3., thr=1.6, dt=1., n_refractory=2., beta=0.005)
optimizer = optim.Adam(network.parameters(), lr=network.lr)

all_loss = []
weights_in = []
weights_rec = []
weights_out = []
bias_out = []

for epoch in range(epochs):
    weights_rec.append(copy.deepcopy(network.w_rec.data))
    weights_in.append(copy.deepcopy(network.w_in.data))
    weights_out.append(copy.deepcopy(network.w_out.data))
    bias_out.append(copy.deepcopy(network.b_out.data))
    batch_loss = []

    network.reset_gradients() # set gradients to 0

    # For each batch of size 32
    batch_loss = []
    for X_batch, y_batch in zip(X_train_data, y_train_data):
        # Forward pass
        outputs, voltage, spikes, adaptation = network.forward(torch.from_numpy(X_batch), torch.from_numpy(np.asarray(y_batch)), training=True)

        # Performance monitoring loss
        loss = F.cross_entropy(torch.from_numpy(np.asarray(outputs)), torch.from_numpy(np.asarray(y_batch)), reduction='mean')
        batch_loss.append(loss)

    optimizer.step() # upgrade weights
    network.w_rec.data.fill_diagonal_(0) # remove self loops
    # reset_state()
    
    mean_loss = torch.mean(torch.from_numpy(np.asarray(batch_loss)))
    print(f"Epoch {epoch}, Loss: {mean_loss:.4f}")
    all_loss.append(mean_loss)

plt.plot(range(epochs), all_loss)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

idx = 5

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
    outputs, _, _, _ = network.forward(torch.from_numpy(np.asarray([X], dtype='float64')), torch.from_numpy(np.asarray([y])), training=False)
    predictions = np.argmax(outputs, axis=1)

    # always predicts the same, that's why accuracy is 0    
    correct_predictions += np.sum(predictions == y)

# Calculate accuracy
accuracy = correct_predictions / len(y_test_data) * 100
print(f"Test Accuracy: {accuracy:.2f}%")