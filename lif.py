import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from itertools import chain

path = '/Users/giselaalbors/Desktop/university/master/semester 5/e-prop-lsnn/timit_processed/train/'


class ALIFNeuron:
    def __init__(self, n_in, n_rec, n_out, t_total, tau_m=20., tau_a=200., thr=1.6, dt=1., n_refractory=2., beta=0.07):
        """
        Initializes an LIF neuron with parameters for learning and recurrence.
        :param n_in: Number of input neurons.
        :param n_rec: Number of recurrent neurons.
        :param tau: Membrane time constant.
        :param thr: Spike threshold.
        :param dt: Discrete time step.
        :param dampening_factor: Dampening factor for pseudo-derivative.
        :param n_refractory: Number of refractory steps after a spike.
        :param w_in_init: Initial input weights.
        :param w_rec_init: Initial recurrent weights.
        :param rec: Whether to include recurrent weights.
        """
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m
        self.tau_a = tau_a
        self.thr = thr
        self.dt = dt
        self.t_total = t_total
        self.alpha = np.exp(-dt / tau_m)  # decay factor alpha
        self.p = np.exp(-dt / tau_a)  # 0 < p < 1, adaptation decay constant , p > alpha (_decay)
        self.n_refractory = n_refractory
        self.time_since_last_spike = np.ones(n_rec) * n_refractory  # Initially assume refractory period has passed

        # ALIF parameters
        self.p = self.p  # 0 < p < 1, adaptation decay constant , p > alpha(_decay)
        self.beta = beta  # beta >= 0, adaptation strenght constant

        # Initialize weights
        # self.w_in = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
        self.w_in = np.ones((n_in, n_rec))
        # Generate the same initial random values for all time dimensions
        # base_weights = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1)

        # # Replicate the base weights along the third axis
        # self.w_rec = np.repeat(base_weights[:, :, np.newaxis], t_total, axis=2)
        
        # # Set the diagonal to 0 for each (n_rec, n_rec) matrix
        # for t in range(t_total):
        #     np.fill_diagonal(self.w_rec[:, :, t], 0)
        # # Set 80% of weights to zero
        # # Create a mask for 80% of weights to be set to zero (same mask for all t)
        # mask = np.random.rand(n_rec, n_rec) < 0.8  # Randomly select 80% of weights

        # # Apply the mask to all time steps (t)
        # self.w_rec[mask] = 0

        self.w_rec = np.ones((n_rec, n_rec, t_total)) * 0.001 

        # State variables
        self.v = np.zeros((n_rec, t_total))  # Initial membrane potential
        self.a = np.zeros((n_rec, t_total))  # threshold adaptive variable
        self.z = np.zeros((n_rec, t_total))
        self.y = np.zeros((n_out, t_total))
        
        # Output neuron weights and biases
        # np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)  # Example output weights
        self.w_out = np.ones((n_rec, n_out)) * 0.001
        self.B = np.transpose(self.w_out)
        self.b_out = np.zeros(n_out)  # Example output biases

        # Readout parameters
        self.k = 0.9  # Decay factor for output neurons
        self.y = np.zeros((n_out, t_total))  # Previous output values

        # Eligibility traces
        self.e_v = np.zeros((n_rec, t_total))
        self.e_a = np.zeros((n_rec, t_total))
        self.eligibility_traces = np.zeros((n_rec, t_total))
        self.filtered_traces = np.zeros((n_rec, t_total))
        self.gamma = 0.3

        # Learning signal 
        self.L = np.zeros((n_rec, t_total))
        self.lr = 0.01
        

    def update_state(self, x, t):
        """
        Updates the neuron's membrane potential and spikes based on input current.
        :param x: input to neuron.
        :return: Tuple (voltage, spike) with updated membrane potential and spike output.
        """

        # Membrane potential update
        self.v[:, t + 1] = self.alpha * self.v[:, t] + np.dot(x, self.w_in) + np.dot(self.z[:, t], self.w_rec[:, :, t]) # Decay, recurrent weights and input weights
        self.v[self.z[:, t] == 1, t + 1] -= self.thr # Reset potential after spike

        # Adaptive threshold
        self.a[:, t + 1] = self.p * self.a[:, t] + self.z[:, t]

        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            return # No spike during refractory period

        # Check for spike
        self.z[:, t + 1] = self.v[:, t] >= (self.thr + self.beta * self.a[:, t])   # 1 if spike, else 0
        self.time_since_last_spike[self.z[:, t + 1] == 1] = 0 # Reset refractory time count

        # Compute the output using the previous output and recurrent activity
        self.y[:, t + 1] = self.k * self.y[:, t] + np.dot(self.z[:, t], self.w_out) + self.b_out

        return
    
    def pseudo_derivative(self, t):
        return (1 / self.thr) * self.gamma * np.maximum(np.zeros(self.n_rec), 1 - np.abs((self.v[:, t] - self.thr - self.beta * self.a[:, t]) / self.thr))
    
    def update_eligibility_trace(self, t):
        pd = self.pseudo_derivative(t)
        self.e_v[:, t] = self.alpha * self.e_v[:, t - 1] + self.z[:, t - 2]
        self.e_a[:, t] = pd * self.e_v[:, t - 1] + (self.p - pd * self.beta) * self.e_a[:, t - 1]
        self.eligibility_traces[:, t] = pd * (self.e_v[:, t] - self.beta * self.e_a[:, t])
        self.filtered_traces[:, t] = (1 - self.k) * self.eligibility_traces[:, t - 1] + self.filtered_traces[:, t]
    
    def update_learning_signal(self, target, t):
        predicted_prob = np.exp(self.y[:, t]) / np.sum(np.exp(self.y[:, t]))
        self.L[:, t] = np.dot(predicted_prob - target, self.B)

    def weight_update(self, t):
        self.w_rec[:, :, t] -= self.lr * self.L[:, t] * self.filtered_traces[:, t]

n_in = 13  # Number of input neurons
n_rec = 100  # Number of recurrent neurons
n_out = 39  # Number of output neurons, one for each class of the TIMIT dataset


X_train_data = pickle.load(open('mfccs_train.pickle', 'rb'))
X_train_data = X_train_data[:1000]
n_samples = len(X_train_data)

y_train_data = pickle.load(open('phonems_train.pickle', 'rb'))
y_train_data = y_train_data[:1000]

mapping = json.load(open(path + 'reduced_phn_index_mapping.json'))

# Initialize the LIF neuron model
network = ALIFNeuron(n_in=n_in, n_rec=n_rec, n_out=n_out, t_total=n_samples, tau_m=20., tau_a=500., thr=50., dt=1., n_refractory=5., beta=0.07)

# Initialize an empty array for one-hot encoding
one_hot_encoded_target = np.zeros((n_samples, n_out))

# Training
t = 0
for X_train_trial, y_train_trial in zip(X_train_data, y_train_data):
    
    if t < n_samples - 1 :
        one_hot_encoded_target[t, mapping[y_train_trial]] = 1
        network.update_state(X_train_trial, t)  # Update neuron with input

        network.update_eligibility_trace(t)
        network.update_learning_signal(one_hot_encoded_target[t, :], t)

        network.weight_update(t)

        t += 1

# Testing
X_test_data = pickle.load(open('mfccs_test.pickle', 'rb'))
X_test_data = X_test_data[:100]
n_samples = len(X_test_data)

y_test_data = pickle.load(open('phonems_test.pickle', 'rb'))
y_test_data = y_test_data[:100]

# for X_test_trial, y_test_trial in zip(X_test_data, y_test_data):
    

# Example plotting of the results (if needed)
# Create the figure and subplots
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 7), constrained_layout=True)

idx = 5


input = X_train_data

x = range(len(input))

v = network.v[idx, :len(input)]
s = network.z[idx, :len(input)]
a = network.a[idx, :len(input)]
e = network.eligibility_traces[idx, :len(input)]
w = network.w_rec[0, idx, :len(input)]

# Plot data in each subplot
axs[0].plot(x, input[:, idx])
axs[0].set_title("Input")
axs[0].set_xlabel("t")

axs[1].plot(x, v, color='blue')
axs[1].set_title("Voltage")
axs[1].set_xlabel("t")
axs[1].set_ylabel("v")

axs[2].plot(x, a, color='red')
axs[2].set_title("Adaptive Threshold")
axs[2].set_xlabel("a")

axs[3].plot(x, s, color='red')
axs[3].set_title("Spikes")
axs[3].set_xlabel("t")

axs[4].plot(x, e, color='blue')
axs[4].set_title("Eligibility trace")
axs[4].set_xlabel("t")

axs[5].plot(range(len(w)), w, color='blue')
axs[5].set_title("Weight")
axs[5].set_xlabel("t")

plt.show()
