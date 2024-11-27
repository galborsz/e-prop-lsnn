import numpy as np
import matplotlib.pyplot as plt
import pickle

path = '/Users/giselaalbors/Desktop/university/master/semester 5/e-prop-lsnn/timit_processed/train/'


class ALIFNeuron:
    def __init__(self, n_in, n_rec, n_out, t_total, tau_m=20., tau_a=20., thr=0.615, dt=1., n_refractory=2., beta=0.07):
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
        self.w_rec = np.ones((n_rec, n_rec)) * 0.001  # np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1)
        np.fill_diagonal(self.w_rec, 0)  # remove self loops
        # # Set 80% of weights to zero
        # mask = np.random.rand(*self.w_rec.shape) < 0.8  # Randomly select 80% of weights
        # self.w_rec[mask] = 0  # Set selected weights to zero

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
        self.lr = 0.001
        

    def update_state(self, x, t):
        """
        Updates the neuron's membrane potential and spikes based on input current.
        :param x: input to neuron.
        :return: Tuple (voltage, spike) with updated membrane potential and spike output.
        """

        # Membrane potential update
        self.v[:, t + 1] = self.alpha * self.v[:, t] + np.dot(x, self.w_in) + np.dot(self.z[:, t], self.w_rec) # Decay, recurrent weights and input weights
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
        self.e_v[:, t + 1] = self.alpha * self.e_v[:, t] + self.z[:, t - 1]
        self.e_a[:, t + 1] = pd * self.e_v[:, t] + (self.p - pd * self.beta) * self.e_a[:, t]
        self.eligibility_traces[:, t] = pd * (self.e_v[:, t] - self.beta * self.e_a[:, t])
    
    def update_learning_signal(self, target, t):
        predicted_prob = np.exp(self.y[:, t]) / np.sum(np.exp(self.y[:, t]))
        self.L[:, t] = np.dot(predicted_prob - target, self.B)

    def weight_update(self, target, t):

        network.update_eligibility_trace(t)
        network.update_learning_signal(target, t)

        self.filtered_traces[:, t] = (1 - self.k) * self.eligibility_traces[:, t - 1] + self.filtered_traces[:, t]
        self.w_rec -= self.lr * np.dot(self.L[:, t], self.filtered_traces[:, t])

n_in = 13  # Number of input neurons
n_rec = 100  # Number of recurrent neurons
n_out = 39  # Number of output neurons, one for each class of the TIMIT dataset

# Simulate for each input x^t
X_train_data = pickle.load(open(path + 'mfccs.pickle', 'rb'))
X_train_data = X_train_data[0]
target_train_data = pickle.load(open(path + 'phonems.pickle', 'rb'))
target_train_data = target_train_data[0]

# Initialize an empty array for one-hot encoding
one_hot_encoded_target = np.zeros((X_train_data.shape[0], n_out))

# Set the corresponding index of each element in input_array to 1
for i in range(X_train_data.shape[0]):
    one_hot_encoded_target[i, target_train_data[i]] = 1

# Find the min and max values for each feature (column)
min_values = np.min(X_train_data, axis=0)  # Minimum value for each feature
max_values = np.max(X_train_data, axis=0)  # Maximum value for each feature

# Apply min-max normalization to scale the features to the range [0, 1]
normalized_X_train_data = (X_train_data - min_values) / (max_values - min_values)

n_samples = len(normalized_X_train_data)

# Initialize the LIF neuron model
network = ALIFNeuron(n_in=n_in, n_rec=n_rec, n_out=n_out, t_total=n_samples, tau_m=20., tau_a=500., thr=50., dt=1., n_refractory=5., beta=0.07)

# for epoch in range(20):
for t in range(n_samples - 1):  
    # Run the simulation for 5 time steps (for each input x^t)
    # for _ in range(5):
    network.update_state(normalized_X_train_data[t], t)  # Update neuron with input

    if t >= 1:
        network.weight_update(one_hot_encoded_target[t, :], t)

# Example plotting of the results (if needed)
# Create the figure and subplots
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(10, 7), constrained_layout=True)

x = range(n_samples)
idx = 99
input = normalized_X_train_data[:,0]
v = network.v[idx, :]
s = network.z[idx, :]
a = network.a[idx, :]
e = network.eligibility_traces[idx, :]

# Plot data in each subplot
axs[0].plot(x, input)
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

plt.show()
