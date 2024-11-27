import numpy as np
import matplotlib.pyplot as plt


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

        print(self.alpha, self.p)

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
        self.z = np.zeros((n_rec, t_total))
        self.a = np.zeros((n_rec, t_total))  # threshold adaptive variable
        
        # Output neuron weights and biases
        # np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)  # Example output weights
        self.w_out = np.ones((n_rec, n_out)) * 0.001
        self.b_out = np.zeros(n_out)  # Example output biases

        # Readout parameters
        self.k = 0.9  # Decay factor for output neurons
        self.y = np.zeros(n_out)  # Previous output values

        # Eligibility traces
        self.elig_trace = np.zeros(n_rec)

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

        return
    
    def pseudo_derivative(self):
        return self.gamma * np.max(0, 1 - np.abs((self.v - self.thr - self.beta * self.a) / self.thr))
    
    def update_eligibility_trace(self):
        self.e_a = self.pseudo_derivative() * self.e_v + (self.p - self.pseudo_derivative() * self.beta) * self.e_a
        self.e_v = self.alpha * self.e_v + self.z # has to be z^{t-1}
        
        self.elig_trace = self.pseudo_derivative() * (self.e_v - self.beta * self.e_a)
        return self.elig_trace
    
    def readout(self):
        """
        Computes the readout from the recurrent network.
        :return: The output of the readout neurons at the current time step.
        """
        # Compute the output using the previous output and recurrent activity
        self.y = self.k * self.y + np.dot(self.z, self.w_out) + self.b_out
        return self.y

    def predicted_probability(self):
        return np.exp(self.y) / (np.sum(np.exp(self.y)))


# def compute_error(self, target):
#     return - np.sum(target * np.log(predicted_probability()))


# Initialize the LIF neuron model
n_in = 13  # Number of input neurons
n_rec = 100  # Number of recurrent neurons
n_out = 61  # Number of output neurons, one for each class of the TIMIT dataset
n_samples = 550
network = ALIFNeuron(n_in=n_in, n_rec=n_rec, n_out=n_out, tau_m=20., tau_a=500., thr=1.6, dt=1., n_refractory=5., beta=0.07)

# Input array with 100 time steps (aka number of input samples) with 13 features (aka input neurons) each
x = np.linspace(0, 50, n_samples)  # Create 150 points over one period
single_wave = 0.001 * np.sin((x + 0.2) / 2) + 0.01
input_currents = np.tile(single_wave, (n_in, 1)).T

# To store the output
outputs = []

# Simulate for each input x^t and 
# for epoch in range(80):
for t in range(n_samples - 1):  
    # Run the simulation for 5 time steps (for each input x^t)
    # for _ in range(5):
    network.update_state(input_currents[t], t)  # Update neuron with input

voltages = network.v
spikes = network.z
adaptive = network.a

# # After all time steps, compute the readout (output y^t) at the end
# final_output = neuron.readout()  # Compute the readout at the last time step
# outputs.append(final_output)  # Store the output

# # Convert outputs to numpy array for easier manipulation
# outputs = np.array(outputs)

# print("Final Outputs (y^t) for all inputs (x^t):")
# print(outputs)

# Example plotting of the results (if needed)
# Create the figure and subplots
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 7), constrained_layout=True)

x = range(n_samples)
idx = 99
input = input_currents[:,0]
v = np.array(voltages)[idx, :]
s = np.array(spikes)[idx, :]
a = np.array(adaptive)[idx, :]
# Plot data in each subplot
axs[0].plot(x, input, color='red')
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

plt.show()
