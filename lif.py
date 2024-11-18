import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, n_in, n_rec, tau=20., thr=0.615, dt=1., n_refractory=1):
    def __init__(self, n_in, n_rec, tau=20., thr=0.615, dt=1., n_refractory=1):
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
        self.tau = tau
        self.thr = thr
        self.dt = dt
        self._decay = np.exp(-dt / tau)
        self.n_refractory = n_refractory
        self.time_since_last_spike = np.ones(n_rec) * n_refractory  # Initially assume refractory period has passed

        # Initialize weights
        self.w_in = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
        self.w_rec = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1)
        np.fill_diagonal(self.w_rec, 0) # remove self loops
        # # Set 80% of weights to zero
        # mask = np.random.rand(*self.w_rec.shape) < 0.8  # Randomly select 80% of weights
        # self.w_rec[mask] = 0  # Set selected weights to zero

        # State variables
        self.v = np.zeros(n_rec)  # Initial membrane potential
        self.z = np.zeros(n_rec)
        
        # Output neuron weights and biases
        self.w_out = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)  # Example output weights
        self.b_out = np.zeros(n_rec)  # Example output biases
        
        # Readout parameters
        self.kappa = 0.9  # Leak factor for readout neurons
        self.y_prev = np.zeros(n_rec)  # Previous output values

    def update(self, x):
        """
        Updates the neuron's membrane potential and spikes based on input current.
        :param x: input to neuron.
        :return: Tuple (voltage, spike) with updated membrane potential and spike output.
        """
        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            return self.v, np.zeros(n_rec)  # No spike during refractory period

        # Membrane potential update
        self.v = self.v + self.dt * (x @ self.w_in)  # self._decay * self.v + self.z @ self.w_rec Decay, recurrent weights and input weights
        self.v[self.z == 1] -= self.thr # Reset potential after spike
        # self.v[self.z == 1] = 0.0 # Reset potential after spike

        # Check for spike
        self.z = self.v >= self.thr  # 1 if spike, else 0
        self.time_since_last_spike[self.z == 1] = 0 
        self.time_since_last_spike[self.z == 0] += 1  # Reset refractory

        return self.v, self.z
    
    def readout(self):
        """
        Computes the readout from the recurrent network.
        :return: The output of the readout neurons at the current time step.
        """
        # Compute the output using the previous output and recurrent activity
        y = self.kappa * self.y_prev + np.dot(self.v, self.w_out) + self.b_out
        self.y_prev = y  # Update previous output for next time step
        return y

# Example Usage

# Initialize the LIF neuron model
n_in = 13  # Number of input neurons
n_rec = 100  # Number of recurrent neurons
n_out = 61 # Number of output neurons, one for each class of the TIMIT dataset
n_samples = 150
network = LIFNeuron(n_in=n_in, n_rec=n_rec, tau=20., thr=1.6, dt=1., n_refractory=2.)

# Input array with 100 time steps (aka number of input samples) with 13 features (aka input neurons) each
# input_currents = np.random.rand(100, 13)
input_currents = np.zeros((n_samples, n_in))
input_currents[30:70, :] = 0.1

# To store the output
outputs = []

voltages, spikes = [], []
# Simulate for each input x^t and 
# for epoch in range(80):
for t in range(n_samples):  
    # Run the simulation for 5 time steps (for each input x^t)
    # for _ in range(5):
    v, spike = network.update(input_currents[t])  # Update neuron with input
    voltages.append(v)
    spikes.append(spike)
    
# # After all time steps, compute the readout (output y^t) at the end
# final_output = neuron.readout()  # Compute the readout at the last time step
# outputs.append(final_output)  # Store the output

# # Convert outputs to numpy array for easier manipulation
# outputs = np.array(outputs)

# print("Final Outputs (y^t) for all inputs (x^t):")
# print(outputs)

print(np.array(voltages).shape)
# Example plotting of the results (if needed)
idx = 99
plt.plot(range(n_samples), np.array(voltages)[:,idx])
plt.title("Neuron Membrane Potential Over Time")
plt.xlabel("Time Step")
plt.ylabel("Voltage")
plt.show()

plt.plot(range(n_samples), np.array(spikes)[:,idx])
plt.show()
