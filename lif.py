import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, n_in, n_rec, n_out, tau=20., thr=0.615, dt=1., n_refractory=1):
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
        self.tau = tau
        self.thr = thr
        self.dt = dt
        self._decay = np.exp(-dt / tau)
        self.n_refractory = n_refractory
        self.time_since_last_spike = np.ones(n_rec) * n_refractory  # Initially assume refractory period has passed

        # Initialize weights
        # self.w_in = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
        self.w_in = np.ones((n_in, n_rec))
        self.w_rec = np.ones((n_rec, n_rec)) * 0.001 #np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1)
        np.fill_diagonal(self.w_rec, 0) # remove self loops
        # # Set 80% of weights to zero
        # mask = np.random.rand(*self.w_rec.shape) < 0.8  # Randomly select 80% of weights
        # self.w_rec[mask] = 0  # Set selected weights to zero

        # State variables
        self.v = np.zeros(n_rec)  # Initial membrane potential
        self.z = np.zeros(n_rec)
        
        # Output neuron weights and biases
        self.w_out = np.ones((n_rec, n_out)) * 0.001 # np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)  # Example output weights
        self.b_out = np.zeros(n_out)  # Example output biases
        
        # Readout parameters
        self.k = 0.9  # Decay factor for output neurons
        self.y = np.zeros(n_out)  # Previous output values

    def update(self, x):
        """
        Updates the neuron's membrane potential and spikes based on input current.
        :param x: input to neuron.
        :return: Tuple (voltage, spike) with updated membrane potential and spike output.
        """

        # Membrane potential update
        self.v = self._decay * self.v + np.dot(x, self.w_in) + np.dot(self.z, self.w_rec) # Decay, recurrent weights and input weights
        self.v[self.z == 1] -= self.thr # Reset potential after spike

        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            self.v = self.v
            self.z = np.zeros(n_rec)
            return self.v, self.z  # No spike during refractory period

        # Check for spike
        self.z = self.v >= self.thr  # 1 if spike, else 0
        self.time_since_last_spike[self.z == 1] = 0 # Reset refractory time count

        return self.v, self.z
    
    def readout(self):
        """
        Computes the readout from the recurrent network.
        :return: The output of the readout neurons at the current time step.
        """
        # Compute the output using the previous output and recurrent activity
        self.y = self.k * self.y + np.dot(self.z, self.w_out) + self.b_out
        return self.y


    def predicted_probability(self):
        return np.exp(self.y)/(np.sum(np.exp(self.y)))

# def compute_error(self, target):
#     return - np.sum(target * np.log(predicted_probability()))


# Example Usage

# Initialize the LIF neuron model
n_in = 13  # Number of input neurons
n_rec = 100  # Number of recurrent neurons
n_out = 61 # Number of output neurons, one for each class of the TIMIT dataset
n_samples = 550
network = LIFNeuron(n_in=n_in, n_rec=n_rec, n_out=n_out, tau=20., thr=1.6, dt=1., n_refractory=2.)

# Input array with 100 time steps (aka number of input samples) with 13 features (aka input neurons) each
# input_currents = np.random.rand(100, 13)
input_currents = np.zeros((n_samples, n_in))
input_currents[10:140, :] = 0.01

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
idx = 10
plt.plot(range(n_samples), np.array(voltages)[:,idx])
plt.title("Neuron Membrane Potential Over Time")
plt.xlabel("Time Step")
plt.ylabel("Voltage")
plt.show()

plt.plot(range(n_samples), np.array(spikes)[:,idx])
plt.show()
