import numpy as np
import json

path = '/Users/giselaalbors/Desktop/university/master/semester 5/e-prop-lsnn/timit_processed/train/'
mapping = json.load(open(path + 'reduced_phn_index_mapping.json'))

class ALIFNetwork():
    def __init__(self, n_in, n_rec, n_out, tau_m, tau_a, thr, dt, n_refractory, beta):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m
        self.tau_a = tau_a
        self.thr = thr
        self.dt = dt
        self.alpha = np.exp(-dt / tau_m)  # decay factor alpha
        self.p = np.exp(-dt / tau_a)  # 0 < p < 1, adaptation decay constant , p > alpha (_decay)
        self.n_refractory = n_refractory
        self.time_since_last_spike = np.ones(n_rec) * n_refractory  # Initially assume refractory period has passed

        # ALIF parameters
        self.p = self.p  # 0 < p < 1, adaptation decay constant , p > alpha(_decay)
        self.beta = beta  # beta >= 0, adaptation strenght constant

        # Initialize weights
        self.w_in = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
        self.w_rec = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1)
        np.fill_diagonal(self.w_rec, 0)  # remove self loops
        # Set 80% of weights to zero
        mask = np.random.rand(*self.w_rec.shape) < 0.8  # Randomly select 80% of weights
        self.w_rec[mask] = 0  # Set selected weights to zero

        # State variables
        self.v = np.zeros(n_rec)  # Initial membrane potential
        self.a = np.zeros(n_rec)  # threshold adaptive variable
        self.z = np.zeros(n_rec)
        self.y = np.zeros(n_out)
        
        # Output neuron weights and biases
        self.w_out = np.random.randn(n_rec, n_out) / np.sqrt(n_rec)
        self.B = np.transpose(self.w_out)
        self.b_out = np.zeros(n_out)  # Example output biases

        # Readout parameters
        self.k = 0.9  # Decay factor for output neurons
        self.y = np.zeros(n_out)  # Previous output values

        # Eligibility traces
        self.e_v = np.zeros((n_rec, n_rec))
        self.e_a = np.zeros((n_rec, n_rec))
        self.eligibility_traces = np.zeros((n_rec, n_rec))
        self.filtered_traces = np.zeros((n_rec, n_rec))
        self.gamma = 0.3

        # Learning signal 
        self.L = np.zeros(n_rec)
        self.lr = 0.01

    def forward(self, x, target):
        # Initialize an empty array for one-hot encoding
        one_hot_encoded_target = np.zeros(self.n_out)
        t_total = x.shape[0]
        outputs = []
        voltage = []
        spikes = []
        adaptation = []
        etrace = []
        weights_rec = []        

        # Simulate for each input x^t
        for t in range(t_total):
            one_hot_encoded_target[mapping[target[t]]] = 1
            self.update_state(x[t])
            voltage.append(self.v)
            spikes.append(self.z)
            adaptation.append(self.a)

            # Compute output
            self.y = self.k * self.y + np.dot(self.z, self.w_out) + self.b_out
            outputs.append(self.y)

            self.compute_grads(one_hot_encoded_target)
            etrace.append(self.eligibility_traces)
            weights_rec.append(self.w_rec)

        return outputs, voltage, spikes, adaptation, etrace, weights_rec

    def update_state(self, x):
        # Membrane potential update
        self.v = self.alpha * self.v + np.dot(x, self.w_in) + np.dot(self.z, self.w_rec) # Decay, recurrent weights and input weights
        self.v[self.z == 1] -= self.thr # Reset potential after spike

        # Adaptive threshold
        self.a = self.p * self.a + self.z

        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            return # No spike during refractory period

        # Check for spike
        self.z = self.v >= (self.thr + self.beta * self.a)   # 1 if spike, else 0
        self.time_since_last_spike[self.z == 1] = 0 # Reset refractory time count

    def compute_grads(self, target):  
        for i in range(self.n_rec):
            for j in range(self.n_rec):
                pd = (1 / self.thr) * self.gamma * np.maximum(0, 1 - np.abs((self.v[j] - self.thr - self.beta * self.a[j]) / self.thr))
                self.e_v[i,j] = self.alpha * self.e_v[i,j] + self.z[i]
                self.e_a[i,j] = pd * self.e_v[i,j] + (self.p - pd * self.beta) * self.e_a[i,j]
    
                self.eligibility_traces[i,j] = pd * (self.e_v[i,j] - self.beta * self.e_a[i,j])
                self.filtered_traces[i,j] = (1 - self.k) * self.eligibility_traces[i,j] + self.filtered_traces[i,j]

        predicted_prob = np.exp(self.y) / np.sum(np.exp(self.y))
        self.L = np.dot(predicted_prob - target, self.B)

        self.w_rec -= self.lr * self.L * self.filtered_traces
    
    def cross_entropy(self, ypred, ytrue):

        ypred_prob = np.exp(ypred)/np.sum(np.exp(ypred))
        
        loss = 0
        for i in range(len(ypred_prob)):
            loss = loss + (-1 * ytrue[i]*np.log(ypred_prob[i]))

        return loss