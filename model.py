import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(123)

class ALIFNetwork(nn.Module):
    def __init__(self, n_in, n_rec, n_out, tau_m, tau_a, tau_out, thr, dt, n_refractory, beta):
        super(ALIFNetwork, self).__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m
        self.tau_a = tau_a
        self.tau_out = tau_out
        self.thr = thr
        self.dt = dt
        self.alpha = np.exp(-dt / tau_m)  # decay factor alpha
        self.n_refractory = n_refractory
        self.time_since_last_spike = np.ones(n_rec) * n_refractory  # Initially assume refractory period has passed

        # ALIF parameters
        self.alpha = np.exp(-dt / tau_a)  # 0 < p < 1, adaptation decay constant , p > alpha
        self.beta = beta  # beta >= 0, adaptation strenght constant

        # # Initialize weights
        self.w_in  = nn.Parameter(torch.Tensor(n_in, n_rec))
        self.w_in.data = torch.rand(n_in, n_rec) / np.sqrt(n_in) 

        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_rec.data = torch.rand(n_rec, n_rec) / np.sqrt(n_rec) 
        self.w_rec.data.fill_diagonal_(0) # remove self loops

        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.w_out.data = torch.rand(n_rec, n_out) / np.sqrt(n_rec)

        self.B = torch.rand(n_out, n_rec)

        self.b_out = nn.Parameter(torch.Tensor(n_out))
        self.b_out.data = torch.zeros_like(torch.Tensor(n_out))

        # State variables
        self.v = torch.zeros(n_rec)  # membrane potential
        self.a = torch.zeros(n_rec)  # adaptive threshold
        self.z = torch.zeros(n_rec)

        # Readout parameters
        self.k = np.exp(-dt / tau_out)  # Decay factor for output neurons
        self.y = torch.zeros(n_out)  # output

        # Eligibility traces
        self.e_v = torch.zeros((n_rec, n_rec))
        self.e_a = torch.zeros((n_rec, n_rec))
        self.trace_rec = torch.zeros((n_rec, n_rec))
        self.filtered_trace_rec = torch.zeros((n_rec, n_rec))
        self.trace_out = torch.zeros((n_rec, n_out))
        self.e_v_in = torch.zeros((n_in, n_rec))
        self.e_a_in = torch.zeros((n_in, n_rec))
        self.trace_in = torch.zeros((n_in, n_rec))
        self.filtered_trace_in = torch.zeros((n_in, n_rec))
        self.gamma = 0.3

        # Learning signal 
        self.L = torch.zeros(n_rec)
        self.lr = 0.01

    def reset_gradients(self):
        self.w_in.grad  = torch.zeros((self.n_in, self.n_rec))
        self.w_rec.grad = torch.zeros((self.n_rec, self.n_rec))
        self.w_out.grad = torch.zeros((self.n_rec, self.n_out))
        self.b_out.grad = torch.zeros(self.n_out)

    def forward(self, x, target, training):
        # Initialize an empty array for one-hot encoding
        one_hot_encoded_target = np.zeros(self.n_out)
        t_total = x.shape[0]
        outputs = []
        voltage = []
        spikes = []
        adaptation = []    

        # Simulate for each sample x^t
        for t in range(t_total):
            one_hot_encoded_target[target[t]] = 1
            self.update_state(x[t])
            voltage.append(self.v)
            spikes.append(self.z)
            adaptation.append(self.a)

            # Compute output
            self.y = self.k * self.y + torch.matmul(self.z, self.w_out.data.to(torch.float64)) + self.b_out.data
            output = F.softmax(self.y, dim=0)
            outputs.append(output)

            if training:
                self.compute_grads(x[t], torch.from_numpy(one_hot_encoded_target))

        return outputs, voltage, spikes, adaptation

    def update_state(self, x):
        # Membrane potential update
        self.v = self.alpha * self.v + torch.matmul(x.to(torch.float64), self.w_in.data.to(torch.float64)) + torch.matmul(self.z.to(torch.float64), self.w_rec.data.to(torch.float64)) # Decay, recurrent weights and input weights
        self.v[self.z == 1] -= self.thr # Reset potential after spike

        # Adaptive threshold
        self.a = self.alpha * self.a + self.z

        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            self.z = self.time_since_last_spike > self.n_refractory
            self.z = torch.from_numpy(self.z).to(torch.float64)
            return # No spike during refractory period

        # Check for spike
        self.z = self.v >= (self.thr + self.beta * self.a)   # 1 if spike, else 0
        self.z = self.z.to(torch.float64)
        self.time_since_last_spike[self.z == 1] = 0 # Reset refractory time count


    def compute_grads(self, x, target):  
        # Eligibility traces
        pd = (1 / self.thr) * self.gamma * np.maximum(0, 1 - np.abs((self.v - self.thr - self.beta * self.a) / self.thr))

        self.e_v = self.alpha * self.e_v + self.z[:, None]
        self.e_a = pd * self.e_v + (self.alpha - pd * self.beta) * self.e_a
        self.trace_rec = pd * (self.e_v - self.beta * self.e_a)
        self.filtered_trace_rec = self.alpha * self.filtered_trace_rec + self.trace_rec

        self.trace_out = self.alpha * self.trace_out + self.z[:, None]

        self.e_v_in = self.alpha * self.e_v_in + x[:, None]
        self.e_a_in = pd * self.e_v_in + (self.alpha - pd * self.beta) * self.e_a_in
        self.trace_in = pd * (self.e_v_in - self.beta * self.e_a_in)
        self.filtered_trace_in = self.alpha * self.filtered_trace_in + self.trace_in

        # Learning signals
        predicted_prob = F.softmax(self.y, dim=0)
        error = predicted_prob - target
        self.L = torch.matmul(error.to(torch.float64), self.B.to(torch.float64))

        self.w_rec.grad += self.lr * self.L * self.filtered_trace_rec
        self.w_in.grad += self.lr * self.L * self.filtered_trace_in
        self.w_out.grad += self.lr * (predicted_prob - target) * self.trace_out
        self.b_out.grad += self.lr * (predicted_prob - target)
