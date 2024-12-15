import numpy as np

np.random.seed(123)

class ALIFNetwork():
    def __init__(self, n_in, n_rec, n_out, tau_m, tau_a, tau_out, thr, dt, n_refractory, beta):
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
        self.p = np.exp(-dt / tau_a)  # 0 < p < 1, adaptation decay constant , p > alpha
        self.beta = beta  # beta >= 0, adaptation strenght constant

        # Initialize weights
        self.w_in = np.random.normal(0, np.sqrt(n_in), (n_in, n_rec)) #np.random.randn(n_in, n_rec) / np.sqrt(n_in) # np.random.normal(0.1, 0.01, size=(n_in, n_rec)) # np.ones((n_in, n_rec))
        self.w_rec = np.random.normal(0, np.sqrt(n_rec - 1), (n_rec, n_rec)) # np.random.randn(n_rec, n_rec) / np.sqrt(n_rec - 1) # np.random.normal(0.001, 0.0001, size=(n_rec, n_rec)) # np.ones((n_rec, n_rec)) * 0.001 
        np.fill_diagonal(self.w_rec, 0)  # remove self loops

        # State variables
        self.v = np.zeros(n_rec)  # membrane potential
        self.a = np.zeros(n_rec)  # adaptive threshold
        self.z = np.zeros(n_rec)
        
        # Output neuron weights and biases
        self.w_out = np.random.normal(0, np.sqrt(n_rec), (n_rec, n_out)) # np.random.randn(n_rec, n_out) / np.sqrt(n_rec) # np.random.normal(1, 0.5, size=(n_rec, n_out)) # np.ones((n_rec, n_out)) * 0.001 
        self.b_out = np.zeros(n_out)

        # Readout parameters
        self.k = np.exp(-dt / tau_out)  # Decay factor for output neurons
        self.y = np.zeros(n_out)  # output

        # Eligibility traces
        self.e_v = np.zeros((n_rec, n_rec))
        self.e_a = np.zeros((n_rec, n_rec))
        self.trace_rec = np.zeros((n_rec, n_rec))
        self.filtered_trace_rec = np.zeros((n_rec, n_rec))
        self.trace_out = np.zeros((n_rec, n_out))
        self.gamma = 0.3
        self.e_v_in = np.zeros((n_in, n_rec))
        self.e_a_in = np.zeros((n_in, n_rec))
        self.trace_in = np.zeros((n_in, n_rec))
        self.filtered_trace_in = np.zeros((n_in, n_rec))

        # Learning signal 
        self.L = np.zeros(n_rec)
        self.lr = 0.001

        # Weight gradients
        self.w_in_grad = np.zeros((n_in, n_rec))
        self.w_rec_grad = np.zeros((n_rec, n_rec))
        self.w_out_grad = np.zeros((n_rec, n_out))
        self.b_out_grad = np.zeros(n_out)

    def reset_gradients(self):
        self.w_in_grad = np.zeros((self.n_in, self.n_rec))
        self.w_rec_grad = np.zeros((self.n_rec, self.n_rec))
        self.w_out_grad = np.zeros((self.n_rec, self.n_out))
        self.b_out_grad = np.zeros(self.n_out)

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
            self.y = self.k * self.y + np.dot(self.z, self.w_out) + self.b_out
            output = self.softmax(self.y)
            outputs.append(output)

            if training:
                self.compute_grads(one_hot_encoded_target)

        return outputs, voltage, spikes, adaptation

    def update_state(self, x):
        # Membrane potential update
        self.v = self.alpha * self.v + np.dot(x, self.w_in) + np.dot(self.z, self.w_rec) # Decay, recurrent weights and input weights
        self.v[self.z == 1] -= self.thr # Reset potential after spike

        # Adaptive threshold
        self.a = self.p * self.a + self.z

        if np.any(self.time_since_last_spike < self.n_refractory):
            self.time_since_last_spike[self.time_since_last_spike < self.n_refractory] += self.dt
            self.z = self.time_since_last_spike > self.n_refractory
            return # No spike during refractory period

        # Check for spike
        self.z = self.v >= (self.thr + self.beta * self.a)   # 1 if spike, else 0
        self.time_since_last_spike[self.z == 1] = 0 # Reset refractory time count


    def compute_grads(self, target):  
        self.e_v = self.alpha * self.e_v + self.z[:, None]
        pd = (1 / self.thr) * self.gamma * np.maximum(0, 1 - np.abs((self.v - self.thr - self.beta * self.a) / self.thr))
        self.e_a = pd * self.e_v + (self.p - pd * self.beta) * self.e_a
        self.trace_rec = pd * (self.e_v - self.beta * self.e_a)
        self.filtered_trace_rec = self.alpha * self.filtered_trace_rec + self.trace_rec

        self.trace_out = self.alpha * self.trace_out + self.z[:, None]

        self.e_v_in = self.alpha * self.e_v_in + self.z[None, :]
        pd = (1 / self.thr) * self.gamma * np.maximum(0, 1 - np.abs((self.v - self.thr - self.beta * self.a) / self.thr))
        self.e_a_in = pd * self.e_v_in + (self.p - pd * self.beta) * self.e_a_in
        self.trace_in = pd * (self.e_v_in - self.beta * self.e_a_in)
        self.filtered_trace_in = self.alpha * self.filtered_trace_in + self.trace_in

        predicted_prob = self.softmax(self.y)
        B = np.transpose(self.w_out) # symmetric e-prop
        self.L = np.dot(predicted_prob - target, B)
        
        self.w_rec_grad += self.L * self.filtered_trace_rec
        self.w_in_grad += self.L * self.filtered_trace_in
        self.w_out_grad += (predicted_prob - target) * self.trace_out
        self.b_out_grad += (predicted_prob - target)

    
    def weight_update(self):
        self.w_in -= self.lr * self.w_in_grad
        self.w_rec -= self.lr * self.w_rec_grad
        np.fill_diagonal(self.w_rec, 0)
        self.w_out -= self.lr * self.w_out_grad
        self.b_out -= self.lr * self.b_out_grad
        return

    def softmax(self, values):
        return np.exp(values)/np.sum(np.exp(values))
    
    def cross_entropy(self, ypred, ytrue):

        ypred_prob = self.softmax(ypred)
        loss = 0
        for i in range(len(ypred_prob)):
            loss = loss + ytrue[i]*np.log(ypred_prob[i])

        return -loss