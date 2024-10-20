import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Baseline, self).__init__()
        self.hidden_size = hidden_size
        
        self.W = nn.Linear(input_size, hidden_size)  # W * x_t
        self.U = nn.Linear(hidden_size, hidden_size) # U * h_(t-1)
        self.tanh = nn.Tanh()

        self.V = nn.Linear(hidden_size, output_size)  # V * h_T
        
    def forward(self, x):
        # (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.tanh(self.W(x_t) + self.U(h_t))
        
        o_T = self.V(h_t)
        return nn.Sigmoid(o_T)