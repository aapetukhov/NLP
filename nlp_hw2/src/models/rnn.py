import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size

        self.W = nn.Linear(embedding_dim, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.tanh(self.W(x_t) + self.U(h_t))
        logits = self.V(h_t)

        return self.sigmoid(logits)
