import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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

        return logits
    
    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
    

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=-1)

        self.input_size = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_f = nn.Linear(self.input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_i = nn.Linear(self.input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_c = nn.Linear(self.input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        self.W_o = nn.Linear(self.input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_states=None):
        batch_size, seq_len = x.size()
        x = self.embedding(x)

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_t) + self.b_f)
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_t) + self.b_i)
            c_hat_t = torch.tanh(self.W_c(x_t) + self.U_c(h_t) + self.b_c)
            c_t = f_t * c_t + i_t * c_hat_t
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_t) + self.b_o)
            h_t = o_t * torch.tanh(c_t)

        logits = self.fc(h_t)
        return logits
    
    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
