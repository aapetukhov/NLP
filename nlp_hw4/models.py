import torch
import torch.nn as nn
from transformers import BertForTokenClassification


class FactorizedEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        factor_size: int,
        source_embeds: torch.Tensor
    ) -> None:
        super(FactorizedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.factor_size = factor_size

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(source_embeds, full_matrices=False)
            U = U[:, :factor_size]
            S = S[:factor_size]
            Vh = Vh[:factor_size, :]

            self.embeddings_VE = nn.Embedding(num_embeddings, factor_size)
            self.linear_EH = nn.Linear(factor_size, embedding_dim, bias=False)

            self.embeddings_VE.weight.copy_(U)
            Vh_T = Vh.T * S.unsqueeze(0)
            self.linear_EH.weight.copy_(Vh_T)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings_E = self.embeddings_VE(input_ids)
        embeddings_H = self.linear_EH(embeddings_E)
        return embeddings_H
    

class SharedAttentionBert(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.shared_attention = self.bert.encoder.layer[0].attention
        for i in range(1, len(self.bert.encoder.layer)):
            self.bert.encoder.layer[i].attention = self.shared_attention
        self.param_savings = self.calculate_param_savings()

    def calculate_param_savings(self):
        param_per_attention_layer = sum(p.numel() for p in self.shared_attention.parameters())
        num_layers = len(self.bert.encoder.layer)
        param_savings = (num_layers - 1) * param_per_attention_layer
        
        return param_savings
