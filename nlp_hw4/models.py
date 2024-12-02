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
    

class FactorizedBertIntermediate(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            k: int
    ):
        super(FactorizedBertIntermediate, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.k = k
        self.U = nn.Parameter(torch.randn(d_model, k))
        self.S = nn.Parameter(torch.randn(k, k))
        self.V = nn.Parameter(torch.randn(k, d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate_output = torch.matmul(torch.matmul(x, self.U), self.S)
        return torch.matmul(intermediate_output, self.V)


class FactorizedBertOutput(nn.Module):
    def __init__(
            self,
            d_ff: int,
            d_model: int,
            k: int
    ):
        super(FactorizedBertOutput, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.k = k
        self.U = nn.Parameter(torch.randn(d_ff, k))
        self.S = nn.Parameter(torch.randn(k, k))
        self.V = nn.Parameter(torch.randn(k, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(torch.matmul(x, self.U), self.S)
        return torch.matmul(output, self.V)


class BasePruner(nn.Module):
    def __init__(self) -> None:
        super(BasePruner, self).__init__()

    def precompute_hessinv(
        self,
        hessian_matrix: torch.Tensor,
    ) -> torch.Tensor:
        zero_diagonal = torch.diag(hessian_matrix) == 0
        hessian_matrix[zero_diagonal, zero_diagonal] = 1
        damping_value = 0.01 * torch.mean(torch.diag(hessian_matrix))
        diagonal_indices = torch.arange(hessian_matrix.shape[0], device=hessian_matrix.device)
        hessian_matrix[diagonal_indices, diagonal_indices] += damping_value
        cholesky_factor = torch.linalg.cholesky(hessian_matrix)
        hessian_inverse = torch.cholesky_inverse(cholesky_factor)
        return torch.linalg.cholesky(hessian_inverse, upper=True)

    def compute_mask(
        self,
        weights: torch.Tensor,
        hessian_inverse: torch.Tensor,
        sparsity_level: float,
        block_size: int
    ) -> torch.Tensor:
        mask_list = []
        for start_col in range(0, weights.shape[1], block_size):
            end_col = min(start_col + block_size, weights.shape[1])
            weight_block = weights[:, start_col:end_col]
            hessian_block = hessian_inverse[start_col:end_col, start_col:end_col]
            importance_scores = weight_block ** 2 / (torch.diag(hessian_block).reshape((1, -1))) ** 2
            threshold = torch.quantile(importance_scores.flatten(), sparsity_level)
            mask_list.append(importance_scores <= threshold)
        return torch.cat(mask_list, dim=1)


class Pruner(BasePruner):
    def __init__(self, layer: nn.Module):
        super(Pruner, self).__init__()
        self.layer = layer
        self.device = layer.weight.device
        weight_matrix = layer.weight.data.clone()
        self.num_rows, self.num_cols = weight_matrix.shape
        self.hessian_matrix = torch.zeros((self.num_cols, self.num_cols), device=self.device)

    def zero_weights(self, sparsity_level: float, block_size: int = 128):
        weight_matrix = self.layer.weight.data.clone().float()
        hessian_inverse = self.precompute_hessinv(self.hessian_matrix)
        pruning_mask = self.compute_mask(weight_matrix, hessian_inverse, sparsity_level, block_size)
        for start_col in range(0, weight_matrix.shape[1], block_size):
            end_col = min(start_col + block_size, weight_matrix.shape[1])
            weight_block = weight_matrix[:, start_col:end_col].clone()
            mask_block = pruning_mask[:, start_col:end_col]
            for col_idx in range(weight_block.shape[1]):
                column = weight_block[:, col_idx]
                column[mask_block[:, col_idx]] = 0
                weight_block[:, col_idx] = column
            weight_matrix[:, start_col:end_col] = weight_block
        self.layer.weight.data = weight_matrix.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
