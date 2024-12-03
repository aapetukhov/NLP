import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import Trainer

from models import FactorizedBertIntermediate, FactorizedBertOutput


class DistilTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.teacher_model.eval()
        outputs = model(**inputs)
        logits_student = outputs.logits
        
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        softmax_teacher = F.softmax(logits_teacher / self.temperature, dim=-1)
        # distil loss
        soft_loss = F.kl_div(F.log_softmax(logits_student / self.temperature, dim=-1), softmax_teacher, reduction="batchmean") * (self.temperature ** 2)
        # student loss
        labels = inputs['labels']
        hard_loss = F.cross_entropy(logits_student.view(-1, logits_student.size(-1)), labels.view(-1), reduction="mean")

        total_loss = soft_loss + hard_loss
        return (total_loss, outputs) if return_outputs else total_loss



def get_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    return optimizer


def get_scheduler(optimizer, num_epochs, steps_per_epoch):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-5,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    return scheduler

def replace_with_factorized_layers(model: nn.Module, k: int):
    for layer in model.bert.encoder.layer:
        intermediate_dense = layer.intermediate.dense
        d_model = intermediate_dense.in_features
        d_ff = intermediate_dense.out_features
        layer.intermediate.dense = FactorizedBertIntermediate(d_model, d_ff, k)

        output_dense = layer.output.dense
        d_ff = output_dense.in_features
        d_model = output_dense.out_features
        layer.output.dense = FactorizedBertOutput(d_ff, d_model, k)

def hook_fn(layer_name):
    def hook(module, inputs, outputs):
        global captured_inputs, captured_outputs
        captured_inputs[layer_name] = inputs[0].detach()
        captured_outputs[layer_name] = outputs.detach()
    return hook
