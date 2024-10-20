import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, num_epochs, device, criterion = nn.BCELoss, max_grad_norm = 8):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm

        self.train_losses = []
        self.val_losses = []
        self.train_f1s = []
        self.val_f1s = []

    @staticmethod
    def compute_f1(y_trues, y_preds):
        assert y_trues.ndim == 2
        assert y_trues.shape == y_preds.shape

        y_pred_bin = (y_preds >= 0.5).astype(int)
        return f1_score(y_trues, y_pred_bin, average='macro', zero_division=1)

    def plot_losses(self):
        clear_output()
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        axs[0].plot(range(1, len(self.train_losses) + 1), self.train_losses, label='train')
        axs[0].plot(range(1, len(self.val_losses) + 1), self.val_losses, label='val')
        axs[0].set_ylabel('loss')

        axs[1].plot(range(1, len(self.train_f1s) + 1), self.train_f1s, label='train F1')
        axs[1].plot(range(1, len(self.val_f1s) + 1), self.val_f1s, label='val F1')
        axs[1].set_ylabel('F1 score')

        for ax in axs:
            ax.set_xlabel('epoch')
            ax.legend()

        plt.show()

    def _clip_grad_norm(self):
        if self.max_grad_norm is not None:
            clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

    def training_epoch(self, tqdm_desc):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc=tqdm_desc):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()

            running_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_preds = np.vstack(all_preds)
        y_trues = np.vstack(all_labels)
        f1 = self.compute_f1(y_trues.T, y_preds.T) # maybe transpose?

        return running_loss / len(self.train_loader), f1

    @torch.no_grad()
    def validation_epoch(self, tqdm_desc):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc=tqdm_desc):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_preds = np.vstack(all_preds)
        y_trues = np.vstack(all_labels)
        f1 = self.compute_f1(y_trues.T, y_preds.T) # transpose

        return running_loss / len(self.val_loader), f1

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_f1 = self.training_epoch(
                tqdm_desc=f'Training {epoch}/{self.num_epochs}'
            )
            val_loss, val_f1 = self.validation_epoch(
                tqdm_desc=f'Validating {epoch}/{self.num_epochs}'
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_f1)
            self.val_f1s.append(val_f1)

            self.plot_losses()

            print(f'Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
