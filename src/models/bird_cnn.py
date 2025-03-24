import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score

class BirdCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 128, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.num_classes = num_classes
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        probs = F.softmax(logits, dim=1).detach().cpu()
        y_true = y.detach().cpu()

        self.validation_step_outputs.append((probs, y_true))
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x[0] for x in self.validation_step_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_step_outputs], dim=0).numpy()
        self.validation_step_outputs.clear()

        # One-hot encode true labels
        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.num_classes).numpy()

        aucs = []
        for i in range(self.num_classes):
            if y_true_oh[:, i].sum() == 0:
                continue  # skip class with no positive samples
            auc = roc_auc_score(y_true_oh[:, i], all_probs[:, i])
            aucs.append(auc)

        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"📈 val_macro_auc = {macro_auc:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)