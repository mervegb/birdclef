# bird_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from sklearn.metrics import roc_auc_score
from utils import mixup_data

class BirdCLEFModel(pl.LightningModule):
    def __init__(self, model_name="efficientnet_b0", num_classes=182, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.validation_outputs = []

        # Load EfficientNet from timm with grayscale input
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pooling(features)
        return self.head(pooled)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(logits).detach().cpu()
        y_true = y.detach().cpu()
        self.validation_outputs.append((probs, y_true))
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x[0] for x in self.validation_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_outputs], dim=0).numpy()
        self.validation_outputs.clear()

        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.num_classes).numpy()
        aucs = []
        for i in range(self.num_classes):
            if y_true_oh[:, i].sum() == 0:
                continue
            auc = roc_auc_score(y_true_oh[:, i], all_probs[:, i])
            aucs.append(auc)

        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"\U0001F4C8 val_macro_auc = {macro_auc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
