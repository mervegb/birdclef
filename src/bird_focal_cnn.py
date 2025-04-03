import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class BirdFocalCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        self.feature_extractor = backbone.features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.classifier[1].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.focal_loss = FocalLoss(gamma=2.0)
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.focal_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.focal_loss(logits, y)
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

        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.num_classes).numpy()
        aucs = [
            roc_auc_score(y_true_oh[:, i], all_probs[:, i])
            for i in range(self.num_classes) if y_true_oh[:, i].sum() > 0
        ]
        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"📈 val_macro_auc (Focal) = {macro_auc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
            "interval": "epoch"
        }
        return [optimizer], [scheduler]