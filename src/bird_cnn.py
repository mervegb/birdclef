import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score
from utils import mixup_data

class BirdCNN(pl.LightningModule):
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

        self.validation_step_outputs = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
        logits = self(x)
        loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
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

        probs = F.softmax(logits, dim=1).detach().cpu()
        y_true = y.detach().cpu()
        self.validation_step_outputs.append((probs, y_true))
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x[0] for x in self.validation_step_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_step_outputs], dim=0).numpy()
        self.validation_step_outputs.clear()

        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.num_classes).numpy()
        aucs = []
        for i in range(self.num_classes):
            if y_true_oh[:, i].sum() == 0:
                continue
            auc = roc_auc_score(y_true_oh[:, i], all_probs[:, i])
            aucs.append(auc)

        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"📈 val_macro_auc = {macro_auc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)  # back to default
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
            "interval": "epoch"
        }
        return [optimizer], [scheduler]