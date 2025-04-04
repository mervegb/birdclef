import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score

class BirdCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        # Load EfficientNetB0 backbone with pretrained weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Unfreeze all layers to allow fine-tuning for spectrograms
        for param in backbone.parameters():
            param.requires_grad = True
        self.feature_extractor = backbone.features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.classifier[1].in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.validation_outputs = []

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        probs = F.softmax(logits, dim=1).detach().cpu()
        targets = y.detach().cpu()
        self.validation_outputs.append((probs, targets))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        all_probs = torch.cat([x[0] for x in self.validation_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_outputs], dim=0).numpy()
        self.validation_outputs.clear()

        # Compute macro AUC per class
        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.hparams.num_classes).numpy()
        aucs = [
            roc_auc_score(y_true_oh[:, i], all_probs[:, i])
            for i in range(self.hparams.num_classes)
            if y_true_oh[:, i].sum() > 0
        ]
        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"📈 val_macro_auc = {macro_auc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]
