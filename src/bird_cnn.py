# bird_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score

class BirdCNN(pl.LightningModule):
    def __init__(self, num_classes, freeze_backbone=True, freeze_epochs=5):
        """
        Args:
            num_classes (int): Number of bird species.
            freeze_backbone (bool): Whether to freeze the pretrained backbone initially.
            freeze_epochs (int): Number of epochs to freeze the backbone before unfreezing.
        """
        super().__init__()
        self.save_hyperparameters()
        self.freeze_backbone = freeze_backbone
        self.freeze_epochs = freeze_epochs

        # Load pretrained EfficientNet-B0
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Patch the first conv layer to accept 1-channel input (grayscale spectrograms)
        in_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=in_conv.out_channels,
            kernel_size=in_conv.kernel_size,
            stride=in_conv.stride,
            padding=in_conv.padding,
            bias=False
        )
        
        # Backbone feature extractor
        self.feature_extractor = backbone.features
        
        # Optionally freeze the backbone initially
        if self.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Classification head
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
        x = self.classifier(x)
        return x

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
        
        # Use sigmoid for per-class independent probabilities (for ROC-AUC)
        probs = torch.sigmoid(logits).detach().cpu()
        targets = y.detach().cpu()
        self.validation_outputs.append((probs, targets))
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Aggregate outputs from validation steps
        all_probs = torch.cat([x[0] for x in self.validation_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_outputs], dim=0).numpy()
        self.validation_outputs.clear()
        
        # One-hot encode targets for per-class ROC-AUC calculation
        y_true_oh = F.one_hot(torch.tensor(all_targets), num_classes=self.hparams.num_classes).numpy()
        aucs = []
        for i in range(self.hparams.num_classes):
            if y_true_oh[:, i].sum() > 0:  # Only compute AUC if there are positive examples
                try:
                    auc = roc_auc_score(y_true_oh[:, i], all_probs[:, i])
                    aucs.append(auc)
                except ValueError:
                    continue
        macro_auc = sum(aucs) / len(aucs) if aucs else 0.0
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        print(f"📈 val_macro_auc = {macro_auc:.4f}")

    def on_train_epoch_start(self):
        # Unfreeze the backbone after freeze_epochs have passed
        if self.freeze_backbone and self.current_epoch == self.freeze_epochs:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print(f"Unfreezing backbone at epoch {self.current_epoch}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)  # Lower LR for stability
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]