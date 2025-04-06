# train.py
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from bird_cnn import BirdCNN

# ========= CONFIG ========= #
TRAIN_DIR = "data/processed/spectrograms/train"
VAL_DIR = "data/processed/spectrograms/val"
BATCH_SIZE = 32
EPOCHS = 30
NUM_WORKERS = 4
SEED = 42

# ========= TRANSFORMS ========= #
# Force grayscale to ensure 1-channel input
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========= TRAINING FUNCTION ========= #
def train_model():
    seed_everything(SEED)

    # ====== Datasets and DataLoaders ====== #
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = ImageFolder(VAL_DIR, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ====== Model Instantiation ====== #
    # Optionally freeze backbone for the first few epochs (default freeze for 5 epochs)
    model = BirdCNN(num_classes=len(train_dataset.classes), freeze_backbone=True, freeze_epochs=5)

    # ====== Callbacks ====== #
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_auc",
        filename="epoch{epoch:02d}-auc{val_macro_auc:.4f}",
        save_top_k=1,
        save_last=True,
        mode="max"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_macro_auc",
        patience=7,
        mode="max",
        verbose=True
    )

    # ====== Trainer ====== #
    trainer = Trainer(
        max_epochs=EPOCHS,
        precision=16,  # Use AMP for efficiency
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    # ====== Start Training ====== #
    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!")

# ========== MAIN ==========
if __name__ == '__main__':
    train_model()