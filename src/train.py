import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.bird_cnn import BirdCNN  

# ========== CONFIG ========== #
DATA_DIR = "data/processed/spectrograms_filtered"
BATCH_SIZE = 32
IMG_SIZE = (128, 512)
EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 0  # 👈 safer for Mac & debugging; increase if you're on Linux
MODEL_WEIGHTS_PATH = "birdnet_weights.pt"
MODEL_FULL_PATH = "birdnet_model_full.pt"

# ========== TRANSFORMS ========== #
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  
    transforms.ToTensor()
])

# ========== DATASET ========== #
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# ✅ Avoid empty val set
val_size = max(1, int(0.2 * len(dataset)))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f"📊 Dataset split: Train={len(train_ds)} | Val={len(val_ds)}")
print(f"📚 Classes found: {dataset.classes}")

# ========== DATALOADERS ========== #
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========== MODEL ========== #
num_classes = len(dataset.classes)
model = BirdCNN(num_classes=num_classes)

# ========== CALLBACKS ========== #
checkpoint_callback = ModelCheckpoint(
    monitor="val_macro_auc",
    filename="birdnet-epoch{epoch:02d}-auc{val_macro_auc:.4f}",
    save_top_k=-1,            # Save all checkpoints
    every_n_epochs=1,         # Save every epoch
    save_last=True,           # (Optional) Save latest separately as last.ckpt
    mode="max"                # Because higher AUC is better
)


early_stopping_callback = EarlyStopping(
    monitor="val_macro_auc",
    patience=5,
    mode="max",
    verbose=True
)

# ========== TRAINER ========== #
trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",  # GPU if available
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=10,
    num_sanity_val_steps=0
)

# ========== TRAIN ========== #
# trainer.fit(model, train_loader, val_loader)

# # ========== SAVE FINAL MODEL ========== #
# # Save model weights (recommended for deployment/fine-tuning)
# torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
# print(f"✅ Model weights saved to: {MODEL_WEIGHTS_PATH}")

# # Optionally save full model (not portable across different environments)
# torch.save(model, MODEL_FULL_PATH)
# print(f"✅ Full model saved to: {MODEL_FULL_PATH}")