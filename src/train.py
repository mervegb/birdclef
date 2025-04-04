import os
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from bird_cnn import BirdCNN 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =============================================================================
#                         PART 2: MODEL TRAINING
# =============================================================================

# ========== CONFIG ========== #
DATA_DIR = "data/processed/spectrograms"
BATCH_SIZE = 64              # Adjust batch size as needed
EPOCHS = 50
IMG_SIZE = (224, 224)
NUM_WORKERS = os.cpu_count()
MODEL_WEIGHTS_PATH = "model_weights.pt"
MODEL_FULL_PATH = "model_full.pt"

# ========== DATA AUGMENTATION & TRANSFORMS ========== #
# NOTE: The following normalization uses placeholder values.
# For best results, compute the mean and std for your spectrogram dataset.
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
])
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== DATA LOADING ========== #
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=val_transforms)
print(f"📊 Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
print(f"🔢 Number of classes: {len(train_dataset.classes)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, persistent_workers=True)

# ========== TRAINING ========== #
# Proceed with training
model = BirdCNN(num_classes=len(train_dataset.classes))
    
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
trainer = Trainer(
    precision=16,
    max_epochs=EPOCHS,
    accelerator="auto",
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=10,
    num_sanity_val_steps=0
)

# Uncomment to start training:
# trainer.fit(model, train_loader, val_loader)

# After training, save model weights if desired:
# torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
# torch.save(model, MODEL_FULL_PATH)
# print(f"✅ Model saved to: {MODEL_FULL_PATH}")