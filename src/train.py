import os
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =============================================================================
#                PART 1: PREPROCESSING & SPECTROGRAM GENERATION
# =============================================================================

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"                   # CSV containing filenames and labels
AUDIO_DIR = Path("data/raw")                  # Directory with raw audio files
OUTPUT_DIR = Path("data/processed/spectrograms")  # Where spectrogram images will be saved
SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0     # 5-second chunks for finer segments
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256       # Smaller hop for better time resolution
FMIN = 40
FMAX = 15000
POWER = 2

# ========== HELPER FUNCTIONS ========== #
def get_audio_info(filepath):
    with sf.SoundFile(filepath) as f:
        return {"frames": f.frames, "sr": f.samplerate, "duration": f.frames / f.samplerate}

def compute_melspec(y, sr, n_mels, fmin, fmax):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=POWER
    )
    # Convert to decibel scale and normalize with reference to the maximum value
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def mono_to_color(X, eps=1e-6):
    """
    Normalize the spectrogram, scale to 0-255, and apply a color map.
    """
    X_norm = (X - X.mean()) / (X.std() + eps)
    X_scaled = 255 * (X_norm - X_norm.min()) / (X_norm.max() - X_norm.min() + eps)
    X_uint8 = X_scaled.astype(np.uint8)
    # Apply a color map for improved visual representation
    color_mapped = cv2.applyColorMap(X_uint8, cv2.COLORMAP_JET)
    # Convert from BGR (OpenCV default) to RGB
    color_mapped = cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)
    return color_mapped

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        n_repeats = length // len(y)
        remainder = length % len(y)
        y = np.concatenate([y] * n_repeats + [y[:remainder]])
    elif len(y) > length:
        start = start or (np.random.randint(len(y) - length) if is_train else 0)
        y = y[start:start + length]
    return y

# ========== DATA SPLITTING ========== #
def stratified_birdclef_split(df, target_col='primary_label', test_size=0.2):
    class_counts = df[target_col].value_counts()
    low_count_classes = class_counts[class_counts < 2].index.tolist()
    df['keep'] = df[target_col].isin(low_count_classes)
    strat_df = df[~df['keep']]
    train_df, val_df = train_test_split(
        strat_df,
        test_size=test_size,
        stratify=strat_df[target_col],
        random_state=42
    )
    train_df = pd.concat([train_df, df[df['keep']]], axis=0).reset_index(drop=True)
    train_df.drop(columns='keep', inplace=True)
    val_df.drop(columns='keep', inplace=True)
    return train_df, val_df

# ========== MEL GENERATION ========== #
def process_row(row_dict_mode):
    row_dict, mode = row_dict_mode
    row = pd.Series(row_dict)
    try:
        audio_path = AUDIO_DIR / row["filename"]
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        chunk_samples = int(CHUNK_DURATION * sr)

        for i in range(0, len(y), chunk_samples):
            clip = y[i:i + chunk_samples]
            if len(clip) < chunk_samples:
                continue

            # Compute the mel-spectrogram
            mel = compute_melspec(y=clip, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
            # Convert single-channel spectrogram to a colored image
            mel_img = mono_to_color(mel)
            # Flip vertically to match conventional spectrogram orientation
            mel_img = np.flip(mel_img, axis=0)

            # Save image in a folder corresponding to its label
            class_name = row["primary_label"]
            out_dir = OUTPUT_DIR / mode / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{row.name}_chunk{i // chunk_samples}.png"
            # cv2.imwrite expects BGR format; convert from RGB
            cv2.imwrite(str(out_path), cv2.cvtColor(mel_img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
    except Exception as e:
        print(f"❌ [{row.name}] Error: {e}")

def generate_spectrograms():
    df = pd.read_csv(CSV_PATH)
    print("📊 Loaded CSV:", df.shape)
    print(df.head())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Splitting dataset...")
    train_df, val_df = stratified_birdclef_split(df, target_col="primary_label", test_size=0.2)
    train_jobs = [(row, "train") for _, row in train_df.iterrows()]
    val_jobs = [(row, "val") for _, row in val_df.iterrows()]
    all_jobs = train_jobs + val_jobs

    print(f"🔁 Processing {len(all_jobs)} audio chunks using {cpu_count()} cores...")
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_row, all_jobs), total=len(all_jobs)))
    print("✅ Done — spectrograms saved to:", OUTPUT_DIR)

# =============================================================================
#                         PART 2: MODEL & TRAINING
# =============================================================================

# ========== MODEL DEFINITION ========== #
class BirdCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        # Load EfficientNetB0 backbone with pretrained ImageNet weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Unfreeze all layers for full fine-tuning on spectrograms
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
        # Gather all validation outputs and compute macro-averaged ROC-AUC
        all_probs = torch.cat([x[0] for x in self.validation_outputs], dim=0).numpy()
        all_targets = torch.cat([x[1] for x in self.validation_outputs], dim=0).numpy()
        self.validation_outputs.clear()

        # One-hot encode the targets for ROC-AUC per class
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

# ========== DATA AUGMENTATION & LOADING ========== #
DATA_DIR = "data/processed/spectrograms"
BATCH_SIZE = 64
EPOCHS = 50
IMG_SIZE = (224, 224)
NUM_WORKERS = os.cpu_count()
MODEL_WEIGHTS_PATH = "model_weights.pt"
MODEL_FULL_PATH = "model_full.pt"

# Data augmentation transforms
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Placeholder values
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
])
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=val_transforms)
print(f"📊 Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
print(f"🔢 Number of classes: {len(train_dataset.classes)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, persistent_workers=True)

# ========== TRAINING ========== #
def train_model():
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
    
    # Uncomment the next line to start training:
    trainer.fit(model, train_loader, val_loader)
    
    # After training, you can save the model weights:
    # torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    # torch.save(model, MODEL_FULL_PATH)
    # print(f"✅ Model saved to: {MODEL_FULL_PATH}")

# =============================================================================
#                                MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    freeze_support()
    
    # To generate spectrograms from raw audio, uncomment the following line:
    #generate_spectrograms()
    
    # To train the model, uncomment the following line:
    train_model()