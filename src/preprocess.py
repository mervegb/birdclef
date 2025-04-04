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
CHUNK_DURATION = 5.0     # Reduced from 10.0 to 5.0 seconds for finer segments
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
    Normalize the spectrogram, scale it to 0-255, and then apply a color map.
    """
    X_norm = (X - X.mean()) / (X.std() + eps)
    X_scaled = 255 * (X_norm - X_norm.min()) / (X_norm.max() - X_norm.min() + eps)
    X_uint8 = X_scaled.astype(np.uint8)
    # Apply a color map for improved visual representation
    color_mapped = cv2.applyColorMap(X_uint8, cv2.COLORMAP_JET)
    # Convert BGR (OpenCV default) to RGB
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
            # cv2.imwrite expects BGR format
            cv2.imwrite(str(out_path), cv2.cvtColor(mel_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    except Exception as e:
        print(f"❌ [{row.name}] Error: {e}")

# ========== MAIN RUN FOR PREPROCESSING ========== #
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

# Uncomment the following line to run spectrogram generation:
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    generate_spectrograms()