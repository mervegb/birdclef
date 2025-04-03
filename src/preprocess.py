import os
import librosa
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import StratifiedKFold
from utils import pad_spectrogram, is_high_energy

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms_grouped"
SR = 32000
CHUNK_DURATION = 10.0
MAX_CHUNKS_PER_FILE = 3
ENERGY_THRESHOLD_DB = 20
VARIANT = "A"  # "A" or "B"
N_SPLITS = 5
FOLD = 0  # default fold to use

# ========== MEL PARAMS ========== #
MEL_CONFIGS = {
    "A": {"n_fft": 1024, "hop_length": 512, "fmin": 50, "fmax": 14000, "n_mels": 128},
    "B": {"n_fft": 2048, "hop_length": 1024, "fmin": 200, "fmax": 14000, "n_mels": 224}
}
mel_params = MEL_CONFIGS[VARIANT]


def stratified_split(df, target_col="common_name", n_splits=5, fold=0):
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (_, val_idx) in enumerate(skf.split(df, df[target_col])):
        df.loc[val_idx, "fold"] = i
    return df[df["fold"] != fold], df[df["fold"] == fold]  # train_df, val_df


def process_row(index_row_mode):
    index, row_dict, mode = index_row_mode
    row = pd.Series(row_dict)

    try:
        subfolder, audio_file = row["filename"].split("/")
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        if not os.path.exists(audio_path):
            return

        y, sr = librosa.load(audio_path, sr=SR)
        chunk_samples = int(CHUNK_DURATION * sr)
        chunk_id = 0

        for start in range(0, len(y), chunk_samples):
            if chunk_id >= MAX_CHUNKS_PER_FILE:
                break
            clip = y[start:start + chunk_samples]
            if len(clip) < chunk_samples or not is_high_energy(clip, sr, ENERGY_THRESHOLD_DB):
                continue

            # Create spectrogram
            S = librosa.feature.melspectrogram(
                y=clip, sr=sr,
                n_fft=mel_params["n_fft"],
                hop_length=mel_params["hop_length"],
                n_mels=mel_params["n_mels"],
                fmin=mel_params["fmin"],
                fmax=mel_params["fmax"],
                power=2,
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
            S_padded = pad_spectrogram(S_dB_norm, S_dB_norm.shape[1])

            img = (S_padded * 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            group = row["group"]
            class_name = row["common_name"].replace(" ", "_").lower()
            class_dir = os.path.join(OUTPUT_DIR, f"{group}/{mode}/{class_name}")
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{index}_chunk{chunk_id}.jpg"
            output_path = os.path.join(class_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"✅ [{group}/{mode}] {output_path}")
            chunk_id += 1

    except Exception as e:
        print(f"❌ [{index}] Error: {e}")


# ========== MAIN ENTRY ========== #
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # Assign group1 or group2 based on label frequency
    df["label_count"] = df["common_name"].map(df["common_name"].value_counts())
    df["group"] = df["label_count"].apply(lambda x: "group1" if x >= 10 else "group2")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Generating MEL-{VARIANT} spectrograms split by group and fold...")

    # Process each group separately with stratified train/val split
    for group_name in ["group1", "group2"]:
        group_df = df[df["group"] == group_name].reset_index(drop=True)
        train_df, val_df = stratified_split(group_df, target_col="common_name", n_splits=N_SPLITS, fold=FOLD)

        train_jobs = [(i, row, "train") for i, row in train_df.iterrows()]
        val_jobs = [(i, row, "val") for i, row in val_df.iterrows()]
        all_jobs = train_jobs + val_jobs

        with Pool(cpu_count()) as pool:
            pool.map(process_row, all_jobs)

    print("✅ All done — no data leakage, clean splits per group.")