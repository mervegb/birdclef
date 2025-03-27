import os
import librosa
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms"
TARGET_HEIGHT = 128
TARGET_WIDTH = 512
CHUNK_DURATION = 5.0
ENERGY_THRESHOLD_DB = 10
MAX_CHUNKS_PER_FILE = 3
SR = 32000

# ========== HELPERS ========== #
def pad_spectrogram(S, target_width):
    if S.shape[1] >= target_width:
        return S[:, :target_width]
    pad_width = target_width - S.shape[1]
    return np.pad(S, ((0, 0), (0, pad_width)), mode='constant')

def is_high_energy(y, sr, threshold_db):
    rms = librosa.feature.rms(y=y)[0]
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)
    return np.mean(energy_db) > -threshold_db

def process_row(index_row):
    index, row_dict = index_row
    row = pd.Series(row_dict)
    try:
        subfolder, audio_file = row['filename'].split('/')
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        print(f"🎧 Processing {index}: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return

        y, sr = librosa.load(audio_path, sr=SR)
        chunk_samples = int(CHUNK_DURATION * sr)

        chunk_id = 0
        for start in range(0, len(y), chunk_samples):
            if chunk_id >= MAX_CHUNKS_PER_FILE:
                break

            clip = y[start:start + chunk_samples]
            if len(clip) < chunk_samples:
                break

            if not is_high_energy(clip, sr, threshold_db=ENERGY_THRESHOLD_DB):
                continue

            S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=TARGET_HEIGHT)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
            S_padded = pad_spectrogram(S_dB, TARGET_WIDTH)

            img = (S_padded * 255).astype(np.uint8)
            img = np.flip(img, axis=0)

            # 🔧 Fix shape: ensure proper 2D image
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]

            class_name = row['common_name'].replace(" ", "_").lower()
            class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{index}_chunk{chunk_id}.jpg"
            output_path = os.path.join(class_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"✅ Saved: {output_path}")
            chunk_id += 1

    except Exception as e:
        print(f"❌ [{index}] Failed for {row['filename']}: {e}")

# ========== MAIN PARALLEL RUN ========== #
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Starting with {len(df)} files using {cpu_count()} workers...")

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_row, list(df.iterrows()))

    print("✅ All done!")