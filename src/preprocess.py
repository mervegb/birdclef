import os
import librosa
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count
from utils import pad_spectrogram, is_high_energy

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms"
SR = 32000
CHUNK_DURATION = 10.0  # Seconds
MAX_CHUNKS_PER_FILE = 3
ENERGY_THRESHOLD_DB = 20

# MEL PARAMS
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 500
FMIN = 40
FMAX = 15000
POWER = 2


def process_row(index_row):
    index, row_dict = index_row
    row = pd.Series(row_dict)
    try:
        subfolder, audio_file = row['filename'].split('/')
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        print(f"🎧 Processing {audio_path}")

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
            if not is_high_energy(clip, sr, ENERGY_THRESHOLD_DB):
                continue

            S = librosa.feature.melspectrogram(
                y=clip, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=POWER
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

            target_width = S_dB_norm.shape[1]
            S_padded = pad_spectrogram(S_dB_norm, target_width)

            img = (S_padded * 255).astype(np.uint8)
            img = np.flip(img, axis=0)  # Flip vertically

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            class_name = row['common_name'].replace(" ", "_").lower()
            class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{index}_chunk{chunk_id}.jpg"
            output_path = os.path.join(class_dir, filename)
            cv2.imwrite(output_path, img)
            print(f"✅ Saved: {output_path}")
            chunk_id += 1

    except Exception as e:
        print(f"❌ [{index}] Error processing {row.get('filename', 'unknown')}: {e}")

# ========== MAIN PARALLEL RUN ========== #
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Starting with {len(df)} files using {cpu_count()} workers...")

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_row, list(df.iterrows()))

    print("✅ All done!")