import os
import librosa
import numpy as np
import pandas as pd
import cv2

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms_filtered"
TARGET_HEIGHT = 128     # mel bands
TARGET_WIDTH = 512      # frames per 5-sec chunk
CHUNK_DURATION = 5.0    # seconds
ENERGY_THRESHOLD_DB = 20  # minimum average energy in dB
SR = 32000              # resample rate (optional but consistent)

# ========== HELPERS ========== #
def pad_spectrogram(S, target_width):
    if S.shape[1] >= target_width:
        return S[:, :target_width]
    pad_width = target_width - S.shape[1]
    return np.pad(S, ((0, 0), (0, pad_width)), mode='constant')

def is_high_energy(y, sr, threshold_db=20):
    rms = librosa.feature.rms(y=y)[0]
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)
    return np.mean(energy_db) > -threshold_db

# ========== MAIN LOOP ========== #
df = pd.read_csv(CSV_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for index, row in df.iterrows():
    try:
        subfolder, audio_file = row['filename'].split('/')
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        print(f"🎧 Processing {index}/{len(df)}: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            continue

        y, sr = librosa.load(audio_path, sr=SR)
        chunk_samples = int(CHUNK_DURATION * sr)

        chunk_id = 0
        for start in range(0, len(y), chunk_samples):
            clip = y[start:start+chunk_samples]
            if len(clip) < chunk_samples:
                break

            if not is_high_energy(clip, sr, threshold_db=ENERGY_THRESHOLD_DB):
                continue

            # 🎼 Compute mel-spectrogram
            S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=TARGET_HEIGHT)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())  # normalize

            # 📏 Pad to fixed width
            S_padded = pad_spectrogram(S_dB, target_width=TARGET_WIDTH)

            # 🖼 Convert to image (8-bit grayscale)
            img = (S_padded * 255).astype(np.uint8)
            img = np.flip(img, axis=0)  # flip to match visual orientation

            # 💾 Save with OpenCV
            class_name = row['common_name'].replace(" ", "_").lower()
            class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            output_path = os.path.join(class_dir, f"{index}_chunk{chunk_id}.jpg")
            cv2.imwrite(output_path, img)

            print(f"✅ Saved: {output_path}")
            chunk_id += 1

    except Exception as e:
        print(f"❌ [{index}] Failed for {row['filename']}: {e}")