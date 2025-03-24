import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ========== CONFIG ========== #
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms_filtered"
TARGET_HEIGHT = 128     # mel bands
TARGET_WIDTH = 512      # frames per 5-sec chunk
FIGSIZE = (3, 3)        # plot size
CHUNK_DURATION = 5.0    # seconds
ENERGY_THRESHOLD_PERCENTILE = 30  # filter out bottom 30% energy chunks

# ========== HELPERS ========== #
def pad_spectrogram(S, target_width):
    height, width = S.shape
    if width >= target_width:
        return S[:, :target_width]  # crop
    else:
        pad_width = target_width - width
        return np.pad(S, ((0, 0), (0, pad_width)), mode='constant')


def is_high_energy(y, sr, threshold_db=20):
    rms = librosa.feature.rms(y=y)[0]
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)
    avg_db = np.mean(energy_db)
    return avg_db > -threshold_db  # e.g. keep anything louder than -20dB


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

        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        chunk_samples = int(CHUNK_DURATION * sr)

        chunk_id = 0
        for start in range(0, len(y), chunk_samples):
            clip = y[start:start+chunk_samples]
            if len(clip) < chunk_samples:
                break  # Skip incomplete chunk

            # 🔍 Skip low-energy clips
            if not is_high_energy(clip, sr, threshold_db=20):
                continue

            # 🎼 Compute mel-spectrogram
            S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=TARGET_HEIGHT)
            S_dB = librosa.power_to_db(S, ref=np.max)
            S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())  # normalize

            # 📏 Pad to fixed width
            S_padded = pad_spectrogram(S_dB, target_width=TARGET_WIDTH)

            # 🎨 Plot and save
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.axis('off')
            librosa.display.specshow(S_padded, sr=sr, ax=ax, cmap='gray_r')

            class_name = row['common_name'].replace(" ", "_").lower()
            class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"{index}_chunk{chunk_id}.png"
            output_path = os.path.join(class_dir, filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"✅ Saved: {output_path}")
            chunk_id += 1

    except Exception as e:
        print(f"❌ [{index}] Failed for {row['filename']}: {e}")