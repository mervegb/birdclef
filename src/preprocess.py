import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths
CSV_PATH = "data/train.csv"
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

for index, row in df.iterrows():
    try:
        subfolder, audio_file = row['filename'].split('/')
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        print(f"🎧 Processing {index}/{len(df)}: {audio_path}")

        # Skip if file doesn't exist
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            continue

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Convert to mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Normalize
        S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

        # Plot spectrogram
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='gray_r')

        # Class-based folder
        class_name = row['common_name'].replace(" ", "_").lower()
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Save image directly into class folder
        output_filename = f"{index}.png"
        output_path = os.path.join(class_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ [{index}] Failed for {row['filename']}: {e}")