import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("data/train.csv")

# Base folder where audio files are stored
AUDIO_BASE_DIR = "data/raw"
OUTPUT_DIR = "data/processed/spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for index, row in df.iterrows():
    try:
        subfolder, audio_file = row['filename'].split('/')
        audio_path = os.path.join(AUDIO_BASE_DIR, subfolder, audio_file)
        print(f'We are at {index}/{len(df)}')

        # Skip if file doesn't exist
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            continue

        # Load the audio
        y, sr = librosa.load(audio_path, sr=None)

        # Create a mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='viridis')

        # Clean filename
        common_name = row['common_name'].replace(' ', '_').lower()
        output_filename = f"{index}_{common_name}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"✅ [{index}] Saved: {output_filename}")

    except Exception as e:
        print(f"❌ [{index}] Failed for {row['filename']}: {e}")