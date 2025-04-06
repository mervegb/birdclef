import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
import librosa
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
import soundfile as sf

# ========== CONFIG ==========
CSV_PATH = "data/train.csv"            # CSV file with metadata
AUDIO_DIR = Path("data/raw")           # Original .ogg audio files
TRAIN_DIR = Path("data/processed/audio/train")
VAL_DIR = Path("data/processed/audio/val")

TRAIN_SEGMENTS_DIR = Path("data/processed/audio/train_segments")
VAL_SEGMENTS_DIR = Path("data/processed/audio/val_segments")
TRAIN_SPECTROGRAM_DIR = Path("data/processed/spectrograms/train")
VAL_SPECTROGRAM_DIR = Path("data/processed/spectrograms/val")

RANDOM_STATE = 42

# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 500
N_MELS = 128
FMIN = 40
FMAX = 15000
POWER = 2
SR = 32000  # sample rate

def birds_stratified_split(df, target_col, test_size=0.2, rare_threshold=5):
    """
    Splits the DataFrame into train and validation sets in a stratified way.
    Classes with fewer than `rare_threshold` clips are added exclusively to training.
    """
    class_counts = df[target_col].value_counts()
    rare_classes = class_counts[class_counts < rare_threshold].index.tolist()
    print(f"Rare classes (<{rare_threshold} clips): {rare_classes}")
    
    df['train_flag'] = df[target_col].isin(rare_classes)
    train_df, val_df = train_test_split(
        df[~df['train_flag']],
        test_size=test_size,
        stratify=df[~df['train_flag']][target_col],
        random_state=RANDOM_STATE
    )
    train_df = pd.concat([train_df, df[df['train_flag']]], axis=0).reset_index(drop=True)
    train_df.drop('train_flag', axis=1, inplace=True)
    val_df.drop('train_flag', axis=1, inplace=True)
    
    return train_df, val_df

def split_audio_dataset():
    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV with shape:", df.shape)
    print("Overall class counts:")
    print(df["primary_label"].value_counts())
    
    train_df, val_df = birds_stratified_split(df, target_col="primary_label", test_size=0.2, rare_threshold=5)
    print("Final training set shape:", train_df.shape)
    print("Final validation set shape:", val_df.shape)
    
    print("Copying training files...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        filename = Path(row["filename"]).name
        label = row["primary_label"]
        src_path = AUDIO_DIR / row["filename"]
        dest_dir = TRAIN_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        if not src_path.exists():
            print(f"File {src_path} not found. Skipping.")
            continue
        copyfile(str(src_path), str(dest_path))
    
    print("Copying validation files...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        filename = Path(row["filename"]).name
        label = row["primary_label"]
        src_path = AUDIO_DIR / row["filename"]
        dest_dir = VAL_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        if not src_path.exists():
            print(f"File {src_path} not found. Skipping.")
            continue
        copyfile(str(src_path), str(dest_path))
    
    print("Dataset split complete.")

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def dbfs(signal):
    rms_val = compute_rms(signal)
    return 20 * np.log10(rms_val + 1e-6)

def segment_audio_with_overlap(audio_path, sr=32000, window_size=5, step_size=1, snr_threshold_db=-20):
    y, sr = librosa.load(audio_path, sr=sr)
    segment_length = int(sr * window_size)
    step = int(sr * step_size)
    
    segments = []
    segment_indices = []
    
    for start in range(0, len(y) - segment_length + 1, step):
        segment = y[start:start + segment_length]
        if dbfs(segment) < snr_threshold_db:
            continue
        segments.append(segment)
        segment_indices.append(start)
    
    return segments, segment_indices

def process_audio_directory(source_dir: Path, output_dir: Path, sr=32000, window_size=5, step_size=1, snr_threshold_db=-20):
    audio_files = list(source_dir.rglob("*.ogg"))
    print(f"Found {len(audio_files)} audio files in {source_dir}.")
    
    for audio_path in audio_files:
        segments, _ = segment_audio_with_overlap(audio_path, sr, window_size, step_size, snr_threshold_db)
        if not segments:
            continue
        relative_path = audio_path.relative_to(source_dir)
        parent_folder = relative_path.parent
        filename_stem = audio_path.stem
        output_subdir = output_dir / parent_folder
        output_subdir.mkdir(parents=True, exist_ok=True)
        for i, segment in enumerate(segments):
            segment_filename = f"{filename_stem}_seg{i}.wav"
            segment_path = output_subdir / segment_filename
            sf.write(str(segment_path), segment, sr)
        print(f"Saved {len(segments)} segments from {audio_path} to {output_subdir}")

def generate_spectrogram(audio_path, output_path):
    try:
        y, _ = librosa.load(str(audio_path), sr=SR)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=POWER
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to 0-1 float range and then scale to 255 for a PNG
        norm_img = 255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        norm_img = norm_img.astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), norm_img)
        return f"Processed: {audio_path.name}"
    except Exception as e:
        return f"Error processing {audio_path}: {e}"

def process_job(job):
    audio_path, output_path = job
    return generate_spectrogram(audio_path, output_path)

def process_directory(source_dir: Path, dest_dir: Path):
    audio_files = list(source_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {source_dir}")
    
    jobs = []
    for audio_path in audio_files:
        relative_path = audio_path.relative_to(source_dir)
        output_path = dest_dir / relative_path.with_suffix(".png")
        jobs.append((audio_path, output_path))
    
    with Pool(cpu_count()) as pool:
        from tqdm import tqdm
        results = list(tqdm(pool.imap_unordered(process_job, jobs), total=len(jobs)))
    
    for res in results:
        print(res)

if __name__ == '__main__':
    # 1) Split the dataset
    #split_audio_dataset()

    # 2) Segment all audio in train and val
    # process_audio_directory(TRAIN_DIR, TRAIN_SEGMENTS_DIR, sr=SR, window_size=5, step_size=1, snr_threshold_db=-20)
    # process_audio_directory(VAL_DIR, VAL_SEGMENTS_DIR, sr=SR, window_size=5, step_size=1, snr_threshold_db=-20)
    
    # 3) Generate spectrograms for train and validation segments
    # print("Processing training spectrograms...")
    # process_directory(TRAIN_SEGMENTS_DIR, TRAIN_SPECTROGRAM_DIR)
    # print("Processing validation spectrograms...")
    # process_directory(VAL_SEGMENTS_DIR, VAL_SPECTROGRAM_DIR)
    
    # print("Spectrogram generation complete.")