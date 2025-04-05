import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool, cpu_count
import soundfile as sf 

# ========== CONFIG ==========
CSV_PATH = "data/train.csv"  # Path to your CSV file
AUDIO_DIR = Path("data/raw")  # Where the original audio (.ogg) files are stored
TRAIN_DIR = Path("data/processed/audio/train")  # Destination for training files
VAL_DIR = Path("data/processed/audio/val")      # Destination for validation files

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
    Splits the DataFrame into train and validation sets in a stratified way,
    adding classes with fewer than `rare_threshold` clips exclusively to the training set.
    """
    # Count clips per species
    class_counts = df[target_col].value_counts()
    # Identify rare classes with fewer than rare_threshold clips
    rare_classes = class_counts[class_counts < rare_threshold].index.tolist()
    print("Rare classes (less than {} clips):".format(rare_threshold))
    print(rare_classes)
    
    # Flag rows belonging to rare classes
    df['train_flag'] = df[target_col].isin(rare_classes)
    
    # Stratified split on classes with sufficient samples
    train_df, val_df = train_test_split(
        df[~df['train_flag']],
        test_size=test_size,
        stratify=df[~df['train_flag']][target_col],
        random_state=RANDOM_STATE
    )
    
    # Add rare classes exclusively to the training set
    train_df = pd.concat([train_df, df[df['train_flag']]], axis=0).reset_index(drop=True)
    
    # Remove the helper flag
    train_df.drop('train_flag', axis=1, inplace=True)
    val_df.drop('train_flag', axis=1, inplace=True)
    
    return train_df, val_df

def split_audio_dataset():
    # Load CSV data
    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV with shape:", df.shape)
    
    # Display class counts for overview
    class_counts = df["primary_label"].value_counts()
    print("Overall class counts:")
    print(class_counts)
    
    # Perform the stratified split using our function (rare classes with <10 clips go to training)
    train_df, val_df = birds_stratified_split(df, target_col="primary_label", test_size=0.2, rare_threshold=5)
    
    print("Final training set shape:", train_df.shape)
    print("Final validation set shape:", val_df.shape)
    
    # Copy training files with a progress bar
    print("Copying training files...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        filename = Path(row["filename"]).name  # Flatten subfolders by taking only the base filename
        label = row["primary_label"]
        src_path = AUDIO_DIR / row["filename"]
        dest_dir = TRAIN_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        if not src_path.exists():
            print(f"File {src_path} not found. Skipping.")
            continue
        copyfile(str(src_path), str(dest_path))
    
    # Copy validation files with a progress bar
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
    """Compute root-mean-square (RMS) of an audio signal."""
    return np.sqrt(np.mean(signal ** 2))

def dbfs(signal):
    """Convert RMS to dBFS (decibels relative to full scale, assuming a maximum of 1.0)."""
    rms_val = compute_rms(signal)
    # Adding a small epsilon to avoid log(0)
    return 20 * np.log10(rms_val + 1e-6)

def segment_audio_with_overlap(audio_path, sr=32000, window_size=5, step_size=1, snr_threshold_db=-20):
    """
    Splits the audio into overlapping segments.
    
    Parameters:
    - audio_path: path to the audio file.
    - sr: sample rate.
    - window_size: length of each segment in seconds.
    - step_size: sliding window step size in seconds (e.g., 1 second).
    - snr_threshold_db: minimum dBFS threshold; segments below this are considered low-energy (noisy) and skipped.
    
    Returns:
    - segments: a list of audio segments that pass the energy filter.
    - segment_indices: the start sample index for each segment.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    segment_length = int(sr * window_size)
    step = int(sr * step_size)
    
    segments = []
    segment_indices = []
    
    # Slide over the audio with the specified step
    for start in range(0, len(y) - segment_length + 1, step):
        segment = y[start:start + segment_length]
        level_db = dbfs(segment)
        # Only keep segments that have enough energy (i.e., above the threshold)
        if level_db < snr_threshold_db:
            continue
        segments.append(segment)
        segment_indices.append(start)
    
    return segments, segment_indices

def process_audio_directory(
    source_dir: Path,
    output_dir: Path,
    sr=32000,
    window_size=5,
    step_size=1,
    snr_threshold_db=-20
):
    """
    Loops through all .ogg files in source_dir, segments each one,
    and saves the resulting segments to output_dir in a mirrored folder structure.
    """
    # Recursively find all .ogg files in the source directory
    audio_files = list(source_dir.rglob("*.ogg"))
    print(f"Found {len(audio_files)} audio files in {source_dir}.")

    for audio_path in audio_files:
        # Segment the audio
        segments, indices = segment_audio_with_overlap(
            audio_path,
            sr=sr,
            window_size=window_size,
            step_size=step_size,
            snr_threshold_db=snr_threshold_db
        )

        if not segments:
            # If no segments pass the threshold, skip
            continue

        # Build a relative path for where to save segments
        relative_path = audio_path.relative_to(source_dir)
        parent_folder = relative_path.parent
        filename_stem = audio_path.stem

        # Create the output subdirectory (mirroring folder structure)
        output_subdir = output_dir / parent_folder
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save each segment
        for i, segment in enumerate(segments):
            segment_filename = f"{filename_stem}_seg{i}.wav"
            segment_path = output_subdir / segment_filename
            sf.write(str(segment_path), segment, sr)

        print(f"Saved {len(segments)} segments from {audio_path} to {output_subdir}")

def generate_spectrogram(audio_path, output_path):
    """
    Loads an audio segment from audio_path, computes its mel spectrogram,
    converts to decibels, normalizes it to grayscale (0-255) and saves it as a PNG.
    """
    try:
        # Load audio file
        y, _ = librosa.load(str(audio_path), sr=SR)
        
        # Compute mel spectrogram
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
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-255 for grayscale image
        norm_img = 255 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        norm_img = norm_img.astype(np.uint8)
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image as PNG
        cv2.imwrite(str(output_path), norm_img)
        return f"Processed: {audio_path.name}"
    except Exception as e:
        return f"Error processing {audio_path}: {e}"

def process_job(job):
    """Top-level function to process a single job tuple."""
    audio_path, output_path = job
    return generate_spectrogram(audio_path, output_path)


def process_directory(source_dir: Path, dest_dir: Path):
    """
    Process all .wav audio files in source_dir (recursively) and generate spectrograms in dest_dir.
    The folder structure is mirrored.
    """
    # Find all .wav files in the source directory recursively
    audio_files = list(source_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {source_dir}")
    
    # Prepare jobs as tuples: (audio file path, destination path)
    jobs = []
    for audio_path in audio_files:
        relative_path = audio_path.relative_to(source_dir)
        output_path = dest_dir / relative_path.with_suffix(".png")
        jobs.append((audio_path, output_path))
    
    # Process files in parallel using the top-level process_job function
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_job, jobs), total=len(jobs)))
    
    for res in results:
        print(res)

def count_spectrograms(directory: Path):
    return len(list(directory.rglob("*.png")))

if __name__ == '__main__':
    # 1) Split the dataset
    # split_audio_dataset()

    # 2) Segment all audio in train and val 
    # process_audio_directory(
    #     source_dir=TRAIN_DIR,
    #     output_dir=TRAIN_SEGMENTS_DIR,
    #     sr=32000,
    #     window_size=5,
    #     step_size=1,
    #     snr_threshold_db=-20
    # )
    # process_audio_directory(
    #     source_dir=VAL_DIR,
    #     output_dir=VAL_SEGMENTS_DIR,
    #     sr=32000,
    #     window_size=5,
    #     step_size=1,
    #     snr_threshold_db=-20
    # )
    
    # 3) Spectrogram generation
    # print("Processing training spectrograms...")
    # process_directory(TRAIN_SEGMENTS_DIR, TRAIN_SPECTROGRAM_DIR)
    
    # print("Processing validation spectrograms...")
    # process_directory(VAL_SEGMENTS_DIR, VAL_SPECTROGRAM_DIR)
    
    # print("Spectrogram generation complete.")
    
  