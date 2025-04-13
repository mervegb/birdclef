import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

label_mapper = {
    label: idx
    for idx, label in enumerate(sorted(df['primary_label'].unique()))
}

# rev_mapper = {
#     idx: label,
#     for label, idx in label_mapper.items()
# }

class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, mode='train'):
        """
        Args:
            df (pd.DataFrame): DataFrame loaded from train.csv containing the columns 'filename' and 'primary_label'.
            audio_dir (str or Path): The directory path where the audio files are stored.
            mode (str): 'train' to return labels along with data, or any other mode for inference.
        """
        self.df = df
        self.mode = mode
        self.audio_dir = Path(audio_dir)
        
        # Form full file paths by joining the audio_dir and the 'filename' column.
        self.file_paths = self.df['filename'].apply(lambda x: self.audio_dir / x).tolist()
        
        # Create label mapping if the 'primary_label' column exists.
        if 'primary_label' in self.df.columns:
            unique_labels = sorted(self.df['primary_label'].unique())
            self.label_mapper = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_mapper = {idx: label for label, idx in self.label_mapper.items()}
            self.labels = self.df['primary_label'].map(self.label_mapper).tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.file_paths)
    
    def process(self, audio_path):
        """
        Loads an audio file and processes it to create a normalized mel spectrogram
        of fixed dimensions (1 x 128 x 640).

        Args:
            audio_path (str or Path): The path to the audio file.
            
        Returns:
            mel_sp (np.ndarray): The processed mel spectrogram.
        """
        # Load audio data with a sample rate of 32000.
        audio_data, sr = librosa.load(audio_path, sr=32000)
        
        # Amplify the signal (adjust this factor as needed).
        data = audio_data * 1024
        
        # Define the target duration and number of samples.
        chunk_duration = 10  # seconds
        min_len = int(chunk_duration * sr)
        
        # If the audio is too short, tile (repeat) the data until reaching the required length.
        if len(data) < min_len:
            cnt = int(np.ceil(min_len / len(data)))
            data = np.tile(data, cnt)
        
        # Center-crop the audio to exactly min_len samples.
        if len(data) > min_len:
            leftover = len(data) - min_len
            front_crop = leftover // 2
            data = data[front_crop: front_crop + min_len]
        else:
            data = data[:min_len]
        
        # Generate the mel spectrogram.
        mel_sp = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_fft=1024,
            hop_length=500,
            n_mels=128,
            fmin=50,
            fmax=16000,
            power=2.0
        )
        
        # Convert the power spectrogram to decibels.
        mel_sp = librosa.power_to_db(mel_sp, ref=1)
        
        # Normalize using min-max scaling.
        eps = 1e-12  # Prevent division by zero.
        mel_sp = (mel_sp - np.min(mel_sp)) / (np.max(mel_sp) - np.min(mel_sp) + eps)
        
        # Crop or pad the time dimension so that it has exactly 640 frames.
        if mel_sp.shape[1] >= 640:
            mel_sp = mel_sp[:, :640]
        else:
            pad_width = 640 - mel_sp.shape[1]
            mel_sp = np.pad(mel_sp, ((0, 0), (0, pad_width)), mode='constant')
        
        return mel_sp

    def __getitem__(self, index):
        # Get the full path to the audio file.
        audio_path = self.file_paths[index]
        # Process the audio file to generate the mel spectrogram.
        spectrogram = self.process(audio_path)
        # Convert the spectrogram to a torch tensor and add a channel dimension.
        x = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0)  # Shape: [1, 128, 640]
        
        if self.mode == 'train' and self.labels is not None:
            label = self.labels[index]
            return x, label
        else:
            return x

# Example usage:
if __name__ == "__main__":
    # Load your CSV; update the path as needed.
    df = pd.read_csv('data/train.csv')
    
    # Specify the directory containing your audio files.
    audio_directory = 'data/raw'
    
    # Instantiate the dataset.
    dataset = BirdCLEFDataset(df, audio_directory, mode='train')
    
    # Process the first file directly for visualization
    first_audio_path = dataset.file_paths[0]
    mel_sp = dataset.process(first_audio_path)
    
    # Retrieve one sample and its label.
    print("Mel spectrogram shape:", mel_sp.shape)  # Expected: (128, 640)

    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_sp, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Mel Spectrogram - {df['primary_label'].iloc[0]}")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency Bins")
    plt.colorbar(label="Normalized dB")
    plt.tight_layout()
    plt.show()