import torch
import torch.nn as nn
import timm
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import kaggle_metric_utilities


# Model definition from training code
class Model(nn.Module):
    def __init__(self, model_name, num_classes=None):
        super().__init__()
        # Create the base model without the classifier
        self.base_model = timm.create_model(
            model_name, 
            pretrained=False,
            in_chans=1, 
            num_classes=0
        )
        # Add a more sophisticated classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.base_model.num_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        features = self.base_model.forward_features(x)
        pooled = self.global_pool(features).flatten(1)
        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x



# Configuration (reusing parts from your training code)
config = {
    "sample_rate": 32000,
    "amplification_factor": 1024,
    "chunk_duration": 5,  # seconds per sample
    "n_fft": 1024,
    "hop_length": 500,
    "n_mels": 128,
    "fmin": 50,
    "fmax": 16000,
    "power": 2.0,
    "model_name": "tf_efficientnet_b0",
    "test_audio_dir": "/kaggle/input/birdclef-2025/test_soundscapes", 
    "submission_csv": "/kaggle/input/birdclef-2025/sample_submission.csv", 
    "model_paths": [
        "/kaggle/input/effnet-b0-dataset/model_fold_0_epoch_20_auc_0.8782.pth", 
        "/kaggle/input/effnet-b0-dataset/model_fold_1_epoch_20_auc_0.8805.pth",
        "/kaggle/input/effnet-b0-dataset/model_fold_2_epoch_20_auc_0.8789.pth"
    ],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_chunks_per_audio": 12,  # Number of chunks to extract per test audio
    "overlap": 0.5,  # Overlap between chunks
}


def process_audio(audio_path, sr=config["sample_rate"], num_chunks=config["num_chunks_per_audio"], overlap=config["overlap"]):
    """
    Process audio file into multiple overlapping chunks and convert to mel spectrograms
    """
    # Load audio file
    audio_data, _ = librosa.load(audio_path, sr=sr)
    
    # Apply amplification
    audio_data = audio_data * config["amplification_factor"]
    
    # Calculate chunk size and stride
    chunk_samples = int(config["chunk_duration"] * sr)
    stride = int(chunk_samples * (1 - overlap))
    
    # Extract chunks
    chunks = []
    for i in range(0, max(1, len(audio_data) - chunk_samples), stride):
        if len(chunks) >= num_chunks:
            break
        chunk = audio_data[i:i + chunk_samples]
        
        # Ensure consistent length
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        # Compute mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            power=config["power"],
        )
        
        # Convert to dB and normalize
        mel_sp = librosa.power_to_db(mel_sp, ref=1)
        mel_sp = (mel_sp - mel_sp.min()) / (mel_sp.max() - mel_sp.min() + 1e-12)
        
        chunks.append(mel_sp)
    
    return chunks


def run_inference():
    """
    Run inference on test data and create submission file
    """
    print("Starting inference...")
    
    # Load sample submission to get the structure and IDs
    submission_df = pd.read_csv(config["submission_csv"])
    print(f"Submission format: {submission_df.shape}")
    
    # Get class names from columns (excluding row_id)
    classes = [col for col in submission_df.columns if col != 'row_id']
    num_classes = len(classes)
    
    # Create label mappers
    label_mapper = {label: idx for idx, label in enumerate(classes)}
    reverse_label_mapper = {idx: label for label, idx in label_mapper.items()}
    
    # Get test audio files
    test_audio_dir = Path(config["test_audio_dir"])
    test_files = []
    test_ids = []
    
    for row_id in submission_df['row_id']:
        # Parse file name and seconds from row_id (format: soundscape_[file_id]_[seconds])
        parts = row_id.split('_')
        file_id = parts[1]
        seconds = int(parts[2])
        
        # Find corresponding audio file
        audio_file = next(test_audio_dir.glob(f"*{file_id}*.ogg"), None)
        if audio_file:
            test_files.append(audio_file)
            test_ids.append(row_id)
    
    print(f"Found {len(test_files)} test audio files")
    
    # Initialize models (one per fold)
    models = []
    for model_path in config["model_paths"]:
        model = Model(
            model_name=config["model_name"],
            num_classes=num_classes
        )
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))
        model = model.to(config["device"])
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models")
    
    # Create predictions dictionary
    predictions = {row_id: np.zeros(num_classes) for row_id in test_ids}
    
    # Process each test file
    for audio_file, row_id in tqdm(zip(test_files, test_ids), total=len(test_files)):
        # Extract the seconds from row_id
        seconds = int(row_id.split('_')[-1])
        
        # Process the audio file - here we can be more specific about which part to process
        audio_data, sr = librosa.load(str(audio_file), sr=config["sample_rate"])
        
        # Calculate start and end points based on seconds
        start_sample = max(0, seconds * sr - config["chunk_duration"] * sr // 2)
        end_sample = min(len(audio_data), start_sample + config["chunk_duration"] * sr)
        
        # Extract the relevant audio segment
        segment = audio_data[start_sample:end_sample]
        
        # Ensure consistent length
        if len(segment) < config["chunk_duration"] * sr:
            segment = np.pad(segment, (0, config["chunk_duration"] * sr - len(segment)), mode='constant')
        
        # Apply amplification
        segment = segment * config["amplification_factor"]
        
        # Compute mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            power=config["power"],
        )
        
        # Convert to dB and normalize
        mel_sp = librosa.power_to_db(mel_sp, ref=1)
        mel_sp = (mel_sp - mel_sp.min()) / (mel_sp.max() - mel_sp.min() + 1e-12)
        
        # Convert to tensor and add batch and channel dimensions
        tensor_input = torch.tensor(mel_sp, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        tensor_input = tensor_input.to(config["device"])
        
        # Get predictions from each model
        all_preds = []
        with torch.no_grad():
            for model in models:
                outputs = model(tensor_input)
                probs = torch.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())
        
        # Average predictions from all models
        avg_preds = np.mean(all_preds, axis=0)[0]
        predictions[row_id] = avg_preds
    
    # Create submission dataframe
    results = []
    for row_id in submission_df['row_id']:
        if row_id in predictions:
            row = {'row_id': row_id}
            row.update({class_name: float(pred) for class_name, pred in zip(classes, predictions[row_id])})
            results.append(row)
        else:
            # If no prediction available, use a default value
            row = {'row_id': row_id}
            row.update({class_name: 0.0048543689320388345 for class_name in classes})  # Using the default from sample submission
            results.append(row)
    
    # Create submission DataFrame
    submission = pd.DataFrame(results)
    
    # Ensure column order matches sample submission
    submission = submission[submission_df.columns]
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved to 'submission.csv' with shape {submission.shape}")
    
    
if __name__ == "__main__":
        run_inference()