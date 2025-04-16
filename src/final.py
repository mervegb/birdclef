import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")

# Kaggle metric utilities
import kaggle_metric_utilities
import sklearn.metrics
import pandas.api.types

########################################
# CONFIG #
########################################

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
    "spec_width": 640,  # Not used in processing; we work with variable widths.
    "model_name": "tf_efficientnet_b0",
    "data_train_csv": "data/train.csv",
    "audio_dir": "data/raw",
    "random_state": 42,
}

training_config = {
    "epochs": 20,
    "num_folds": 3,
    "batch_size": 16,
    "val_batch_size": 32,
    "target_col": "primary_label",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "freeze_epochs": 3,
}

########################################
# KAGGLE METRIC UTILITIES #
########################################

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    if not pd.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype for x in submission.columns
                      if not pd.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f"Invalid submission data types found: {bad_dtypes}")
    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0
    return kaggle_metric_utilities.safe_call_score(
        sklearn.metrics.roc_auc_score,
        solution[scored_columns].values,
        submission[scored_columns].values,
        average="macro",
    )

def cal_score(labels, preds, label_mapper):
    """Calculate ROC-AUC score using Kaggle metrics"""
    # Convert to appropriate format for scoring
    columns = list(label_mapper.keys())
    
    # Ensure prediction shape matches number of classes
    num_classes = len(label_mapper)
    if preds[0].shape[1] != num_classes:
        print(f"Warning: Reshaping predictions from {preds[0].shape[1]} to {num_classes} classes")
        # Extract only the columns we need from predictions
        preds = [p[:, :num_classes] for p in preds]
    
    # Concatenate all predictions and labels
    labels_arr = np.concatenate(labels)
    preds_arr = np.concatenate(preds)
    
    # Create DataFrame for scoring
    labels_df = pd.DataFrame(labels_arr, columns=columns)
    pred_df = pd.DataFrame(preds_arr, columns=columns)
    labels_df['id'] = np.arange(len(labels_df))
    pred_df['id'] = np.arange(len(pred_df))
    
    return score(labels_df, pred_df, row_id_column_name='id')

########################################
# DATASET DEFINITION #
########################################

def get_random_overlapping_window(data, min_len):
    """Extract a random window from data with a stride of 50%."""
    stride = int(min_len * 0.5)
    n_windows = (len(data) - min_len) // stride + 1
    if n_windows > 1:
        window_idx = np.random.randint(0, n_windows)
        start = window_idx * stride
    else:
        start = 0
    return data[start:start+min_len]

class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, mode="train", augment=False, global_label_mapper=None):
        self.df = df
        self.mode = mode
        self.audio_dir = Path(audio_dir)
        self.augment = augment
        self.file_paths = self.df["filename"].apply(lambda x: self.audio_dir / x).tolist()
        
        if global_label_mapper is not None:
            # Use provided global label mapper
            self.label_mapper = global_label_mapper
            self.reverse_label_mapper = {idx: label for label, idx in self.label_mapper.items()}
            self.labels = self.df["primary_label"].map(self.label_mapper).tolist()
        elif "primary_label" in self.df.columns:
            # Create a new label mapper based on this dataset
            unique_labels = sorted(self.df["primary_label"].unique())
            self.label_mapper = {label: idx for idx, label in enumerate(unique_labels)}
            self.reverse_label_mapper = {idx: label for label, idx in self.label_mapper.items()}
            self.labels = self.df["primary_label"].map(self.label_mapper).tolist()
        else:
            self.labels = None
            self.label_mapper = {}
            self.reverse_label_mapper = {}

    def __len__(self):
        return len(self.file_paths)

    def process(self, audio_path):
        audio_data, sr = librosa.load(audio_path, sr=config["sample_rate"])
        
        # Apply augmentation if enabled
        if self.augment and self.mode == "train":
            # Time shift
            shift_factor = np.random.uniform(-0.2, 0.2)
            shift_samples = int(shift_factor * len(audio_data))
            if shift_samples > 0:
                audio_data = np.pad(audio_data, (shift_samples, 0), mode='constant')[:-shift_samples]
            elif shift_samples < 0:
                audio_data = np.pad(audio_data, (0, -shift_samples), mode='constant')[-shift_samples:]
                
            # Random amplitude scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            audio_data = audio_data * scale_factor
            
            # Optional: Add some random noise
            noise_factor = np.random.uniform(0, 0.01)
            noise = np.random.normal(0, noise_factor, len(audio_data))
            audio_data = audio_data + noise
        
        # Amplify the signal
        data = audio_data * config["amplification_factor"]
        
        # Ensure consistent length
        min_len = config["chunk_duration"] * sr
        if len(data) < min_len:
            # Pad with repetition if audio is too short
            data = np.tile(data, int(np.ceil(min_len / len(data))))
        elif len(data) > min_len:
            # Extract a random window if audio is too long
            data = get_random_overlapping_window(data, min_len)
        
        # Clip to exact length
        data = data[:min_len]
        
        # Compute mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=data,
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
        
        # Return the variable width mel spectrogram
        return mel_sp

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        spectrogram = self.process(audio_path)
        x = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0)  # [1, 128, variable_width]
        
        if self.mode == "train" and self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

########################################
# MODEL DEFINITION #
########################################

class Model(nn.Module):
    def __init__(self, model_name=config["model_name"], num_classes=None):
        super().__init__()
        # Create the base model without the classifier
        self.base_model = timm.create_model(
            model_name, 
            pretrained=True, 
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
        
        # Freeze state flag
        self.freeze_base = True
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def unfreeze_base(self):
        """Unfreeze the base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.freeze_base = False
        print("Base model unfrozen for fine-tuning")

    def forward(self, x):
        # Handle base model with or without gradients
        if self.freeze_base:
            with torch.no_grad():
                features = self.base_model.forward_features(x)
        else:
            features = self.base_model.forward_features(x)
        
        # Apply classifier head
        pooled = self.global_pool(features).flatten(1)
        x = self.dropout1(pooled)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

########################################
# CUSTOM COLLATE FN #
########################################

def custom_collate_fn(batch):
    """
    Pads variable-width mel spectrograms to the maximum width in the batch.
    """
    xs, ys = zip(*batch)
    max_width = max(x.shape[-1] for x in xs)
    xs_padded = []
    for x in xs:
        pad_width = max_width - x.shape[-1]
        if pad_width > 0:
            x = nn.functional.pad(x, (0, pad_width))
        xs_padded.append(x)
    xs_stacked = torch.stack(xs_padded, dim=0)
    ys = torch.tensor(ys)
    return xs_stacked, ys

########################################
# TRAINING UTILITIES #
########################################

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    pred_train = []
    label_train = []
    
    pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * x_batch.size(0)
        
        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)
        pred_train.append(probs.detach().cpu().numpy())
        
        # Create one-hot encoded labels
        one_hot_y = torch.nn.functional.one_hot(y_batch, num_classes=outputs.shape[1]).float()
        label_train.append(one_hot_y.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(loader.dataset)
    
    return epoch_loss, pred_train, label_train

def validate(model, loader, criterion, device):
    """Run validation"""
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    pred_val = []
    label_val = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Validation", leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss_val = criterion(outputs, y_batch)
            
            # Track metrics
            running_val_loss += loss_val.item() * x_batch.size(0)
            
            # Calculate accuracy
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            
            # Store predictions and labels
            pred_val.append(probs.detach().cpu().numpy())
            one_hot_y_val = torch.nn.functional.one_hot(y_batch, num_classes=outputs.shape[1]).float()
            label_val.append(one_hot_y_val.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = running_val_loss / len(loader.dataset)
    val_acc = correct / total if total > 0 else 0
    
    return val_loss, val_acc, pred_val, label_val

########################################
# MAIN CODE #
########################################

if __name__ == "__main__":
    # 1. Read full data
    print("Loading data...")
    data_df = pd.read_csv(config["data_train_csv"])
    df = data_df.reset_index(drop=True)
    
    # Get all classes and update config
    all_classes = sorted(df[training_config["target_col"]].unique())
    num_total_classes = len(all_classes)
    config["num_classes"] = num_total_classes
    
    # Create a global label mapper for consistent class mapping
    global_label_mapper = {label: idx for idx, label in enumerate(all_classes)}
    
    # Print dataset statistics
    print(f"Dataset size: {len(df)} samples")
    print(f"Number of classes: {num_total_classes}")
    print(f"Class distribution (top 5):\n{df['primary_label'].value_counts().head()}")

    # 2. Stratified K-fold splitting
    skf = StratifiedKFold(
        n_splits=training_config["num_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )
    df["kfold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[training_config["target_col"]])):
        df.loc[val_idx, "kfold"] = fold

    # Save CSV with 'kfold' column
    new_csv_path = "data/train_kfold.csv"
    df.to_csv(new_csv_path, index=False)
    print(f"Saved CSV with kfold column at {new_csv_path}")

    # 3. Train/validation loop for each fold
    for fold in range(training_config["num_folds"]):
        print(f"\n{'='*20} FOLD {fold} {'='*20}")
        train_df = df[df["kfold"] != fold].reset_index(drop=True)
        val_df = df[df["kfold"] == fold].reset_index(drop=True)
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")

        # Class count and weight calculation for WeightedRandomSampler
        class_counts = train_df[training_config["target_col"]].value_counts().to_dict()
        max_count = max(class_counts.values())
        class_weights = {cls: max_count/count for cls, count in class_counts.items()}
        sample_weights = [class_weights[cls] for cls in train_df[training_config["target_col"]]]
        
        # Cap weights to prevent extreme oversampling
        max_weight = 10.0
        sample_weights = [min(w, max_weight) for w in sample_weights]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(train_df), 
            replacement=True
        )

        # Create datasets with global label mapper
        train_ds = BirdCLEFDataset(
            train_df, 
            config["audio_dir"], 
            mode="train", 
            augment=True,
            global_label_mapper=global_label_mapper
        )
        
        val_ds = BirdCLEFDataset(
            val_df, 
            config["audio_dir"], 
            mode="train", 
            augment=False,
            global_label_mapper=global_label_mapper
        )

        # Create data loaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=training_config["batch_size"],
            sampler=sampler, 
            num_workers=2, 
            drop_last=True, 
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=training_config["val_batch_size"],
            shuffle=False, 
            num_workers=2, 
            drop_last=False, 
            collate_fn=custom_collate_fn
        )

        # 4. Create model, optimizer, criterion
        model = Model(num_classes=num_total_classes).to(training_config["device"])
        
        # Freeze base model parameters initially
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.freeze_base = True
        
        # Set up optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': training_config["lr"] * 0.1},
            {'params': model.fc1.parameters()},
            {'params': model.fc2.parameters()},
            {'params': model.bn1.parameters()}
        ], lr=training_config["lr"], weight_decay=training_config["weight_decay"])
        
        # Set up criterion with class weights
        criterion = nn.CrossEntropyLoss()
        
        # Use cosine annealing scheduler with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup for first 10% of steps
        warmup_steps = int(0.1 * training_config["epochs"] * len(train_loader))
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine annealing for remaining steps
        cosine_steps = int(0.9 * training_config["epochs"] * len(train_loader))
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_steps,
            eta_min=training_config["lr"] * 0.01
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        # Track best model
        best_auc = 0
        best_model_path = None

        # 5. Epoch loop
        for epoch in range(training_config["epochs"]):
            print(f"\nEpoch {epoch+1}/{training_config['epochs']}")
            
            # Unfreeze base model after specified number of epochs
            if epoch == training_config["freeze_epochs"] and model.freeze_base:
                model.unfreeze_base()
            
            # Train for one epoch
            train_loss, pred_train, label_train = train_epoch(
                model, train_loader, optimizer, criterion, 
                training_config["device"], epoch
            )
            
            # Calculate training AUC
            try:
                auc_train = cal_score(label_train, pred_train, global_label_mapper)
                train_metrics = f"Train Loss: {train_loss:.4f} | Train AUC: {auc_train:.4f}"
            except Exception as e:
                train_metrics = f"Train Loss: {train_loss:.4f} | Train AUC Error: {str(e)}"
            
            print(f"Fold {fold} | Epoch {epoch+1} | {train_metrics}")
            
            # Validate
            val_loss, val_acc, pred_val, label_val = validate(
                model, val_loader, criterion, training_config["device"]
            )
            
            # Calculate validation AUC
            try:
                auc_val = cal_score(label_val, pred_val, global_label_mapper)
                val_metrics = f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {auc_val:.4f}"
            except Exception as e:
                auc_val = 0
                val_metrics = f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC Error: {str(e)}"
            
            print(f"Fold {fold} | Epoch {epoch+1} | {val_metrics}")
            
            # Update learning rate scheduler - for per-batch update
            # scheduler.step()
            
            # Checkpointing: Save best model based on validation AUC
            if isinstance(auc_val, float) and auc_val > best_auc:
                best_auc = auc_val
                # Create directories if they don't exist
                checkpoints_dir = Path("checkpoints")
                checkpoints_dir.mkdir(exist_ok=True)
                
                # Save model
                best_model_path = checkpoints_dir / f"model_fold_{fold}_epoch_{epoch+1}_auc_{auc_val:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f" -> Saved best model with Val AUC: {auc_val:.4f}")
        
        print(f"\nTraining completed for fold {fold}. Best validation AUC: {best_auc:.4f}")
        
        # 6. Evaluate final model on validation set
        print("\nEvaluating best model on validation set...")
        
        # Load best model
        if best_model_path and Path(best_model_path).exists():
            model = Model(num_classes=num_total_classes).to(training_config["device"])
            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            
            # Run full validation
            val_loss, val_acc, pred_val, label_val = validate(
                model, val_loader, criterion, training_config["device"]
            )
            
            try:
                final_auc = cal_score(label_val, pred_val, global_label_mapper)
                print(f"Final evaluation - Fold {fold}: Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {final_auc:.4f}")
            except Exception as e:
                print(f"Error in final evaluation: {str(e)}")
        else:
            print(f"Warning: No best model checkpoint found for fold {fold}")
    
    print("\nTraining completed for all folds!")