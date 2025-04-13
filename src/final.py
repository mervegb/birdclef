import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm  # For progress bars

# Kaggle metric utilities (as in your script)
import kaggle_metric_utilities
import sklearn.metrics
import pandas.api.types

########################################
#              CONFIG                #
########################################

config = {
    "sample_rate": 32000,
    "amplification_factor": 1024,
    "chunk_duration": 10,       # seconds per sample
    "n_fft": 1024,
    "hop_length": 500,
    "n_mels": 128,
    "fmin": 50,
    "fmax": 16000,
    "power": 2.0,
    "spec_width": 640,          # fixed time frames for mel spectrogram
    "model_name": "tf_efficientnet_b0",
    "num_classes": 206,
    "data_train_csv": "data/train.csv",
    "audio_dir": "data/raw",
    "sample_size": 1000,
    "random_state": 42,
}

training_config = {
    "epochs": 10,             # Increased from 1 to 10 epochs for longer training
    "num_folds": 3,
    "batch_size": 8,
    "val_batch_size": 16,
    "target_col": "primary_label",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
}

########################################
#       KAGGLE METRIC UTILITIES        #
########################################

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Macro-averaged ROC-AUC score ignoring classes with no positive labels (Kaggle BirdCLEF style).
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pd.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pd.api.types.is_numeric_dtype(submission[x])}
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

def cal_score(labels, preds):
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    
    labels_df = pd.DataFrame(labels > 0.5, columns = list(label_mapper.keys()))
    pred_df = pd.DataFrame(preds, columns=list(label_mapper.keys()))
    
    label_df['id'] = np.arange(len(labels_df))
    pred_df['id'] = np.arange(len(pred_df))
    
    return score(label_df, pred_df, row_id_column_name='id')

########################################
#         DATASET DEFINITION           #
########################################

class BirdCLEFDataset(Dataset):
    def __init__(self, df, audio_dir, mode="train"):
        self.df = df
        self.mode = mode
        self.audio_dir = Path(audio_dir)
        self.file_paths = self.df["filename"].apply(lambda x: self.audio_dir / x).tolist()

        if "primary_label" in self.df.columns:
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
        data = audio_data * config["amplification_factor"]
        min_len = config["chunk_duration"] * sr

        if len(data) < min_len:
            data = np.tile(data, int(np.ceil(min_len / len(data))))
        if len(data) > min_len:
            leftover = len(data) - min_len
            data = data[leftover // 2 : leftover // 2 + min_len]
        else:
            data = data[:min_len]

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
        mel_sp = librosa.power_to_db(mel_sp, ref=1)
        mel_sp = (mel_sp - mel_sp.min()) / (mel_sp.max() - mel_sp.min() + 1e-12)

        if mel_sp.shape[1] >= config["spec_width"]:
            mel_sp = mel_sp[:, :config["spec_width"]]
        else:
            pad = config["spec_width"] - mel_sp.shape[1]
            mel_sp = np.pad(mel_sp, ((0, 0), (0, pad)), mode="constant")
        return mel_sp

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        spectrogram = self.process(audio_path)
        x = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0)  # [1, 128, spec_width]
        if self.mode == "train" and self.labels is not None:
            return x, self.labels[idx]
        return x

########################################
#           MODEL DEFINITION           #
########################################

class Model(nn.Module):
    def __init__(self, model_name=config["model_name"]):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=model_name,
            num_classes=config["num_classes"],
            pretrained=False,
            in_chans=1,
        )

    def forward(self, x):
        return self.base_model(x)

########################################
#              MAIN CODE               #
########################################

if __name__ == "__main__":
    # 1. Read and subset data.
    data_df = pd.read_csv(config["data_train_csv"])
    df = data_df.sample(1000, random_state=config["random_state"]).reset_index(drop=True)

    # 2. Stratified K-fold splitting.
    skf = StratifiedKFold(
        n_splits=training_config["num_folds"],
        shuffle=True,
        random_state=config["random_state"]
    )
    df["kfold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[training_config["target_col"]])):
        df.loc[val_idx, "kfold"] = fold

    # (Optional) Save CSV with kfold column.
    new_csv_path = "data/train_kfold.csv"
    df.to_csv(new_csv_path, index=False)
    print(f"Saved CSV with kfold column at {new_csv_path}")

    # 3. Train/validation loop for each fold.
    for fold in range(training_config["num_folds"]):
        print(f"\n==========  FOLD {fold}  ==========")
        train_df = df[df["kfold"] != fold].reset_index(drop=True)
        val_df = df[df["kfold"] == fold].reset_index(drop=True)

        train_ds = BirdCLEFDataset(train_df, config["audio_dir"], mode="train")
        val_ds = BirdCLEFDataset(val_df, config["audio_dir"], mode="train")

        train_loader = DataLoader(
            train_ds, batch_size=training_config["batch_size"],
            shuffle=True, num_workers=2, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=training_config["val_batch_size"],
            shuffle=False, num_workers=2, drop_last=False
        )

        # Initializing model
        model = Model(name=config['model_name']).to(training_config["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
        criterion = nn.CrossEntropyLoss()

        best_auc = 0

        # Per fold training
        for epoch in range(training_config["epochs"]):
            print(f"\nEpoch {epoch} / {training_config['epochs']-1}")
            model.train()
            
            pred_train = []
            label_train = []
            
            running_loss = 0.0

            # --- TRAINING ---
            for x, y in tqdm(train_loader, desc="Training"):
                x = x.to(training_config["device"])
                y = y.to(training_config["device"])

                # Converting to one hot encoding  
                y_one_hot = nn.functional(
                    y,
                    num_classes = config['num_classes']
                ).float()      

                optimizer.zero_grad()
                outputs = model(x)
               
                loss = criterion(outputs, y)
                loss.backward() # backpropagation
                optimizer.step()

                running_loss += loss.item() 
                props = torch.softmax(outputs, dim=1)
                pred_train.append(props.detach().cpu().numpy())
                label_train.append()
                
            
    # --- VALIDATION ---
    model.eval()
    
    pred_val = []
    label_val = []
    
    running_val_loss = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x = x.to(training_config["device"])
            y = y.to(training_config["device"])

            # Converting to one hot encoding  
            y_one_hot = nn.functional(
                y,
                num_classes = config['num_classes']
            ).float()      

            outputs = model(x)
            
            loss = criterion(outputs, y)

            running_val_loss += loss.item() 
            props = torch.softmax(outputs, dim=1)
            pred_val.append(props.detach().cpu().numpy())
            label_val.append()
    
    
    # --- Computing AUC and Loss ---
    auc_train = cal_score(label_train, pred_train)
    auc_val = cal_score(label_val, pred_val)
    
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = running_loss / len(val_loader)

    print(f'[Fold]: {fold} | [EPOCH]: {epoch} | Loss: {avg_train_loss:4f} | Val_Loss: {avg_val_loss:4f}')    
    print(f'[Fold]: {fold} |  [EPOCH]: {epoch} | Train AUC: {auc_train:4f} | Val AUC: {auc_val:4f}')    
    
    if best_auc <= auc_val and epoch < 5:
        best_auc = auc_val
        torch.save(model.state_dict(), f"fold_{fold}_epoch_{epoch}_effnetB0_val_auc_{auc_val}_val_loss_{avg_val_loss}.pth" )
        
        

     




          
