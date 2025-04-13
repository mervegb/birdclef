import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from bird_model import BirdCLEFModel
from unused.dataset import BirdCLEFDataset
import pandas as pd
from pytorch_lightning import seed_everything

# ======= CONFIG ===========
TRAIN_FOLDS_CSV = "data/train_folds.csv"  # CSV with 'kfold' column
FOLD = 0  # choose fold 0 as validation
BATCH_SIZE = 64
EPOCHS = 30
NUM_WORKERS = 4
SEED = 42

def main():
    seed_everything(SEED)
    
    # Read the CSV with folds
    df = pd.read_csv(TRAIN_FOLDS_CSV)
    # Use rows where kfold != FOLD for training and kfold == FOLD for validation
    train_df = df[df.kfold != FOLD].reset_index(drop=True)
    val_df = df[df.kfold == FOLD].reset_index(drop=True)
    
    # Define any transforms for the spectrogram images if needed (optional)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Force grayscale
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = BirdCLEFDataset(train_df, target_sample_rate=32000, max_time=5, image_transforms=transform)
    val_dataset = BirdCLEFDataset(val_df, target_sample_rate=32000, max_time=5, image_transforms=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Initialize model (number of classes inferred from the dataset)
    num_classes = len(train_df['primary_label'].unique())
    model = BirdCLEFModel(model_name="efficientnet_b0", num_classes=num_classes)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_auc",
        mode="max",
        filename="best-{epoch:02d}-{val_macro_auc:.4f}",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="val_macro_auc", patience=5, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        precision=16,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()