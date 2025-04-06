# train.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from bird_cnn import BirdCLEFModel
import cv2

# ======= CONFIG ======= #
TRAIN_DIR = "data/processed/spectrograms/train"
VAL_DIR = "data/processed/spectrograms/val"
BATCH_SIZE = 64
EPOCHS = 30
NUM_WORKERS = 4
SEED = 42

# ======= TRAINING FUNCTION ======= #
def main():
    
    img = cv2.imread("data/processed/spectrograms/train/21211/XC882648_seg0.png", cv2.IMREAD_UNCHANGED)
    print('IMG SHAPE MERVE', img.shape)  # Expect (128, width) or (128, width, 1) if grayscale
    
    seed_everything(SEED)

    # ======= TRANSFORMS ======= #
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Force grayscale
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ======= DATALOADERS ======= #
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    val_dataset = ImageFolder(root=VAL_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ======= MODEL ======= #
    NUM_CLASSES = len(train_dataset.classes)
    model = BirdCLEFModel(model_name="efficientnet_b0", num_classes=NUM_CLASSES)

    # ======= CALLBACKS ======= #
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_auc",
        mode="max",
        filename="best-{epoch:02d}-{val_macro_auc:.4f}",
        save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor="val_macro_auc", patience=5, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ======= TRAINING ======= #
    trainer = Trainer(
        max_epochs=EPOCHS,
        precision=16,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True
    )

    trainer.fit(model, train_loader, val_loader)

    print("✅ Training complete!")

# ======= MAIN ======= #
if __name__ == "__main__":
    main()
