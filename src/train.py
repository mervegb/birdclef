import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from bird_cnn import BirdCNN        # For group1 (BCE loss)
from bird_focal_cnn import BirdFocalCNN  # For group2 (Focal loss)

# ========== CONFIG ========== #
BATCH_SIZE = 96
EPOCHS = 50
IMG_SIZE = (224, 224)
NUM_WORKERS = os.cpu_count()

# ========== TRANSFORMS ========== #
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ========== FUNCTION ========== #
def train_group(group_name):
    print(f"\n🚀 Training model for {group_name.upper()}...\n")
    data_dir = f"data/processed/spectrograms_grouped/{group_name}"
    model_weights_path = f"{group_name}_weights.pt"
    model_full_path = f"{group_name}_model_full.pt"

    dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

    num_classes = len(dataset.classes)
    print(f"📊 Classes in {group_name}: {num_classes}")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Choose model
    if group_name == "group1":
        model = BirdCNN(num_classes=num_classes)
    else:
        model = BirdFocalCNN(num_classes=num_classes)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_auc",
        filename=f"{group_name}-epoch{{epoch:02d}}-auc{{val_macro_auc:.4f}}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        mode="max"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_macro_auc",
        patience=5,
        mode="max",
        verbose=True
    )

    trainer = Trainer(
        precision=16,
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, val_loader)

    # Save final model
    #torch.save(model.state_dict(), model_weights_path)
    #torch.save(model, model_full_path)
    print(f"✅ [{group_name.upper()}] Model saved to: {model_full_path}")


# ========== MAIN LOOP ========== #
if __name__ == "__main__":
    for group in ["group1", "group2"]:
        train_group(group)