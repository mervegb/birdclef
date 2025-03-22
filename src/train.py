import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from dataset.bird_dataset import BirdSpectrogramDataset
from models.bird_cnn import BirdCNN

# 🔧 Configs
DATA_PATH = "data/processed/spectrograms"
BATCH_SIZE = 32
NUM_CLASSES = 100  # Update this after checking the actual number

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and Dataloader
dataset = BirdSpectrogramDataset(DATA_PATH, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = BirdCNN(num_classes=len(dataset.label_to_index))

# Trainer
trainer = Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader)