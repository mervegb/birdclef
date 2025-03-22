import os
from PIL import Image
from torch.utils.data import Dataset

class BirdSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

        # Create label-to-index map once
        self.labels = sorted({f.split("_", 1)[1].replace(".png", "") for f in self.files})
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        image_path = os.path.join(self.root_dir, filename)

        label_name = filename.split("_", 1)[1].replace(".png", "")
        label_idx = self.label_to_index[label_name]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx