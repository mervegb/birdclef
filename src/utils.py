import random
import torch
import torch.nn as nn
import numpy as np
import torchaudio

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_spec_augmenter():
    """SpecAugment for mel spectrograms"""
    return nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
        torchaudio.transforms.TimeMasking(time_mask_param=20),
    )
    
       
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




def mixup_data(x, y, alpha=1.0):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def cutmix_data(x, y, alpha=1.0):
    """Applies CutMix augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random coordinates
    h, w = x.size()[2], x.size()[3]
    r_x = np.random.randint(w)
    r_y = np.random.randint(h)
    r_w = int(np.sqrt(1 - lam) * w)
    r_h = int(np.sqrt(1 - lam) * h)

    x1 = max(r_x - r_w // 2, 0)
    y1 = max(r_y - r_h // 2, 0)
    x2 = min(r_x + r_w // 2, w)
    y2 = min(r_y + r_h // 2, h)

    # Apply patch
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def custom_collate_fn(batch):
    """
    Pads variable-width mel spectrograms to the maximum width in the batch.
    Handles both (x, y) tuples and x tensors.
    """
    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
        max_width = max(x.shape[-1] for x in xs)
        xs_padded = []
        for x in xs:
            pad_width = max_width - x.shape[-1]
            if pad_width > 0:
                x = nn.functional.pad(x, (0, pad_width))
            xs_padded.append(x)
        xs_stacked = torch.stack(xs_padded, dim=0)
        
        # Handle different types of y values (int labels or one-hot vectors)
        if torch.is_tensor(ys[0]) and ys[0].dim() > 0:
            ys = torch.stack(ys)
        else:
            ys = torch.tensor(ys)
        return xs_stacked, ys
    else:
        xs = batch
        max_width = max(x.shape[-1] for x in xs)
        xs_padded = []
        for x in xs:
            pad_width = max_width - x.shape[-1]
            if pad_width > 0:
                x = nn.functional.pad(x, (0, pad_width))
            xs_padded.append(x)
        return torch.stack(xs_padded, dim=0)