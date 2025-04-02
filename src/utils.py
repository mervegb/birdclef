import numpy as np
import torch
import librosa

def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def pad_spectrogram(S, target_width):
    current_width = S.shape[1]
    if current_width >= target_width:
        return S[:, :target_width]
    pad_total = target_width - current_width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(S, ((0, 0), (pad_left, pad_right)), mode='constant')

def is_high_energy(y, sr, threshold_db):
    rms = librosa.feature.rms(y=y)[0]
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)
    return np.mean(energy_db) > -threshold_db