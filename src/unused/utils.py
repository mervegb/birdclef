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

# ========== HELPERS ========== #
def get_audio_info(filepath):
    with sf.SoundFile(filepath) as f:
        return {"frames": f.frames, "sr": f.samplerate, "duration": f.frames / f.samplerate}


def compute_melspec(y, sr, n_mels, fmin, fmax):
    S = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax or sr // 2)
    return lb.power_to_db(S).astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = 255 * (np.clip(X, _min, _max) - _min) / (_max - _min)
        return V.astype(np.uint8)
    return np.zeros_like(X, dtype=np.uint8)


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        n_repeats = length // len(y)
        remainder = length % len(y)
        y = np.concatenate([y] * n_repeats + [y[:remainder]])
    elif len(y) > length:
        start = start or (np.random.randint(len(y) - length) if is_train else 0)
        y = y[start:start + length]
    return y