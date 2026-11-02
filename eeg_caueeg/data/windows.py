import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import AGE_MEAN, AGE_STD


class Normalizer:
    """Z-score theo kênh, tính từ phần train của TỪNG fold."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, store, serials, cfg):
        sub = np.concatenate([store.signals[s][:, cfg.skip::10] for s in serials], axis=1)
        self.mean = sub.mean(1, keepdims=True).astype(np.float32)
        self.std = (sub.std(1, keepdims=True) + 1e-6).astype(np.float32)
        return self


class TrainWindows(Dataset):
    def __init__(self, store, serials, norm, cfg):
        self.store, self.norm, self.cfg = store, norm, cfg
        self.items = [s for s in serials for _ in range(cfg.windows_per_record)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cfg = self.cfg
        s = self.items[i]; x = self.store.signals[s]
        start = np.random.randint(cfg.skip, x.shape[1] - cfg.win + 1)
        w = (x[:, start:start + cfg.win] - self.norm.mean) / self.norm.std
        w = w * np.random.uniform(1 - cfg.aug_scale, 1 + cfg.aug_scale) \
            + np.random.normal(0, cfg.aug_noise, w.shape).astype(np.float32)
        w = np.clip(w, -cfg.clip, cfg.clip).astype(np.float32)
        age = (self.store.ages[s] - AGE_MEAN) / AGE_STD + np.random.normal(0, 0.05)
        return torch.from_numpy(w), torch.tensor(age, dtype=torch.float32), self.store.labels[s]


class EvalWindows(Dataset):
    def __init__(self, store, serials, n_windows, norm, cfg):
        self.store, self.norm, self.cfg = store, norm, cfg
        self.items = []
        for s in serials:
            hi = store.signals[s].shape[1] - cfg.win
            for st in np.linspace(cfg.skip, hi, n_windows):
                self.items.append((s, int(st)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cfg = self.cfg
        s, st = self.items[i]
        w = np.clip((self.store.signals[s][:, st:st + cfg.win] - self.norm.mean) / self.norm.std,
                    -cfg.clip, cfg.clip).astype(np.float32)
        age = torch.tensor((self.store.ages[s] - AGE_MEAN) / AGE_STD, dtype=torch.float32)
        return torch.from_numpy(w), age, self.store.labels[s], s


def make_train_loader(store, serials, norm, cfg):
    return DataLoader(TrainWindows(store, serials, norm, cfg), batch_size=cfg.batch_size, shuffle=True,
                      drop_last=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())


def make_eval_loader(store, serials, norm, cfg, n_windows=None):
    return DataLoader(EvalWindows(store, serials, n_windows or cfg.eval_windows, norm, cfg),
                      batch_size=cfg.eval_batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
