import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .edf import load_recording


class RecordStore:
    """Giữ toàn bộ tín hiệu và metadata theo `serial` (thay cho các biến global trong notebook)."""

    def __init__(self):
        self.signals, self.labels, self.ages = {}, {}, {}
        self.symptoms, self.origin, self.kept = {}, {}, {}

    @property
    def dev(self):
        """train + val chính thức, dùng cho cross-validation."""
        return self.kept['train'] + self.kept['val']

    @property
    def test(self):
        """test no-overlap chính thức, chỉ để chấm cuối."""
        return self.kept['test']

    def load(self, split_frames, cache_dir, cfg):
        os.makedirs(cache_dir, exist_ok=True)
        failed = Counter()
        t0 = time.time()
        for name, fr in split_frames.items():
            self.kept[name] = []
            for row in tqdm(fr.itertuples(index=False), total=len(fr), desc=f'Load {name}'):
                x = load_recording(row.path, cache_dir, cfg)
                if x is None:
                    failed[name] += 1
                    continue
                s = row.serial
                self.signals[s] = x
                self.labels[s] = int(row.target)
                self.ages[s] = row.age
                self.symptoms[s] = row.symptom
                self.origin[s] = name
                self.kept[name].append(s)
            print(name, 'kept', len(self.kept[name]), dict(Counter(self.labels[s] for s in self.kept[name])))
        print('Seconds:', round(time.time() - t0, 1), '| failed:', dict(failed))
        print('DEV =', len(self.dev), '| TEST =', len(self.test))

    def summary(self, cfg):
        rows = []
        for name in ('train', 'val', 'test'):
            ids = self.kept[name]
            a = np.array([self.ages[s] for s in ids]); y = np.array([self.labels[s] for s in ids])
            mins = np.mean([self.signals[s].shape[1] / cfg.sfreq / 60 for s in ids])
            rows.append({'split': name, 'n': len(ids), 'HC': int((y == 0).sum()), 'AD': int((y == 1).sum()),
                         'age_HC': f'{a[y == 0].mean():.1f}+-{a[y == 0].std():.1f}',
                         'age_AD': f'{a[y == 1].mean():.1f}+-{a[y == 1].std():.1f}',
                         'minutes_mean': round(mins, 1)})
        print(pd.DataFrame(rows).to_string(index=False))
        print('Bài báo: tuổi toàn dataset 70.77 +- 9.90, độ dài mean 13.34 +- 2.83 phút.')
        tags = Counter(t for s in self.dev + self.test for t in self.symptoms[s].split('|') if t)
        print('Nhóm symptom (đếm record):', dict(tags.most_common()))

    def y_dev(self):
        return np.array([self.labels[s] for s in self.dev])

    def y_test(self):
        return np.array([self.labels[s] for s in self.test])
