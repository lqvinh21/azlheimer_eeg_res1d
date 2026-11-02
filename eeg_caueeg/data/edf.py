import os

import numpy as np

from ..config import ALIAS, CHANNELS_19


def norm_name(name):
    name = name.replace('EEG ', '').split('-')[0].strip()
    name = name[0].upper() + name[1:].lower() if name.upper() in ('FZ', 'CZ', 'PZ') else name.capitalize()
    return ALIAS.get(name, name)


def load_recording(path, cache_dir, cfg):
    """Band-pass 0.5-45 Hz, resample về cfg.sfreq, 19 kênh theo thứ tự CHANNELS_19.

    Kết quả được cache thành .npy (tên: <file>.edf.<sfreq>.npy). Trả về None nếu thiếu kênh
    hoặc bản ghi quá ngắn. Z-score không tính ở đây mà tính theo từng fold.
    """
    cache = os.path.join(cache_dir, os.path.basename(path) + f'.{cfg.sfreq}.npy')
    if os.path.isfile(cache):
        return np.load(cache)
    try:
        import mne
    except ImportError as err:
        raise ImportError('Cần cài mne để đọc EDF: pip install mne') from err
    mne.set_log_level('ERROR')
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    mapping = {}
    for ch in raw.ch_names:
        n = norm_name(ch)
        if n in CHANNELS_19 and n not in mapping.values():
            mapping[ch] = n
    if len(mapping) != len(CHANNELS_19):
        return None
    raw.pick(list(mapping)); raw.rename_channels(mapping); raw.reorder_channels(CHANNELS_19)
    raw.filter(0.5, 45.0, verbose=False)
    raw.resample(cfg.sfreq, verbose=False)
    x = raw.get_data().astype(np.float32)
    if x.shape[1] < cfg.skip + cfg.win:
        return None
    np.save(cache, x)
    return x
