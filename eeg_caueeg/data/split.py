import json
import os

import pandas as pd

from ..config import AD_SYMPTOMS


def _build_frame(items, edf_root, ad_only):
    rows = []
    for it in items:
        label = int(it['class_label'])
        if label == 1:                                  # bỏ MCI
            continue
        symptoms = set(it.get('symptom', []))
        if label == 2 and ad_only and not (symptoms & AD_SYMPTOMS):
            continue
        serial = int(str(it['serial']))
        path = os.path.join(edf_root, f'{serial:05d}.edf')
        if not os.path.isfile(path):
            continue
        rows.append({'serial': serial, 'age': float(it['age']), 'target': 0 if label == 0 else 1,
                     'symptom': '|'.join(sorted(symptoms)), 'path': path})
    return pd.DataFrame(rows)


def read_official_split(split_file, edf_root, ad_only=False):
    """Đọc split chính thức của CAUEEG, nhãn HC=0, AD/dementia=1 (bỏ MCI)."""
    assert os.path.isfile(split_file), split_file
    assert os.path.isdir(edf_root), edf_root
    with open(split_file, 'r', encoding='utf-8') as f:
        official = json.load(f)
    print('class_label_to_name:', official.get('class_label_to_name'))

    frames = {
        'train': _build_frame(official['train_split'], edf_root, ad_only),
        'val': _build_frame(official['validation_split'], edf_root, ad_only),
        'test': _build_frame(official['test_split'], edf_root, ad_only),
    }
    for name, fr in frames.items():
        assert fr.serial.is_unique and fr.target.nunique() == 2, name
        print(f'{name:5s} | EDF={len(fr):3d} | HC={(fr.target == 0).sum():3d} | AD={(fr.target == 1).sum():3d}')
    sets = {k: set(v.serial) for k, v in frames.items()}
    assert not (sets['train'] & sets['val']) and not (sets['train'] & sets['test']) and not (sets['val'] & sets['test'])
    print('No-overlap OK (theo serial)')
    return frames
