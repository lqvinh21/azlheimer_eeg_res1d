import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(store, n_folds=5, seed=42, group_of=None):
    """StratifiedGroupKFold trên DEV (train+val chính thức).

    `group_of[serial]` = group của record. Mặc định mỗi record là một group (tương đương
    stratified K-fold). Nếu có bảng record -> patient_id thì truyền vào đây, không cần sửa gì khác.
    """
    dev = store.dev
    group_of = group_of or {s: s for s in dev}
    dev_arr = np.array(dev)
    y_dev = store.y_dev()
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, ho) in enumerate(skf.split(dev_arr, y_dev, np.array([group_of[s] for s in dev]))):
        folds.append((dev_arr[tr].tolist(), dev_arr[ho].tolist()))
        print(f'fold {k + 1}: train {len(tr)} (AD {y_dev[tr].sum()}) | held-out {len(ho)} (AD {y_dev[ho].sum()})')
    assert sorted(sum((f[1] for f in folds), [])) == sorted(dev)       # mỗi record held-out đúng một lần
    for tr, ho in folds:
        assert not (set(group_of[s] for s in tr) & set(group_of[s] for s in ho))
    return folds
