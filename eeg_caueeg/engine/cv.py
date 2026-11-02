import os
import pickle
import time

from .inference import auc_of
from .train_fold import run_fold


def run_cross_validation(store, folds, variants, cfg, device, out_dir, seed=42):
    """Chạy CV cho từng biến thể. Mỗi fold lưu vào out_dir/<variant>_fold<k>.pkl;
    chạy lại sẽ tiếp tục từ fold còn thiếu (file không có 'hist' sẽ bị huấn luyện lại)."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for variant in variants:
        results[variant] = []
        for k, (tr, ho) in enumerate(folds):
            path = os.path.join(out_dir, f'{variant}_fold{k}.pkl')
            res = None
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    res = pickle.load(f)
                if 'hist' not in res:
                    print(f'[{variant}] fold {k + 1}: file cũ không có lịch sử epoch -> huấn luyện lại')
                    res = None
                else:
                    print(f'[{variant}] fold {k + 1}: đã có kết quả, bỏ qua huấn luyện')
            if res is None:
                t0 = time.time()
                print(f'=== {variant} | fold {k + 1}/{len(folds)} | train {len(tr)} | held-out {len(ho)}')
                res = run_fold(variant == 'age', tr, ho, seed + k, store, store.test, cfg, device,
                               tag=f'[{variant} f{k + 1}]')
                with open(path, 'wb') as f:
                    pickle.dump(res, f)
                print(f'    xong sau {(time.time() - t0) / 60:.1f} phút | held-out AUC {auc_of(res["ho"], store.labels):.3f} '
                      f'| test AUC (model này) {auc_of(res["test"], store.labels):.3f}')
            results[variant].append(res)
    return results
