import argparse
from pathlib import Path

import pandas as pd
import torch

from eeg_caueeg.config import VARIANTS, TrainConfig
from eeg_caueeg.data.folds import make_folds
from eeg_caueeg.data.records import RecordStore
from eeg_caueeg.data.split import read_official_split
from eeg_caueeg.engine.age_baseline import age_only_baseline
from eeg_caueeg.engine.cv import run_cross_validation
from eeg_caueeg.engine.reporting import (collect_scores, error_analysis, oof_classification_report,
                                         paired_differences, summary_table)
from eeg_caueeg.models.resnet1d import check_models
from eeg_caueeg.utils.plots import plot_oof_diagnostics, plot_roc_and_epoch_auc, plot_train_val
from eeg_caueeg.utils.seed import seed_all


def parse_args():
    p = argparse.ArgumentParser(description="1D-ResNet AD vs HC trên CAUEEG (K-fold theo group, EMA, ablation tuổi)")
    p.add_argument("--data-root", required=True,
                   help="Thư mục CAUEEG chứa dementia-no-overlap.json và signal/edf/")
    p.add_argument("--split-file", default=None, help="Mặc định: <data-root>/dementia-no-overlap.json")
    p.add_argument("--edf-root", default=None, help="Mặc định: <data-root>/signal/edf")
    p.add_argument("--output-dir", default="./caueeg_output")
    p.add_argument("--cache-dir", default=None,
                   help="Cache .npy (band-pass + resample). Mặc định: <output-dir>/cache. "
                        "Trỏ vào cache cũ (tên <file>.edf.128.npy) để khỏi đọc lại EDF")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="'age' = EEG + tuổi, 'no_age' = chỉ EEG")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hist-every", type=int, default=1, help="Ghi lịch sử train/val mỗi N epoch")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ad-only", action="store_true", help="Chỉ giữ dementia thuộc nhóm AD (ad/load/eoad)")
    p.add_argument("--show", action="store_true", help="Hiện đồ thị trên màn hình (mặc định chỉ lưu PNG)")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    cv_dir = out_dir / "cv"
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else out_dir / "cache"
    for d in (out_dir, cv_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate,
                      hist_every=args.hist_every, num_workers=args.num_workers)

    # 1. Split chính thức + nạp EDF
    data_root = Path(args.data_root).expanduser()
    split_file = args.split_file or str(data_root / "dementia-no-overlap.json")
    edf_root = args.edf_root or str(data_root / "signal" / "edf")
    frames = read_official_split(split_file, edf_root, ad_only=args.ad_only)
    store = RecordStore()
    store.load(frames, str(cache_dir), cfg)
    store.summary(cfg)
    check_models(cfg, device)

    # 2. Fold + cross-validation
    folds = make_folds(store, args.n_folds, args.seed)
    results = run_cross_validation(store, folds, args.variants, cfg, device, str(cv_dir), seed=args.seed)

    # 3. Baseline chỉ tuổi + báo cáo OOF / test
    age_oof, age_test = age_only_baseline(store, folds)
    OOF, TESTZ = collect_scores(results, store, age_oof, age_test)
    table = summary_table(OOF, TESTZ, results, store)
    print(table.to_string())
    for line in paired_differences(OOF, store):
        print(line)
    table.to_csv(out_dir / "results_summary.csv", encoding="utf-8-sig")
    pd.DataFrame({"serial": store.dev,
                  "label": [store.labels[s] for s in store.dev],
                  **{f"oof_logit_{v}": [OOF[v][s] for s in store.dev] for v in OOF}}
                 ).to_csv(out_dir / "oof_scores.csv", index=False)

    # 4. Đồ thị
    plot_roc_and_epoch_auc(OOF, TESTZ, results, store, str(out_dir), args.show)
    for v in args.variants:
        plot_train_val(results, v, str(out_dir), show=args.show)

    # 5. Phân tích lỗi + báo cáo phân loại OOF
    _, tab, _ = error_analysis(OOF, TESTZ, store)
    tab.to_csv(out_dir / "error_by_symptom.csv", encoding="utf-8-sig")
    variant, z_oof, pred_oof, thr = oof_classification_report(OOF, store)
    plot_oof_diagnostics(store.y_dev(), z_oof, pred_oof, thr, variant, str(out_dir), args.show)
    print(f"All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
