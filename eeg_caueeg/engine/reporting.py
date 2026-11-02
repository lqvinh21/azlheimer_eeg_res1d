import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score, roc_curve

from ..config import CLASS_NAMES
from .inference import auc_of


def youden(y, z):
    fpr, tpr, thr = roc_curve(y, z)
    return float(thr[np.argmax(tpr - fpr)])


def boot_auc(y, z, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], z[i]))
    return np.percentile(out, [2.5, 97.5])


def boot_diff(y, z1, z2, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], z1[i]) - roc_auc_score(y[i], z2[i]))
    return np.percentile(out, [2.5, 97.5])


def stats(y, z, thr):
    pred = (z >= thr).astype(int)
    return {'bal_acc': balanced_accuracy_score(y, pred), 'sens_AD': pred[y == 1].mean(),
            'spec_HC': 1 - pred[y == 0].mean()}


def collect_scores(results, store, age_oof, age_test):
    """OOF: mỗi record DEV được dự đoán bởi model không thấy nó.
    TESTZ: trung bình logit của các model fold trên test chính thức."""
    oof_all, test_all = {'age_only': age_oof}, {'age_only': age_test}
    for v, res in results.items():
        oof = {}
        for r in res:
            oof.update(r['ho'])
        assert set(oof) == set(store.dev)
        oof_all[v] = oof
        test_all[v] = {s: float(np.mean([r['test'][s] for r in res])) for s in store.test}
    return oof_all, test_all


def summary_table(OOF, TESTZ, results, store):
    y_dev, y_te = store.y_dev(), store.y_test()
    rows = []
    for v in OOF:
        zo = np.array([OOF[v][s] for s in store.dev]); zt = np.array([TESTZ[v][s] for s in store.test])
        thr = youden(y_dev, zo)
        lo, hi = boot_auc(y_dev, zo); tlo, thi = boot_auc(y_te, zt)
        so0, sot, st0, stt = stats(y_dev, zo, 0.0), stats(y_dev, zo, thr), stats(y_te, zt, 0.0), stats(y_te, zt, thr)
        fold_aucs = [auc_of(r['ho'], store.labels) for r in results[v]] if v in results else []
        rows.append({
            'model': v, 'thr_OOF': round(thr, 2),
            'OOF_AUC': f'{roc_auc_score(y_dev, zo):.3f} [{lo:.3f},{hi:.3f}]',
            'fold_AUC': f'{np.mean(fold_aucs):.3f}+-{np.std(fold_aucs):.3f}' if fold_aucs else '-',
            'OOF_balacc@0': round(so0['bal_acc'], 3), 'OOF_balacc@thr': round(sot['bal_acc'], 3),
            'TEST_AUC': f'{roc_auc_score(y_te, zt):.3f} [{tlo:.3f},{thi:.3f}]',
            'TEST_balacc@0': round(st0['bal_acc'], 3), 'TEST_balacc@thr': round(stt['bal_acc'], 3),
            'TEST_sens@thr': round(stt['sens_AD'], 3), 'TEST_spec@thr': round(stt['spec_HC'], 3),
        })
    return pd.DataFrame(rows).set_index('model').T


def paired_differences(OOF, store):
    """Chênh lệch OOF AUC (kèm bootstrap 95% CI): EEG+tuổi vs chỉ tuổi, có tuổi vs không tuổi."""
    y_dev = store.y_dev()
    lines = []
    if 'age' in OOF:
        zo = np.array([OOF['age'][s] for s in store.dev]); za = np.array([OOF['age_only'][s] for s in store.dev])
        d = roc_auc_score(y_dev, zo) - roc_auc_score(y_dev, za); lo, hi = boot_diff(y_dev, zo, za)
        lines.append(f'OOF AUC(EEG+tuổi) - AUC(chỉ tuổi) = {d:+.3f}  (95% CI [{lo:+.3f}, {hi:+.3f}])')
    if 'age' in OOF and 'no_age' in OOF:
        z1 = np.array([OOF['age'][s] for s in store.dev]); z2 = np.array([OOF['no_age'][s] for s in store.dev])
        d = roc_auc_score(y_dev, z1) - roc_auc_score(y_dev, z2); lo, hi = boot_diff(y_dev, z1, z2)
        lines.append(f'OOF AUC(có tuổi) - AUC(không tuổi) = {d:+.3f}  (95% CI [{lo:+.3f}, {hi:+.3f}])')
    return lines


def pick_variant(OOF):
    return 'age' if 'age' in OOF else [k for k in OOF if k != 'age_only'][0]


def error_analysis(OOF, TESTZ, store):
    """Lỗi theo nhóm bệnh (symptom), dùng ngưỡng Youden từ OOF, gộp OOF + test."""
    v = pick_variant(OOF)
    zo = np.array([OOF[v][s] for s in store.dev]); thr = youden(store.y_dev(), zo)
    recs = []
    for s in store.dev + store.test:
        z = OOF[v][s] if s in OOF[v] else TESTZ[v][s]
        recs.append({'serial': s, 'part': 'dev' if s in OOF[v] else 'test', 'y': store.labels[s], 'z': z,
                     'wrong': int((z >= thr) != store.labels[s]), 'age': store.ages[s],
                     'tags': store.symptoms[s].split('|')})
    df = pd.DataFrame(recs)
    print(f'Biến thể: {v} | thr = {thr:.2f} | lỗi toàn bộ: {df.wrong.mean():.3f}')
    ex = df.explode('tags')
    tab = ex.groupby('tags').agg(n=('wrong', 'size'), err=('wrong', 'mean'), age=('age', 'mean')).round(3)
    tab = tab.sort_values('n', ascending=False)
    by_class = df.groupby('y').wrong.agg(['size', 'mean']).round(3)
    print(tab.to_string())
    print('\nLỗi theo lớp thật:'); print(by_class.to_string())
    return df, tab, by_class


def oof_classification_report(OOF, store):
    """Báo cáo phân loại trên OOF với ngưỡng Youden. Trả về (variant, z_oof, pred_oof, thr)."""
    variant = 'age' if 'age' in OOF else 'no_age'
    y_dev = store.y_dev()
    z_oof = np.array([OOF[variant][s] for s in store.dev])
    thr_opt = youden(y_dev, z_oof)
    pred_oof = (z_oof >= thr_opt).astype(int)
    print(f'  OOF report - Model: {variant}')
    print(f'  Ngưỡng logit tối ưu (Youden): {thr_opt:.3f}')
    print('Cross-Validation')
    print(classification_report(y_dev, pred_oof, target_names=CLASS_NAMES, digits=4))
    return variant, z_oof, pred_oof, thr_opt
