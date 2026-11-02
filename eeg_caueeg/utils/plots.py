import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def _finish(fig, out_dir, name, show):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print('Đã lưu', path)
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_and_epoch_auc(OOF, TESTZ, results, store, out_dir, show=False):
    """ROC (OOF, test chính thức) và AUC held-out theo epoch (mỗi đường = 1 fold)."""
    y_dev, y_te = store.y_dev(), store.y_test()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for v in OOF:
        zo = np.array([OOF[v][s] for s in store.dev]); fpr, tpr, _ = roc_curve(y_dev, zo)
        ax[0].plot(fpr, tpr, label=f'{v} (AUC {roc_auc_score(y_dev, zo):.3f})')
        zt = np.array([TESTZ[v][s] for s in store.test]); fpr, tpr, _ = roc_curve(y_te, zt)
        ax[1].plot(fpr, tpr, label=f'{v} (AUC {roc_auc_score(y_te, zt):.3f})')
    for a_, t in zip(ax[:2], [f'ROC - OOF ({len(y_dev)} record)', f'ROC - test chính thức ({len(y_te)} record)']):
        a_.plot([0, 1], [0, 1], 'k:', lw=.8); a_.set_title(t); a_.set_xlabel('FPR'); a_.set_ylabel('TPR')
        a_.legend(loc='lower right'); a_.grid(alpha=.3)
    for v, res in results.items():
        for k, r in enumerate(res):
            ep, _, au = zip(*r['curve'])
            ax[2].plot(ep, au, color='C0' if v == 'age' else 'C1', alpha=.6, label=v if k == 0 else None)
    ax[2].set_title('AUC held-out theo epoch (mỗi đường = 1 fold)'); ax[2].set_xlabel('epoch')
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    _finish(fig, out_dir, 'roc_and_epoch_auc.png', show)


def plot_train_val(results, variant, out_dir, show_folds=True, show=False):
    """Một hình cho mỗi biến thể: train-vs-val Accuracy, AUC, Loss (trung bình các fold, ±std)."""
    res = [r for r in results.get(variant, []) if 'hist' in r]
    if not res:
        print(f'[{variant}] chưa có lịch sử huấn luyện'); return
    H = [r['hist'] for r in res]
    ep = np.array(H[0]['epoch'])
    panels = [('acc', 'Accuracy', 'Accuracy'), ('auc', 'AUC', 'ROC-AUC'), ('loss', 'Loss', 'Cross-entropy')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (key, title, ylab) in zip(axes, panels):
        for split, color, name in (('tr', 'tab:blue', 'Train'), ('va', 'tab:red', 'Val')):
            M = np.array([h[f'{split}_{key}'] for h in H])
            if show_folds:
                for m in M:
                    ax.plot(ep, m, color=color, alpha=.15, lw=.8)
            mu, sd = M.mean(0), M.std(0)
            ax.plot(ep, mu, color=color, lw=2.2, label=f'{name} (cuối: {mu[-1]:.3f})')
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=.15)
        ax.set_title(f'Train vs Val {title}'); ax.set_xlabel('Epoch'); ax.set_ylabel(ylab)
        ax.grid(alpha=.3); ax.legend(loc='best')
    tag = 'EEG + tuổi' if variant == 'age' else 'chỉ EEG (không tuổi)'
    fig.suptitle(f'Biến thể {variant} - {tag} | trung bình {len(res)} fold (val = fold held-out)', y=1.02)
    fig.tight_layout()
    _finish(fig, out_dir, f'curves_{variant}.png', show)


def plot_oof_diagnostics(y_dev, z_oof, pred_oof, thr, variant, out_dir, show=False):
    """Confusion matrix và phân bố logit trên OOF."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_dev, pred_oof, display_labels=['Bình thường', 'AD'], cmap='Blues', ax=ax[0], colorbar=False)
    ax[0].set_title(f'Confusion Matrix - OOF ({variant}, n={len(y_dev)})')
    ax[1].hist(z_oof[y_dev == 0], bins=25, alpha=0.6, label='Bình thường (HC)', color='C0', density=True)
    ax[1].hist(z_oof[y_dev == 1], bins=25, alpha=0.6, label='Alzheimer (AD)', color='C1', density=True)
    ax[1].axvline(thr, color='red', linestyle='--', lw=1.5, label=f'Ngưỡng ({thr:.2f})')
    ax[1].set_title('Phân bố Logit phân tách nhóm (OOF)')
    ax[1].set_xlabel('Logit (AD - HC)'); ax[1].set_ylabel('Mật độ phân phối')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, out_dir, 'oof_diagnostics.png', show)
