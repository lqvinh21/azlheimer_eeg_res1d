import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def record_logits(net, loader, device):
    """Mỗi record = trung bình logit (logit_AD - logit_HC) trên các cửa sổ đều nhau."""
    net.eval(); acc = {}
    for x, a, y, s in loader:
        lg = net(x.to(device, non_blocking=True), a.to(device, non_blocking=True)).float()
        d = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        for si, di in zip(s.numpy(), d):
            acc.setdefault(int(si), []).append(float(di))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def auc_of(zdict, labels):
    ids = list(zdict)
    return roc_auc_score([labels[s] for s in ids], [zdict[s] for s in ids])


def metrics_of(zdict, labels):
    """loss / acc / AUC ở mức record (logit trung bình các cửa sổ), ngưỡng logit = 0."""
    ids = list(zdict)
    y = np.array([labels[s] for s in ids]); z = np.array([zdict[s] for s in ids])
    loss = float(np.mean(np.logaddexp(0, -(2 * y - 1) * z)))   # = cross-entropy 2 lớp, không label smoothing
    acc = float(((z >= 0).astype(int) == y).mean())
    return loss, acc, float(roc_auc_score(y, z))
