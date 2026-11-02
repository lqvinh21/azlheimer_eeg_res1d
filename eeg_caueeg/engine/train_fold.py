import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from ..data.windows import Normalizer, make_eval_loader, make_train_loader
from ..models.resnet1d import ema_update, make_model
from ..utils.seed import seed_all
from .inference import metrics_of, record_logits


def run_fold(use_age, tr_ids, ho_ids, seed, store, test_ids, cfg, device, tag=''):
    """Huấn luyện một fold, không early stopping/chọn epoch; dùng EMA ở epoch cuối.

    Trả về dict: 'ho' (logit record held-out), 'test' (logit record test chính thức),
    'curve' [(epoch, train_loss_run, val_auc)] và 'hist' (loss/acc/AUC train-val theo epoch).
    """
    seed_all(seed)
    norm = Normalizer().fit(store, tr_ids, cfg)          # z-score riêng cho fold, từ phần train
    model = make_model(use_age, cfg, device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    loader = make_train_loader(store, tr_ids, norm, cfg)
    ho_loader = make_eval_loader(store, ho_ids, norm, cfg)
    te_loader = make_eval_loader(store, test_ids, norm, cfg)

    # tập con train cố định để đo train acc/AUC/loss ở chế độ eval (không aug, không dropout)
    sub = np.random.RandomState(seed).permutation(tr_ids)[:cfg.train_eval_n].tolist()
    tr_ev = make_eval_loader(store, sub, norm, cfg, cfg.curve_windows)
    ho_ev = make_eval_loader(store, ho_ids, norm, cfg, cfg.curve_windows)

    cw = None
    if cfg.class_weight:
        cnt = Counter(store.labels[s] for s in tr_ids)
        w = torch.tensor([1 / cnt[0], 1 / cnt[1]], dtype=torch.float32)
        cw = (w / w.sum() * 2).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, epochs=cfg.epochs,
                                                steps_per_epoch=len(loader), pct_start=0.1, anneal_strategy='cos')
    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    hist = {k: [] for k in ['epoch', 'tr_loss', 'va_loss', 'tr_acc', 'va_acc', 'tr_auc', 'va_auc', 'tr_loss_run']}
    curve = []
    for epoch in range(cfg.epochs):
        model.train(); losses = []
        for x, a, y in loader:
            x, a, y = x.to(device, non_blocking=True), a.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                loss = criterion(model(x, a).float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            ema_update(ema, model, cfg.ema_decay)
            losses.append(loss.item())

        last = epoch + 1 == cfg.epochs
        do_hist = (epoch + 1) % cfg.hist_every == 0 or last
        do_log = (epoch + 1) % cfg.log_every == 0 or last
        if do_hist or do_log:
            tl, ta, tu = metrics_of(record_logits(ema, tr_ev, device), store.labels)
            vl, va, vu = metrics_of(record_logits(ema, ho_ev, device), store.labels)
        if do_hist:
            for k, v in zip(hist, [epoch + 1, tl, vl, ta, va, tu, vu, float(np.mean(losses))]):
                hist[k].append(v)
            curve.append((epoch + 1, float(np.mean(losses)), vu))
        if do_log:
            print(f'{tag} ep {epoch + 1:02d} | loss tr {tl:.3f} va {vl:.3f} | acc tr {ta:.3f} va {va:.3f} '
                  f'| AUC tr {tu:.3f} va {vu:.3f}', flush=True)

    out = {'ho': record_logits(ema, ho_loader, device), 'test': record_logits(ema, te_loader, device),
           'curve': curve, 'hist': hist}
    del model, ema, opt, loader
    if use_cuda:
        torch.cuda.empty_cache()
    return out
