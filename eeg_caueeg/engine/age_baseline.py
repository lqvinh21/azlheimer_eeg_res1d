import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config import AGE_MEAN, AGE_STD
from .inference import auc_of


def _age_feat(store, ids):
    return np.array([[(store.ages[s] - AGE_MEAN) / AGE_STD] for s in ids])


def age_only_baseline(store, folds):
    """Baseline chỉ dùng tuổi (logistic regression), cùng fold với mô hình EEG."""
    age_oof = {}
    for tr, ho in folds:
        lr = LogisticRegression().fit(_age_feat(store, tr), [store.labels[s] for s in tr])
        for s, z in zip(ho, lr.decision_function(_age_feat(store, ho))):
            age_oof[s] = float(z)
    lr = LogisticRegression().fit(_age_feat(store, store.dev), store.y_dev())
    age_test = {s: float(z) for s, z in zip(store.test, lr.decision_function(_age_feat(store, store.test)))}
    print('AGE-ONLY | OOF AUC', round(auc_of(age_oof, store.labels), 3),
          '| test AUC', round(auc_of(age_test, store.labels), 3))
    return age_oof, age_test
