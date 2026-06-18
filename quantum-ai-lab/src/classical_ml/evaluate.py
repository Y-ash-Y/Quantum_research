import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


def cross_validate_model(model, X, y, n_splits=5):
    """k-fold cross-validation returning accuracy/AUC means and std."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, aucs = [], []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        try:
            proba = model.predict_proba(X_val)[:, 1]
        except AttributeError:
            proba = model.decision_function(X_val)

        accs.append(accuracy_score(y_val, preds))
        aucs.append(roc_auc_score(y_val, proba))

    return {
        'acc_mean': np.mean(accs),
        'acc_std':  np.std(accs),
        'auc_mean': np.mean(aucs),
        'auc_std':  np.std(aucs),
    }


def plot_roc_curves(models_dict, X_test, y_test, ax=None):
    """Plot ROC curves for all models on the same axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for name, model in models_dict.items():
        try:
            scores = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            scores = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Sub-threshold Attack Detection')
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
