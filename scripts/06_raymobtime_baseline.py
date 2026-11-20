"""
Raymobtime baseline: train MLP on train split, evaluate on test split.
Reports Top-K accuracy. This is the within-dataset number for your paper.

Output: ./results/raymobtime_baseline.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from data.raymobtime_loader import build_raymobtime_dataset
from models.networks import build_model
from utils.train_eval import train_model, evaluate_model


def split_train_val(X, y, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_va = int(len(X) * val_frac)
    va, tr = idx[:n_va], idx[n_va:]
    return (X[tr], y[tr]), (X[va], y[va])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    d = build_raymobtime_dataset()
    (Xtr_full, ytr_full) = d['train']
    (Xte, yte) = d['test']
    num_beams = d['num_beams']
    feat_dim = d['feat_dim']

    (Xtr, ytr), (Xva, yva) = split_train_val(Xtr_full, ytr_full)
    print(f'Train={len(Xtr)}, Val={len(Xva)}, Test={len(Xte)}')
    print(f'feat_dim={feat_dim}, num_beams={num_beams}')

    print('\n--- Training MLP on Raymobtime ---')
    model = build_model('mlp', feat_dim, num_beams)
    model, history = train_model(model, Xtr, ytr, Xva, yva,
                                 epochs=40, device=device, verbose=True)

    test_metrics = evaluate_model(model, Xte, yte, device=device)
    print(f'\nTEST on Raymobtime s008-test: {test_metrics}')

    os.makedirs('./results', exist_ok=True)
    pd.DataFrame([{'dataset': 'raymobtime', **test_metrics}]).to_csv(
        './results/raymobtime_baseline.csv', index=False)

    # Save model for reuse in noise-robustness script
    torch.save(model.state_dict(), './results/raymobtime_mlp.pt')
    print('Saved ./results/raymobtime_baseline.csv and raymobtime_mlp.pt')
