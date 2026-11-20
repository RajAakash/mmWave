"""
Raymobtime noise-robustness sweep. Same methodology as Miami (script 04)
so results can be compared side-by-side in the paper.

Output: ./results/raymobtime_robustness.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from data.raymobtime_loader import build_raymobtime_dataset
from models.networks import build_model
from utils.train_eval import train_model, evaluate_model


NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


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
    Xtr_full, ytr_full = d['train']
    Xte, yte = d['test']
    num_beams = d['num_beams']
    feat_dim = d['feat_dim']

    (Xtr, ytr), (Xva, yva) = split_train_val(Xtr_full, ytr_full)

    rows = []
    for training in ('clean', 'noise-augmented'):
        print(f'\n=== Training ({training}) on Raymobtime ===')
        train_noise = 0.0 if training == 'clean' else 0.15
        model = build_model('mlp', feat_dim, num_beams)
        model, _ = train_model(model, Xtr, ytr, Xva, yva,
                               epochs=40, device=device,
                               noise_std=train_noise, verbose=False)
        for noise in NOISE_LEVELS:
            metrics = evaluate_model(model, Xte, yte, device=device, noise_std=noise)
            rows.append({'dataset': 'raymobtime', 'training': training,
                         'noise_std': noise, **metrics})
            print(f'  {training}  noise={noise:.2f}  '
                  f'top1={metrics["top1"]:.3f}  top5={metrics["top5"]:.3f}')

    df = pd.DataFrame(rows)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/raymobtime_robustness.csv', index=False)
    print('\nSaved ./results/raymobtime_robustness.csv')
