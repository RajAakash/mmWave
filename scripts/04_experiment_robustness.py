"""
Angle 5: Noise robustness of sub-6 GHz → mmWave beam prediction.

Experiment:
  - Train TWO models on O1:
      (a) 'clean'           — no noise during training
      (b) 'noise-augmented' — Gaussian noise injected into inputs each epoch
  - Evaluate both on O1 test set with varying test-time noise levels.

Output: ./results/robustness.csv

The 'noise-augmented training helps' result (Contribution 4 of the paper) comes
from comparing the two curves.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from data.deepmimo_loader import build_paired_dataset
from models.networks import build_model
from utils.train_eval import train_model, evaluate_model


NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def split3(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    n_tr, n_va = int(n*0.7), int(n*0.15)
    return ((X[idx[:n_tr]], y[idx[:n_tr]]),
            (X[idx[n_tr:n_tr+n_va]], y[idx[n_tr:n_tr+n_va]]),
            (X[idx[n_tr+n_va:]], y[idx[n_tr+n_va:]]))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, y = build_paired_dataset('MIAMI')
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split3(X, y)
    num_beams = int(y.max() + 1)

    rows = []
    for training in ('clean', 'noise-augmented'):
        print(f'\n=== Training ({training}) ===')
        train_noise = 0.0 if training == 'clean' else 0.15
        model = build_model('mlp', X.shape[1], num_beams)
        model, _ = train_model(model, Xtr, ytr, Xva, yva,
                               epochs=40, device=device,
                               noise_std=train_noise, verbose=False)
        # Sweep test-time noise levels
        for noise in NOISE_LEVELS:
            # Evaluate with noise injected at test time
            metrics = evaluate_model(model, Xte, yte, device=device, noise_std=noise)
            rows.append({'training': training, 'noise_std': noise, **metrics})
            print(f'  {training}  noise={noise:.2f}  '
                  f'top1={metrics["top1"]:.3f}  top5={metrics["top5"]:.3f}')

    df = pd.DataFrame(rows)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/robustness.csv', index=False)
    print('\nSaved ./results/robustness.csv')
