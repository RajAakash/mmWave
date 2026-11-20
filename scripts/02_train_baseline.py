"""
Step 2: Train baseline MLP on O1 scenario.
Reproduces the Alrabeiah & Alkhateeb (2020) setup.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pickle

from data.deepmimo_loader import build_paired_dataset
from models.networks import build_model
from utils.train_eval import train_model, evaluate_model


def split(X, y, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    X, y = build_paired_dataset('MIAMI')
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    print(f'Train={len(Xtr)}, Val={len(Xva)}, Test={len(Xte)}')
    print(f'Feature dim: {X.shape[1]}, Beams: {int(y.max()+1)}')

    num_beams = int(y.max() + 1)
    for model_name in ('mlp', 'cnn'):
        print(f'\n--- Training {model_name.upper()} ---')
        model = build_model(model_name, input_dim=X.shape[1], num_beams=num_beams)
        model, history = train_model(model, Xtr, ytr, Xva, yva,
                                     epochs=50, device=device, verbose=True)
        test_metrics = evaluate_model(model, Xte, yte, device=device)
        print(f'TEST {model_name}: {test_metrics}')

        os.makedirs('./results', exist_ok=True)
        torch.save(model.state_dict(), f'./results/baseline_{model_name}.pt')
        with open(f'./results/baseline_{model_name}_history.pkl', 'wb') as f:
            pickle.dump({'history': history, 'test': test_metrics}, f)
