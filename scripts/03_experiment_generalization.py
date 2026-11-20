"""
Angle 1: Cross-scene generalization across 3 cities x 3 BSs each.
9 scenes total -> 81-cell generalization matrix.

Output: ./results/generalization.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from data.deepmimo_loader import build_per_bs_datasets
from models.networks import build_model
from utils.train_eval import train_model, evaluate_model
from utils.env_distance import environmental_distance


CITIES = ['MIAMI', 'SEATTLE', 'DALLAS']


def split(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * 0.85)
    return (X[idx[:n_tr]], y[idx[:n_tr]]), (X[idx[n_tr:]], y[idx[n_tr:]])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load every city's per-BS datasets, merge into one dict of 9 scenes
    datasets = {}
    for city in CITIES:
        print(f'\nLoading {city} ...')
        city_ds = build_per_bs_datasets(city)
        datasets.update(city_ds)

    scene_names = list(datasets.keys())
    print(f'\nAll scenes ({len(scene_names)}): {scene_names}')

    num_beams = max(int(y.max() + 1) for _, y in datasets.values())
    feat_dim = next(iter(datasets.values()))[0].shape[1]
    print(f'num_beams={num_beams}, feat_dim={feat_dim}')

    # Train one model per scene
    trained = {}
    for scene in scene_names:
        X_full, y_full = datasets[scene]
        (Xtr, ytr), (Xva, yva) = split(X_full, y_full)
        print(f'\n=== Training on {scene} (n={len(Xtr)}) ===')
        model = build_model('mlp', feat_dim, num_beams)
        model, _ = train_model(model, Xtr, ytr, Xva, yva,
                               epochs=40, device=device, verbose=False)
        trained[scene] = model

    # Cross-evaluate every model on every scene
    rows = []
    for train_scene, model in trained.items():
        X_train_src, _ = datasets[train_scene]
        for test_scene in scene_names:
            X_test, y_test = datasets[test_scene]
            metrics = evaluate_model(model, X_test, y_test, device=device)
            dist = environmental_distance(X_train_src, X_test)
            # Classify regime: same-BS, cross-BS-same-city, or cross-city
            same_bs = train_scene == test_scene
            same_city = train_scene.split('_')[0] == test_scene.split('_')[0]
            if same_bs:
                regime = 'same-BS'
            elif same_city:
                regime = 'cross-BS-same-city'
            else:
                regime = 'cross-city'
            print(f'  {train_scene:16s} -> {test_scene:16s}  '
                  f'[{regime:20s}]  dist={dist:.3f}  top1={metrics["top1"]:.3f}')
            rows.append({
                'train_scene': train_scene,
                'test_scene': test_scene,
                'regime': regime,
                'env_distance': dist,
                **metrics,
            })

    df = pd.DataFrame(rows)
    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/generalization.csv', index=False)
    print('\nSaved ./results/generalization.csv')

    # Headline summary: mean Top-1 per regime
    print('\n--- Mean Top-1 by regime (the headline table of your paper) ---')
    print(df.groupby('regime')['top1'].agg(['mean', 'std', 'count']).to_string())