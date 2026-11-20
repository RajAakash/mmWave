"""
Raymobtime s008/s009 data loader for vehicular mmWave beam prediction.

Input:  position_matrix_array (N, 23, 250) int8 — bird's-eye-view scatterer grid
Target: best_ray_array (N, 2) float32 — (AoD bin, AoA bin), binned into a
        2D angular codebook. We fold into a single beam index.

Returns (X, y):
    X: (N, 23*250) float32 — flattened + normalized position grid
    y: (N,)       int64   — beam index in [0, num_beams)
"""
import os
import pickle
import numpy as np

# Paths relative to the scripts/ directory (where you run from)
RAYMOB_DIR = './raymobtime_data/5GM-Beam-selection'
FILES = {
    'train': 'all_train_classification.npz',
    'test':  'all_test_classification.npz',
}


def load_split(split, data_dir=RAYMOB_DIR):
    """Load one split: returns (position_grids, raw_labels)."""
    path = os.path.join(data_dir, FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f'Raymobtime file not found: {path}')
    f = np.load(path, allow_pickle=True)
    X = f['position_matrix_array']         # (N, 23, 250) int8
    y_raw = f['best_ray_array']            # (N, 2) float32
    return X, y_raw


def labels_to_beam_index(y_raw, n_az_bins=16):
    """
    Convert (col0_bin, col1_bin) -> single beam index.
    Combined into N*M space by beam = col0 * n_az_bins + col1 (clipped).
    """
    a = np.clip(y_raw[:, 0].astype(int), 0, n_az_bins - 1)
    b = np.clip(y_raw[:, 1].astype(int), 0, n_az_bins - 1)
    beam = a * n_az_bins + b                        # in [0, n_az_bins**2)
    # Compact the label space: map actual seen values to 0..K-1
    unique = np.unique(beam)
    remap = {v: i for i, v in enumerate(unique)}
    beam_compact = np.array([remap[v] for v in beam], dtype=np.int64)
    return beam_compact, unique


def extract_position_features(position_grids):
    """
    Flatten position grids into real-valued feature vectors and normalize.
    Shape: (N, 23, 250) -> (N, 23*250) float32
    """
    X = position_grids.astype(np.float32).reshape(position_grids.shape[0], -1)
    # Normalize to zero mean, unit variance per dimension (ignore constant cols)
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    X = (X - mu) / (sigma + 1e-8)
    return X.astype(np.float32)


def build_raymobtime_dataset(cache_dir='./cache'):
    """Returns dict with train/test (X, y) splits."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'raymobtime_s008.pkl')
    if os.path.exists(cache_file):
        print(f'[cache hit] {cache_file}')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print('Loading Raymobtime train split ...')
    Xtr_raw, ytr_raw = load_split('train')
    print(f'  raw: X={Xtr_raw.shape}, y={ytr_raw.shape}')
    print('Loading Raymobtime test split ...')
    Xte_raw, yte_raw = load_split('test')
    print(f'  raw: X={Xte_raw.shape}, y={yte_raw.shape}')

    # Combine then re-bin so the label space is consistent across splits
    all_labels = np.concatenate([ytr_raw, yte_raw])
    _, unique = labels_to_beam_index(all_labels)
    print(f'  num unique beams: {len(unique)}')

    # Per-split beam index using the shared mapping
    def _map(y_raw):
        a = np.clip(y_raw[:, 0].astype(int), 0, 15)
        b = np.clip(y_raw[:, 1].astype(int), 0, 15)
        combined = a * 16 + b
        remap = {v: i for i, v in enumerate(unique)}
        return np.array([remap[v] for v in combined], dtype=np.int64)

    ytr = _map(ytr_raw)
    yte = _map(yte_raw)

    # Feature extraction (fit normalization on train, apply to test)
    Xtr_flat = Xtr_raw.astype(np.float32).reshape(Xtr_raw.shape[0], -1)
    Xte_flat = Xte_raw.astype(np.float32).reshape(Xte_raw.shape[0], -1)
    mu = Xtr_flat.mean(axis=0, keepdims=True)
    sigma = Xtr_flat.std(axis=0, keepdims=True)
    Xtr = ((Xtr_flat - mu) / (sigma + 1e-8)).astype(np.float32)
    Xte = ((Xte_flat - mu) / (sigma + 1e-8)).astype(np.float32)

    result = {
        'train': (Xtr, ytr),
        'test':  (Xte, yte),
        'num_beams': len(unique),
        'feat_dim': Xtr.shape[1],
    }
    print(f'  train: {Xtr.shape}, test: {Xte.shape}, beams: {len(unique)}')
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f'  cached to {cache_file}')
    return result


if __name__ == '__main__':
    d = build_raymobtime_dataset()
    Xtr, ytr = d['train']
    Xte, yte = d['test']
    print(f'\ntrain X: {Xtr.shape}, y: {ytr.shape}, beams used: {len(np.unique(ytr))}')
    print(f'test  X: {Xte.shape}, y: {yte.shape}, beams used: {len(np.unique(yte))}')
    print(f'label distribution (train, top 10):')
    vals, counts = np.unique(ytr, return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1])[:10]:
        print(f'  beam {v}: {c} ({100*c/len(ytr):.1f}%)')
