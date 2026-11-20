"""
DeepMIMO v4 data loader for sub-6 GHz / mmWave paired channels.

v4 API uses dm.download() + dm.load(), auto-fetches from the online database.
Scenario names changed — we use paired ASU campus / Miami city scenarios
that are shipped at both 3.5 GHz (sub-6) and 28 GHz (mmWave).

Outputs:
    X: real-valued sub-6 GHz channel features  (N_samples, F)
    y: optimal mmWave beam indices             (N_samples,)
"""
import os
import numpy as np
import pickle

try:
    import deepmimo as dm
except ImportError:
    dm = None
    print("WARNING: deepmimo package not installed. Run: pip install deepmimo")


# -------------------------------------------------------------------
# v4 scene presets — each scene has a sub-6 and a mmWave scenario name
# -------------------------------------------------------------------
# These scenarios exist in the public v4 database and are co-located
# (same ray-tracing environment, two carrier frequencies).
SCENE_PRESETS = {
    'MIAMI':   {'sub6': 'city_6_miami_3p5',    'mmwave': 'city_6_miami_28'},
    'SEATTLE': {'sub6': 'city_17_seattle_3p5', 'mmwave': 'city_17_seattle_28'},
    'DALLAS':  {'sub6': 'city_8_dallas_3p5',   'mmwave': 'city_8_dallas_28'},
}

def ensure_downloaded(scenario_name):
    """Download a scenario if it isn't already cached locally."""
    if dm is None:
        raise ImportError("deepmimo package required. pip install deepmimo")
    try:
        dm.download(scenario_name)
    except Exception as e:
        # Some versions skip silently if already cached, some raise.
        # Just report and continue — dm.load() will fail loudly if truly missing.
        print(f'  (download note: {e})')


def load_scenario(scenario_name, max_users=None):
    """
    Load a DeepMIMO v4 scenario and return channels + user positions.

    Returns:
        channels: complex array, shape (n_ue, n_rx_ant, n_tx_ant, n_subcarriers)
                  (exact shape may vary with dataset config)
        positions: user (x, y, z) positions, shape (n_ue, 3), or None
    """
    ensure_downloaded(scenario_name)
    print(f'  loading {scenario_name} ...')
    dataset = dm.load(scenario_name)

    # The v4 dataset API evolved between betas — try the common access patterns.
    # Try to boost subcarriers if the dataset exposes a setter
    if hasattr(dataset, 'set_params'):
        try:
            dataset.set_params(num_subcarriers=64)
        except Exception:
            pass
    channels = None
    for attr in ('channel', 'channels'):
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if callable(val):
                val = val()
            channels = val
            break
    if channels is None and hasattr(dataset, 'compute_channels'):
        # Request richer channel: more subcarriers and bandwidth give the sub-6 input
        # enough spatial/frequency info to actually predict mmWave beams.
        try:
            channels = dataset.compute_channels(num_subcarriers=64, bandwidth=0.05)
        except TypeError:
            # Older v4 API: try without kwargs
            channels = dataset.compute_channels()

    if channels is None:
        raise RuntimeError(
            f'Could not find channels on dataset for {scenario_name}. '
            f'Available attrs: {[a for a in dir(dataset) if not a.startswith("_")][:30]}'
        )

    channels = np.asarray(channels)
    if channels.ndim == 5:
        n_txsets, n_ue = channels.shape[0], channels.shape[1]
        channels = channels.reshape(n_txsets * n_ue, *channels.shape[2:])
        print(f'    flattened TX sets: {n_txsets} x {n_ue} -> {channels.shape[0]} users')

    # Optional user positions
    positions = None
    for attr in ('rx_pos', 'ue_pos', 'positions', 'user_pos'):
        if hasattr(dataset, attr):
            try:
                positions = np.asarray(getattr(dataset, attr))
                break
            except Exception:
                pass

    # Downsample if user-count cap is set (DeepMIMO scenes can have 100k+ users)
    if max_users is not None and channels.shape[0] > max_users:
        idx = np.linspace(0, channels.shape[0] - 1, max_users).astype(int)
        channels = channels[idx]
    positions = None  # not used by the pipeline; skip to avoid shape mismatches
    print(f'    shape: {channels.shape}  dtype: {channels.dtype}')
    return channels, positions

# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------
# def extract_sub6_features(channel_per_user):
#     """
#     Convert one user's sub-6 GHz channel to a real-valued feature vector.
#     ...
#     """
#     arr = np.asarray(channel_per_user)
#     real = np.real(arr).ravel()
#     imag = np.imag(arr).ravel()
#     return np.concatenate([real, imag]).astype(np.float32)

def _flatten_macro(macro_dataset):
    """
    Concatenate per-user arrays across all BSs in a MacroDataset.
    Returns a dict of stacked arrays, each with leading dim = total users.
    """
    out = {}
    datasets = macro_dataset.datasets if hasattr(macro_dataset, 'datasets') else [macro_dataset]
    keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'power', 'delay', 'num_paths']
    for key in keys:
        parts = []
        for d in datasets:
            arr = np.asarray(getattr(d, key))
            parts.append(arr)
        out[key] = np.concatenate(parts, axis=0)
    return out


def extract_path_features(paths_dict, user_idx, max_paths=25):
    """
    Build a fixed-length physically-meaningful feature vector for one user.

    Features (per path, up to max_paths):
      [aoa_az, aoa_el, aod_az, aod_el, log_power, delay]
    Zero-padded for users with fewer paths. Final dim = max_paths * 6 + 1.
    The +1 is the actual number of valid paths (a useful summary feature).
    """
    n_paths = int(paths_dict['num_paths'][user_idx])
    n_use = min(n_paths, max_paths)

    feats = np.zeros((max_paths, 6), dtype=np.float32)
    if n_use > 0:
        feats[:n_use, 0] = paths_dict['aoa_az'][user_idx, :n_use]
        feats[:n_use, 1] = paths_dict['aoa_el'][user_idx, :n_use]
        feats[:n_use, 2] = paths_dict['aod_az'][user_idx, :n_use]
        feats[:n_use, 3] = paths_dict['aod_el'][user_idx, :n_use]
        # log power is far more useful than raw linear power (large dynamic range)
        pwr = paths_dict['power'][user_idx, :n_use]
        feats[:n_use, 4] = np.log10(np.clip(pwr, 1e-20, None))
        feats[:n_use, 5] = paths_dict['delay'][user_idx, :n_use]

    return np.concatenate([feats.ravel(), [float(n_use)]])


def _load_paths(scenario_name, max_users=None):
    """Load a scenario and return its concatenated path dict."""
    ensure_downloaded(scenario_name)
    print(f'  loading paths from {scenario_name} ...')
    ds = dm.load(scenario_name)
    paths = _flatten_macro(ds)
    print(f'    users across BSs: {paths["num_paths"].shape[0]}')
    if max_users is not None and paths['num_paths'].shape[0] > max_users:
        idx = np.linspace(0, paths['num_paths'].shape[0] - 1, max_users).astype(int)
        for k in paths:
            paths[k] = paths[k][idx]
        print(f'    subsampled to {max_users}')
    return paths


def _compute_mmwave_beams(mmwave_paths, num_beams=64):
    """
    Compute optimal DFT beam per user directly from AoD and path power.
    For a ULA, the beam response at direction theta aligns with aod_az.
    We pick the DFT beam whose angle is closest to the dominant path.
    """
    n_users = mmwave_paths['num_paths'].shape[0]
    y = np.zeros(n_users, dtype=np.int64)
    aod_az = mmwave_paths['aod_az']          # (users, paths) in degrees
    power = mmwave_paths['power']            # (users, paths) linear

    # DFT codebook angles: beam n corresponds to angle = arcsin(2n/N - 1) (approx)
    # Simpler: uniformly spaced in sine-space from -1 to 1.
    beam_sines = np.linspace(-1 + 1 / num_beams, 1 - 1 / num_beams, num_beams)
    beam_angles_deg = np.degrees(np.arcsin(beam_sines))  # (num_beams,)

    for u in range(n_users):
        np_u = int(mmwave_paths['num_paths'][u])
        if np_u == 0:
            y[u] = 0
            continue
        # Weight each path's AoD by its power, pick dominant direction
        p = power[u, :np_u]
        a = aod_az[u, :np_u]
        dominant_angle = a[np.argmax(p)]
        # Nearest beam
        y[u] = int(np.argmin(np.abs(beam_angles_deg - dominant_angle)))
    return y

def build_per_bs_datasets(scene_name, max_users_per_bs=7000,
                          num_beams=64, max_paths=25, cache_dir='./cache'):
    """
    Build separate (X, y) per BS of a scene — treats each BS as its own
    pseudo-environment for cross-site generalization experiments.

    Returns: dict {bs_name: (X, y)}  e.g. {'MIAMI_BS1': (X1,y1), ...}
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{scene_name}_per_bs_K{num_beams}.pkl')
    if os.path.exists(cache_file):
        print(f'[cache hit] {cache_file}')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    preset = SCENE_PRESETS[scene_name]
    print(f'Building per-BS datasets for {scene_name}')

    # Load both bands and keep per-BS structure
    ensure_downloaded(preset['sub6'])
    ensure_downloaded(preset['mmwave'])
    sub6_macro = dm.load(preset['sub6'])
    mmw_macro = dm.load(preset['mmwave'])

    keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'power', 'delay', 'num_paths']

    def _get(d, k):
        return np.asarray(getattr(d, k))

    sub6_bss = sub6_macro.datasets
    mmw_bss = mmw_macro.datasets
    n_bs = min(len(sub6_bss), len(mmw_bss))
    print(f'  found {n_bs} BSs')

    out = {}
    for bs_idx in range(n_bs):
        sub6_paths = {k: _get(sub6_bss[bs_idx], k) for k in keys}
        mmw_paths = {k: _get(mmw_bss[bs_idx], k) for k in keys}

        n_users = min(sub6_paths['num_paths'].shape[0],
                      mmw_paths['num_paths'].shape[0])
        if n_users > max_users_per_bs:
            idx = np.linspace(0, n_users - 1, max_users_per_bs).astype(int)
            for k in keys:
                sub6_paths[k] = sub6_paths[k][idx]
                mmw_paths[k] = mmw_paths[k][idx]
            n_users = max_users_per_bs

        y_all = _compute_mmwave_beams(mmw_paths, num_beams=num_beams)

        X_list, y_list = [], []
        for u in range(n_users):
            if (int(sub6_paths['num_paths'][u]) == 0 or
                    int(mmw_paths['num_paths'][u]) == 0):
                continue
            X_list.append(extract_path_features(sub6_paths, u, max_paths=max_paths))
            y_list.append(y_all[u])

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64)
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        X = X.astype(np.float32)

        bs_name = f'{scene_name}_BS{bs_idx+1}'
        out[bs_name] = (X, y)
        print(f'  {bs_name}: {X.shape[0]} samples, feat_dim={X.shape[1]}')

    with open(cache_file, 'wb') as f:
        pickle.dump(out, f)
    return out

def build_paired_dataset(scene_name, max_users=20000,
                         num_beams=64, max_paths=25, cache_dir='./cache'):
    """
    Build (X, y) using sub-6 GHz path features as input and mmWave optimal
    beam (derived from mmWave paths) as target.
    """
    if scene_name not in SCENE_PRESETS:
        raise KeyError(f'Unknown scene {scene_name!r}. '
                       f'Known: {list(SCENE_PRESETS.keys())}')

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{scene_name}_K{num_beams}_paths.pkl')
    if os.path.exists(cache_file):
        print(f'[cache hit] {cache_file}')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    preset = SCENE_PRESETS[scene_name]
    print(f'Building paired path-feature dataset for {scene_name}')

    sub6_paths = _load_paths(preset['sub6'], max_users=max_users)
    mmw_paths = _load_paths(preset['mmwave'], max_users=max_users)

    n_users = min(sub6_paths['num_paths'].shape[0], mmw_paths['num_paths'].shape[0])
    print(f'  pairing {n_users} users')

    # Build X from sub-6 paths, y from mmWave paths
    X_list, y_list = [], []
    y_all = _compute_mmwave_beams(
        {k: v[:n_users] for k, v in mmw_paths.items()},
        num_beams=num_beams,
    )

    kept, skipped = 0, 0
    for u in range(n_users):
        if int(sub6_paths['num_paths'][u]) == 0 or int(mmw_paths['num_paths'][u]) == 0:
            skipped += 1
            continue
        X_list.append(extract_path_features(sub6_paths, u, max_paths=max_paths))
        y_list.append(y_all[u])
        kept += 1

    if kept == 0:
        raise RuntimeError(f'No valid users in {scene_name}.')

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    print(f'  → {kept} samples ({skipped} skipped)')
    print(f'  feature dim = {X.shape[1]}, beams used = {len(np.unique(y))}/{num_beams}')

    # Normalize features
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    X = X.astype(np.float32)
    print(f'  normalized: X mean={X.mean():.4f}, std={X.std():.4f}')

    with open(cache_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f'  cached to {cache_file}')
    return X, y

if __name__ == '__main__':
    X, y = build_paired_dataset('ASU')
    print(f'\nX shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'y range: [{y.min()}, {y.max()}]')
