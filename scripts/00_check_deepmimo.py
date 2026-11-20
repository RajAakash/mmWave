"""
Diagnostic: confirm the installed DeepMIMO version works end-to-end before
running the real pipeline. Run this first.
"""
import deepmimo as dm

print(f'deepmimo version: {getattr(dm, "__version__", "unknown")}')
print(f'Top-level API: {[a for a in dir(dm) if not a.startswith("_")][:20]}')

# Try the smallest known scenario
scenario = 'asu_campus_3p5'
print(f'\nAttempting download of {scenario} ...')
try:
    dm.download(scenario)
except Exception as e:
    print(f'download exception (may be benign if already cached): {e}')

print(f'\nLoading {scenario} ...')
dataset = dm.load(scenario)
print(f'dataset type: {type(dataset)}')
print(f'dataset attrs: {[a for a in dir(dataset) if not a.startswith("_")][:40]}')

# Try to get channels
for attr in ('channel', 'channels'):
    if hasattr(dataset, attr):
        val = getattr(dataset, attr)
        if callable(val):
            val = val()
        print(f'dataset.{attr}: shape = {val.shape}, dtype = {val.dtype}')
        break
else:
    if hasattr(dataset, 'compute_channels'):
        val = dataset.compute_channels()
        print(f'dataset.compute_channels(): shape = {val.shape}')
    else:
        print('No channel attribute found — investigate attrs above.')

print('\nIf shape prints above, the loader should work. Proceed to 01_generate_data.py')
