"""
Step 1: Load DeepMIMO scenarios and cache preprocessed (X, y) tensors.

Produces one cached pickle per scene. Re-running is instant after the first pass.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.deepmimo_loader import build_paired_dataset

SCENES = ['MIAMI', 'SEATTLE', 'DALLAS']  # v4 auto-downloads these from the DeepMIMO database

if __name__ == '__main__':
    for scene in SCENES:
        print(f'\n=== Building dataset for {scene} ===')
        try:
            X, y = build_paired_dataset(scene)
            print(f'OK: {scene} -> X {X.shape}, y {y.shape}')
        except Exception as e:
            print(f'FAILED for {scene}: {e}')
            import traceback; traceback.print_exc()
