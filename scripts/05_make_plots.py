"""
Step 5: Read ./results/*.csv and produce paper-ready plots + summary tables.

Outputs (in ./results/):
  - fig_generalization.png   — headline: top-5 acc vs env distance
  - fig_robustness.png       — headline: noise curves
  - fig_topk_vs_k.png        — reference comparison (baseline vs CNN vs random)
  - summary_table.md         — numbers for the report
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from utils.plotting import plot_generalization_map, plot_noise_robustness


if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)

    # --- Generalization map ---
    gen_path = './results/generalization.csv'
    if os.path.exists(gen_path):
        gen_df = pd.read_csv(gen_path)
        plot_generalization_map(gen_df, './results/fig_generalization.png')
        print('wrote fig_generalization.png')
    else:
        print('skipping generalization plot (run script 03 first)')

    # --- Robustness curves ---
    rob_path = './results/robustness.csv'
    if os.path.exists(rob_path):
        rob_df = pd.read_csv(rob_path)
        plot_noise_robustness(rob_df, './results/fig_robustness.png')
        print('wrote fig_robustness.png')
    else:
        print('skipping robustness plot (run script 04 first)')

    # --- Summary table for the report / paper ---
    lines = ['# Summary of Results\n']
    if os.path.exists(gen_path):
        lines.append('## Generalization (train → test)\n')
        g = pd.read_csv(gen_path)
        lines.append(g[['train_scene','test_scene','env_distance',
                        'top1','top3','top5','top10']].to_markdown(index=False))
        lines.append('\n')
    if os.path.exists(rob_path):
        lines.append('\n## Robustness (noise sweep)\n')
        r = pd.read_csv(rob_path)
        lines.append(r[['training','noise_std','top1','top3','top5']].to_markdown(index=False))
    with open('./results/summary_table.md', 'w') as f:
        f.write('\n'.join(lines))
    print('wrote summary_table.md')
