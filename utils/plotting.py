"""Reusable plotting functions for paper-ready figures."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_topk_vs_k(results_dict, outpath, title='Top-K accuracy'):
    """results_dict: {method_name: {'top1':.., 'top3':.., 'top5':.., 'top10':..}}"""
    ks = [1, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in results_dict.items():
        ax.plot(ks, [m[f'top{k}'] for k in ks], marker='o', linewidth=2, label=name)
    ax.set_xlabel('K (candidate beams)')
    ax.set_ylabel('Top-K accuracy')
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.savefig(outpath)
    plt.close(fig)


def plot_generalization_map(df, outpath):
    """
    Headline plot of the paper.
    df: columns = ['train_scene', 'test_scene', 'env_distance',
                   'top1', 'top3', 'top5']
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df['env_distance'], df['top5'],
                         c=df['top1'], s=140, cmap='viridis',
                         edgecolors='black', linewidth=0.8)
    for _, row in df.iterrows():
        ax.annotate(f"{row['train_scene']}→{row['test_scene']}",
                    (row['env_distance'], row['top5']),
                    textcoords='offset points', xytext=(7, 5), fontsize=9)
    # Trend line
    if len(df) >= 2:
        z = np.polyfit(df['env_distance'], df['top5'], 1)
        xs = np.linspace(df['env_distance'].min(), df['env_distance'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), '--', color='red', alpha=0.6,
                label=f'linear fit: slope={z[0]:.2f}')
        ax.legend()
    ax.set_xlabel('Environmental distance (train ↔ test)')
    ax.set_ylabel('Top-5 accuracy')
    ax.set_title('Generalization boundary: accuracy vs environment drift')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Top-1 accuracy')
    ax.grid(alpha=0.3)
    fig.savefig(outpath)
    plt.close(fig)


def plot_noise_robustness(df, outpath):
    """
    df: columns = ['noise_std', 'top1', 'top3', 'top5', 'training']
    'training' ∈ {'clean', 'noise-augmented'}
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for training, sub in df.groupby('training'):
        sub = sub.sort_values('noise_std')
        ax.plot(sub['noise_std'], sub['top5'],
                marker='o', linewidth=2, label=f'{training} (Top-5)')
        ax.plot(sub['noise_std'], sub['top1'],
                marker='s', linewidth=2, linestyle='--',
                label=f'{training} (Top-1)')
    ax.set_xlabel('Sub-6 GHz noise level (σ, relative)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Robustness to sub-6 GHz channel estimation noise')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.savefig(outpath)
    plt.close(fig)
