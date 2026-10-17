"""
Embedding visualization (latent space):
  Read the high-confidence set high_conf_result.tsv, load model/ESM embeddings from h5,
  project to 2D via PCA + t-SNE, and draw scatter plots colored by cluster / probability / species.

Run dual_track_filter.py first to produce high_conf_result.tsv.
"""
import os

os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from helpers import DATA_SAVE_FP_DICT, get_check_df, parse_cluster_tsv, load_experiment_labels


sns.set_style("ticks")         # use ticks style to avoid overriding the font settings below
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# =====================
# Global plot style
# =====================

FS_AXIS_LABEL = 30         # X/Y axis label font size
FS_TICK       = 30        # X/Y tick label font size
FS_LEGEND     = 20       # legend font size

plt.rcParams.update(
    {
        # Sizes (global)
        "font.size": FS_TICK,
        "axes.labelsize": FS_AXIS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
    }
)

# =====================
# Shared colors / config
# =====================

COLOR_STAR       = '#E63946'   # experimentally validated points (the only highlight)
COLOR_BG         = '#E0E0E0'   # background points (light gray)
COLOR_BG_LEGEND  = '#C0C0C0'   # background points in the legend

EMBEDDINGS_INFO = [
    ('model_tsne', 'rec'),
    ('esm_tsne', 'esm'),
]


# =====================
# Projection and annotation
# =====================

def project_to_tsne(matrix, random_state=999):
    """StandardScaler -> (PCA to 50 when dims > 50) -> t-SNE 2D."""
    scaled = StandardScaler().fit_transform(matrix)
    if scaled.shape[1] > 50:
        print(f"   -> dims = {scaled.shape[1]}, running PCA to 50 first...")
        scaled = PCA(n_components=50, random_state=random_state).fit_transform(scaled)
    reducer = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
    return reducer.fit_transform(scaled)


def add_latent_coords(df, h5_path, esm_h5_path, num_processes=8):
    """Load model/ESM embeddings, project to 2D, and write back to df (adds *_tsne_1 / *_tsne_2)."""
    result_df, all_embeddings, all_esms, _, _ = get_check_df(
        h5_path=h5_path,
        esm_h5_path=esm_h5_path,
        df=df,
        num_processes=num_processes,
    )

    print("Processing Model Embeddings...")
    emb_tsne = project_to_tsne(all_embeddings)
    result_df['model_tsne_1'] = emb_tsne[:, 0]
    result_df['model_tsne_2'] = emb_tsne[:, 1]

    print("Processing ESM Embeddings...")
    esm_tsne = project_to_tsne(all_esms)
    result_df['esm_tsne_1'] = esm_tsne[:, 0]
    result_df['esm_tsne_2'] = esm_tsne[:, 1]

    return result_df


def add_experiment_label(df, labels, data_name):
    """Label segments by experimental validation (1.0 = positive, NaN = background)."""
    df['exp_label'] = np.nan
    success_ids = labels[data_name]['pos']
    df.loc[df['seg_key'].isin(success_ids), 'exp_label'] = 1.0
    return df


def add_cluster_label(df, cluster_tsv, top_k=10):
    """Label seg_key by cluster membership (only the top_k largest clusters)."""
    sorted_cluster_dict, _ = parse_cluster_tsv(cluster_tsv)
    top_clusters = list(sorted_cluster_dict.items())[:top_k]

    cluster_mapping = {}
    for i, (rep, members) in enumerate(top_clusters):
        cluster_name = f"Cluster {i+1} (Size: {len(members)})"
        for member in members:
            cluster_mapping[member] = cluster_name

    df['cluster_label'] = df['seg_key'].map(cluster_mapping)
    return df


# =====================
# Shared plotting helpers
# =====================

def _iter_embeddings(df):
    """Iterate the two embedding kinds as (prefix, plot_name, dim1, dim2), skipping missing columns."""
    for prefix, plot_name in EMBEDDINGS_INFO:
        dim1, dim2 = f'{prefix}_1', f'{prefix}_2'
        if dim1 in df.columns and dim2 in df.columns:
            yield prefix, plot_name, dim1, dim2


def _draw_exp_stars(ax, df, dim1, dim2, s=200):
    """Layer: draw experimentally validated points (red stars)."""
    mask = df["exp_label"] == 1.0
    if not mask.any():
        return
    exp_data = df[mask]
    ax.scatter(exp_data[dim1], exp_data[dim2],
               c=COLOR_STAR, label='_nolegend_',
               marker='*', s=s, edgecolors='black', linewidths=1.2, zorder=5)


def _exp_proxy(markersize=18):
    """Legend proxy for experimentally validated points."""
    return mlines.Line2D([0], [0], marker='*', color='w',
                         markerfacecolor=COLOR_STAR, markeredgecolor='black', markersize=markersize)


def _style_axis(ax, linewidth_spines=False):
    """Shared axis styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if linewidth_spines:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('t-SNE 1', fontweight='bold', labelpad=8)
    ax.set_ylabel('t-SNE 2', fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', width=1.5)


def _save_fig(fig, save_path):
    """Shared save routine: tight_layout -> savefig -> close."""
    plt.tight_layout()
    plt.savefig(save_path, format="png", bbox_inches='tight', pad_inches=0.02, dpi=600, facecolor='white')
    plt.close(fig)


# =====================
# The three plots
# =====================

def plot_embeddings_color_by_cluster(df, save_dir, top_n_legends=3, figsize=(10, 6)):
    """
    Plot and save rec / esm scatter plots colored by cluster from the existing t-SNE coords.

    Args:
        df: DataFrame with model/ESM t-SNE coords, experiment labels and cluster labels
        save_dir: directory to save the figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # "safe" palette without red interference (red is reserved for the experimental positives)
    safe_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#17becf', '#bcbd22', '#393b79', '#e6550d', '#31a354',
        '#7b4173', '#bd9e39', '#8c6d31', '#6b6ecf', '#b5cf6b'
    ]

    # sort by the number in "Cluster X" so colors stay contiguous
    unique_clusters = [c for c in df["cluster_label"].unique() if pd.notna(c)]
    try:
        unique_clusters = sorted(unique_clusters, key=lambda x: int(str(x).split(' ')[1]))
    except (IndexError, ValueError):
        unique_clusters = sorted(unique_clusters)  # fall back to plain string sort

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)

        # Layer 1: gray background (neither in a Top-K cluster nor an experimental positive)
        mask_bg = df["cluster_label"].isna() & (df["exp_label"] != 1.0)
        ax.scatter(df.loc[mask_bg, dim1], df.loc[mask_bg, dim2],
                   c=COLOR_BG, s=5, alpha=0.4, label='_nolegend_', zorder=1)

        legend_handles = []
        legend_labels = []
        bg_proxy = mlines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=COLOR_BG_LEGEND, markersize=8)
        legend_handles.append(bg_proxy)
        legend_labels.append('Background Candidates')

        # Layer 2: Top-K clusters (collapsed legend)
        other_top_handles = []  # collect colors of clusters that missed the Top N
        for i, cluster_name in enumerate(unique_clusters):
            mask_cluster = (df["cluster_label"] == cluster_name) & (df["exp_label"] != 1.0)
            cluster_data = df[mask_cluster]
            current_color = safe_colors[i % len(safe_colors)]

            ax.scatter(cluster_data[dim1], cluster_data[dim2],
                       color=current_color, s=25, alpha=0.8, label="_nolegend_", zorder=2)

            if i < top_n_legends:
                # VIP: the first top_n_legends clusters get their own legend entry
                proxy_dot = mlines.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=current_color, markersize=10)
                legend_handles.append(proxy_dot)
                legend_labels.append(f'{cluster_name}')
            elif len(other_top_handles) < 4:
                # merged: remaining cluster colors go into a combined legend (max 4 to keep it short)
                other_top_handles.append(
                    mlines.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=current_color, markersize=8)
                )

        if other_top_handles:
            legend_handles.append(tuple(other_top_handles))
            legend_labels.append('Other Top Clusters')

        # Layer 3: experimental positives (prominent red stars)
        _draw_exp_stars(ax, df, dim1, dim2, s=200)
        legend_handles.append(_exp_proxy())
        legend_labels.append('Exp Success')

        _style_axis(ax)

        leg = ax.legend(
            legend_handles, legend_labels,
            loc='best', bbox_to_anchor=(1.01, 1),
            frameon=True, framealpha=0.10, handlelength=1.6,
            borderpad=0.5, labelspacing=0.4,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.30)

        _save_fig(fig, os.path.join(save_dir, f"{plot_name}_color_by_cluster.png"))


def plot_embeddings_color_by_prob(df, hmm_df, save_dir, figsize=(10, 6)):
    """t-SNE scatter colored by HMM-HMM alignment probability (rec / esm)."""
    CMAP_PROB = 'mako_r'  # probability colormap
    os.makedirs(save_dir, exist_ok=True)

    # merge probability (0 for unmatched), then sort ascending so high-prob points are drawn on top
    merged_df = pd.merge(df, hmm_df[['query', 'prob']],
                         left_on='seg_key', right_on='query', how='left')
    merged_df['prob'] = merged_df['prob'].fillna(0)
    merged_df = merged_df.sort_values(by='prob', ascending=True)
    merged_df['prob'] = merged_df['prob'] / 100.0

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(merged_df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)

        # Layer 1: candidates (continuous colormap)
        mask = merged_df['exp_label'] != 1.0
        scatter = ax.scatter(merged_df.loc[mask, dim1], merged_df.loc[mask, dim2],
                             c=merged_df.loc[mask, 'prob'],
                             cmap=CMAP_PROB, vmin=0, vmax=1.0,
                             s=8, alpha=0.7, zorder=1)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Alignment Prob', fontweight='bold', labelpad=12)
        cbar.outline.set_visible(False)

        # Layer 2: experimental positives (red stars remain the highlight)
        _draw_exp_stars(ax, merged_df, dim1, dim2, s=200)

        _style_axis(ax, linewidth_spines=True)

        ax.legend([_exp_proxy()], ['Exp Success'], loc='best', frameon=False)

        _save_fig(fig, os.path.join(save_dir, f"{plot_name}_color_by_hmm_prob.png"))


def plot_embeddings_color_by_species(df, species_df, save_dir, figsize=(10, 6),
                                     top_n_legends=3, bundle_n=4):
    """
    t-SNE scatter colored by Organism ID: first 3 species get individual legend entries, the next 4 are bundled,
    and the rest are gray background.
    """
    safe_colors = [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
        '#2C3E50', '#E67E22', '#27AE60', '#8E44AD', '#F39C12'
    ]
    os.makedirs(save_dir, exist_ok=True)

    # Data alignment and merge
    if 'acc' not in df.columns:
        df = df.copy()
        df['acc'] = df['seg_key'].apply(lambda x: x.split('_')[0])

    merged_df = pd.merge(df, species_df[['Entry', 'Organism (ID)']],
                         left_on='acc', right_on='Entry', how='left')

    def clean_species_id(x):
        if pd.isna(x) or x == 'Unknown':
            return 'Unknown'
        try:
            return str(int(float(x)))
        except Exception:
            return str(x)

    merged_df = merged_df.copy()
    merged_df['Organism (ID)'] = merged_df['Organism (ID)'].apply(clean_species_id)

    # total species to highlight (top_n_legends + bundle_n)
    total_top = top_n_legends + bundle_n
    species_counts = merged_df[merged_df['Organism (ID)'] != 'Unknown']['Organism (ID)'].value_counts()
    top_species_list = species_counts.index[:total_top].tolist()
    vip_species = top_species_list[:top_n_legends]
    other_top_species = top_species_list[top_n_legends:]

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(merged_df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)
        legend_handles = []
        legend_labels = []

        # Layer 1: background (Other Species & Unknown)
        mask_bg = (~merged_df['Organism (ID)'].isin(top_species_list)) & (merged_df['exp_label'] != 1.0)
        ax.scatter(merged_df.loc[mask_bg, dim1], merged_df.loc[mask_bg, dim2],
                   c=COLOR_BG, s=5, alpha=0.25, label='_nolegend_', zorder=1)

        bg_proxy = mlines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=COLOR_BG_LEGEND, markersize=8)
        legend_handles.append(bg_proxy)
        legend_labels.append('Other Species (Bg)')

        # Middle layer: Other Top Species
        other_top_handles = []
        for i, species_id in enumerate(other_top_species):
            mask = (merged_df['Organism (ID)'] == species_id) & (merged_df['exp_label'] != 1.0)
            current_color = safe_colors[(top_n_legends + i) % len(safe_colors)]

            ax.scatter(merged_df.loc[mask, dim1], merged_df.loc[mask, dim2],
                       color=current_color, s=10, alpha=0.6, label="_nolegend_", zorder=2)

            if len(other_top_handles) < bundle_n:
                other_top_handles.append(
                    mlines.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=current_color, markersize=7)
                )

        if other_top_handles:
            legend_handles.append(tuple(other_top_handles))
            legend_labels.append('Other Top Species')

        # VIP species layer (draw most abundant first)
        vip_to_plot = sorted(vip_species, key=lambda s: species_counts[s], reverse=True)
        vip_proxies = {}
        for species_id in vip_to_plot:
            mask = (merged_df['Organism (ID)'] == species_id) & (merged_df["exp_label"] != 1.0)
            current_color = safe_colors[top_species_list.index(species_id) % len(safe_colors)]

            ax.scatter(merged_df.loc[mask, dim1], merged_df.loc[mask, dim2],
                       color=current_color, s=15, alpha=0.7, label="_nolegend_", zorder=3)

            count = int(mask.sum())
            vip_proxies[species_id] = (
                mlines.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=current_color, markersize=8),
                f'Sp: {species_id} (n={count})',
            )

        for i, species_id in enumerate(vip_species):
            proxy_dot, label_text = vip_proxies[species_id]
            legend_handles.insert(1 + i, proxy_dot)
            legend_labels.insert(1 + i, label_text)

        # Experimental positives
        _draw_exp_stars(ax, merged_df, dim1, dim2, s=300)
        legend_handles.append(_exp_proxy())
        legend_labels.append('Exp Success')

        _style_axis(ax, linewidth_spines=True)

        leg = ax.legend(
            legend_handles, legend_labels,
            loc='best', bbox_to_anchor=(1.01, 1),
            frameon=True, framealpha=0.10, handlelength=1.6,
            borderpad=0.5, labelspacing=0.4,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        )
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.30)

        _save_fig(fig, os.path.join(save_dir, f"{plot_name}_color_by_species.png"))


# =====================
# Main flow
# =====================

def run(data_name):
    data_save_path = DATA_SAVE_FP_DICT['data_save_path'][data_name]
    picture_save_path = DATA_SAVE_FP_DICT['picture_save_path'][data_name]
    emb_save_path = os.path.join(picture_save_path, "emb_tsne_2d")
    os.makedirs(emb_save_path, exist_ok=True)

    fasta_dir = os.path.join(data_save_path, "fasta_dir")
    df = pd.read_csv(os.path.join(data_save_path, "high_conf_result.tsv"), sep="\t")

    labels = load_experiment_labels()

    # --- Projection (cached to avoid re-running t-SNE) ---
    tsne_tsv = os.path.join(data_save_path, "high_conf_with_tsne.tsv")
    if os.path.exists(tsne_tsv):
        df = pd.read_csv(tsne_tsv, sep="\t")
    else:
        df = add_latent_coords(
            df,
            h5_path=DATA_SAVE_FP_DICT['h5'][data_name],
            esm_h5_path=DATA_SAVE_FP_DICT['esm_h5'][data_name],
            num_processes=8,
        )
        df = add_experiment_label(df, labels, data_name)
        df.to_csv(tsne_tsv, sep="\t", index=False)

    # --- Cluster labels (depends on FASTA clustering, recomputed each run) ---
    fasta_file = Path(os.path.join(fasta_dir, "high_conf_seg_seq.fasta")).resolve()
    cluster_tsv = os.path.join(fasta_file.parent, f"{fasta_file.stem}_cluster_result", "clusterRes_cluster.tsv")
    df = add_cluster_label(df, cluster_tsv)

    # --- Plots ---
    plot_embeddings_color_by_cluster(df, save_dir=emb_save_path)

    hmm_df_fp = os.path.join(data_save_path, "hhm_dir", "align_best_prob.tsv")
    hmm_df = pd.read_csv(hmm_df_fp, sep="\t")
    plot_embeddings_color_by_prob(df, hmm_df, save_dir=emb_save_path)

    species_df_fp = os.path.join(data_save_path, "species_information", "idmapping.tsv")
    species_df = pd.read_csv(species_df_fp, sep="\t")
    plot_embeddings_color_by_species(df, species_df, save_dir=emb_save_path)


def main():
    for data_name in ["Ocean", "Asgard"]:
        run(data_name)


if __name__ == "__main__":
    main()
