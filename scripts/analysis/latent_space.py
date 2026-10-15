"""
Embedding 可视化 (latent space)：
  读取高置信度集合 high_conf_result.tsv，从 h5 加载模型/ESM embedding，
  PCA + t-SNE 降到 2D，并绘制按簇 / 概率 / 物种着色的散点图。

运行前请先运行 dual_track_filter.py 生成 high_conf_result.tsv。
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


sns.set_style("ticks")         # 使用 ticks 风格，防止覆盖下方字体设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# =====================
# Global plot style
# =====================

FS_AXIS_LABEL = 30         # X/Y 轴标题字号
FS_TICK       = 30        # X/Y 轴刻度数字字号
FS_LEGEND     = 20       # 图例字号

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
# 公共颜色 / 配置
# =====================

COLOR_STAR       = '#E63946'   # 实验验证点（唯一主角）
COLOR_BG         = '#E0E0E0'   # 背景点（浅灰）
COLOR_BG_LEGEND  = '#C0C0C0'   # 图例中的背景点

EMBEDDINGS_INFO = [
    ('model_tsne', 'rec'),
    ('esm_tsne', 'esm'),
]


# =====================
# 投影与标注
# =====================

def project_to_tsne(matrix, random_state=999):
    """StandardScaler -> (维度 > 50 时先 PCA 到 50) -> t-SNE 2D。"""
    scaled = StandardScaler().fit_transform(matrix)
    if scaled.shape[1] > 50:
        print(f"   -> 维度为 {scaled.shape[1]}，先执行 PCA 降维至 50...")
        scaled = PCA(n_components=50, random_state=random_state).fit_transform(scaled)
    reducer = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
    return reducer.fit_transform(scaled)


def add_latent_coords(df, h5_path, esm_h5_path, num_processes=8):
    """读取模型/ESM embedding 并投影到 2D，写回 df（新增 *_tsne_1 / *_tsne_2）。"""
    result_df, all_embeddings, all_esms, _, _ = get_check_df(
        h5_path=h5_path,
        esm_h5_path=esm_h5_path,
        df=df,
        num_processes=num_processes,
    )

    print("⏳ 正在处理 Model Embeddings...")
    emb_tsne = project_to_tsne(all_embeddings)
    result_df['model_tsne_1'] = emb_tsne[:, 0]
    result_df['model_tsne_2'] = emb_tsne[:, 1]

    print("⏳ 正在处理 ESM Embeddings...")
    esm_tsne = project_to_tsne(all_esms)
    result_df['esm_tsne_1'] = esm_tsne[:, 0]
    result_df['esm_tsne_2'] = esm_tsne[:, 1]

    return result_df


def add_experiment_label(df, labels, data_name):
    """按实验验证标注给片段打标签 (1.0 = 成功，NaN = 背景)。"""
    df['exp_label'] = np.nan
    success_ids = labels[data_name]['pos']
    df.loc[df['seg_key'].isin(success_ids), 'exp_label'] = 1.0
    return df


def add_cluster_label(df, cluster_tsv, top_k=10):
    """按聚类结果给 seg_key 打簇标签（只标注成员数最多的前 top_k 个簇）。"""
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
# 绘图公共底座
# =====================

def _iter_embeddings(df):
    """按 (prefix, plot_name, dim1, dim2) 遍历两类 embedding，自动跳过缺失列。"""
    for prefix, plot_name in EMBEDDINGS_INFO:
        dim1, dim2 = f'{prefix}_1', f'{prefix}_2'
        if dim1 in df.columns and dim2 in df.columns:
            yield prefix, plot_name, dim1, dim2


def _draw_exp_stars(ax, df, dim1, dim2, s=200):
    """图层：绘制实验验证点（红星）。"""
    mask = df["exp_label"] == 1.0
    if not mask.any():
        return
    exp_data = df[mask]
    ax.scatter(exp_data[dim1], exp_data[dim2],
               c=COLOR_STAR, label='_nolegend_',
               marker='*', s=s, edgecolors='black', linewidths=1.2, zorder=5)


def _exp_proxy(markersize=18):
    """实验验证点的图例代理。"""
    return mlines.Line2D([0], [0], marker='*', color='w',
                         markerfacecolor=COLOR_STAR, markeredgecolor='black', markersize=markersize)


def _style_axis(ax, linewidth_spines=False):
    """统一坐标轴样式。"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if linewidth_spines:
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('t-SNE 1', fontweight='bold', labelpad=8)
    ax.set_ylabel('t-SNE 2', fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', width=1.5)


def _save_fig(fig, save_path):
    """统一出图流程：tight_layout -> savefig -> close。"""
    plt.tight_layout()
    plt.savefig(save_path, format="png", bbox_inches='tight', pad_inches=0.02, dpi=600, facecolor='white')
    plt.close(fig)


# =====================
# 三张图
# =====================

def plot_embeddings_color_by_cluster(df, save_dir, top_n_legends=3, figsize=(10, 6)):
    """
    根据已有的 t-SNE 坐标，绘制并保存按聚类着色的 rec / esm 两张图。

    Args:
        df: 包含模型/ESM t-SNE 坐标、实验标签和簇标签的 DataFrame
        save_dir: 保存图片的文件夹路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 无红色干扰的"安全高级调色板"（红色独家留给实验成功点）
    safe_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#17becf', '#bcbd22', '#393b79', '#e6550d', '#31a354',
        '#7b4173', '#bd9e39', '#8c6d31', '#6b6ecf', '#b5cf6b'
    ]

    # 按 "Cluster X" 中的数字排序，保证色彩连续
    unique_clusters = [c for c in df["cluster_label"].unique() if pd.notna(c)]
    try:
        unique_clusters = sorted(unique_clusters, key=lambda x: int(str(x).split(' ')[1]))
    except (IndexError, ValueError):
        unique_clusters = sorted(unique_clusters)  # 降级为普通字母排序

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)

        # 图层 1：灰色背景（既不在 Top-K 簇中，也不是实验成功点）
        mask_bg = df["cluster_label"].isna() & (df["exp_label"] != 1.0)
        ax.scatter(df.loc[mask_bg, dim1], df.loc[mask_bg, dim2],
                   c=COLOR_BG, s=5, alpha=0.4, label='_nolegend_', zorder=1)

        legend_handles = []
        legend_labels = []
        bg_proxy = mlines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=COLOR_BG_LEGEND, markersize=8)
        legend_handles.append(bg_proxy)
        legend_labels.append('Background Candidates')

        # 图层 2：Top K 簇（极简折叠图例）
        other_top_handles = []  # 收集未能进入 Top N 的簇颜色
        for i, cluster_name in enumerate(unique_clusters):
            mask_cluster = (df["cluster_label"] == cluster_name) & (df["exp_label"] != 1.0)
            cluster_data = df[mask_cluster]
            current_color = safe_colors[i % len(safe_colors)]

            ax.scatter(cluster_data[dim1], cluster_data[dim2],
                       color=current_color, s=25, alpha=0.8, label="_nolegend_", zorder=2)

            if i < top_n_legends:
                # VIP 席位：前 top_n_legends 个簇拥有独立图例
                proxy_dot = mlines.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=current_color, markersize=10)
                legend_handles.append(proxy_dot)
                legend_labels.append(f'{cluster_name}')
            elif len(other_top_handles) < 4:
                # 合并席位：剩余簇颜色收进组合图例（最多 4 个防止过长）
                other_top_handles.append(
                    mlines.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=current_color, markersize=8)
                )

        if other_top_handles:
            legend_handles.append(tuple(other_top_handles))
            legend_labels.append('Other Top Clusters')

        # 图层 3：实验验证点（醒目红星）
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
    """按 HMM-HMM 比对概率着色的 t-SNE 图（rec / esm 两张）。"""
    CMAP_PROB = 'mako_r'  # 概率渐变色带
    os.makedirs(save_dir, exist_ok=True)

    # 合并概率（未比对上的置 0），再按概率从小到大排序
    merged_df = pd.merge(df, hmm_df[['query', 'prob']],
                         left_on='seg_key', right_on='query', how='left')
    merged_df['prob'] = merged_df['prob'].fillna(0)
    merged_df = merged_df.sort_values(by='prob', ascending=True)
    merged_df['prob'] = merged_df['prob'] / 100.0

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(merged_df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)

        # 图层 1：候选点（连续色带）
        mask = merged_df['exp_label'] != 1.0
        scatter = ax.scatter(merged_df.loc[mask, dim1], merged_df.loc[mask, dim2],
                             c=merged_df.loc[mask, 'prob'],
                             cmap=CMAP_PROB, vmin=0, vmax=1.0,
                             s=8, alpha=0.7, zorder=1)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Alignment Prob', fontweight='bold', labelpad=12)
        cbar.outline.set_visible(False)

        # 图层 2：实验验证点（红星依然是主角）
        _draw_exp_stars(ax, merged_df, dim1, dim2, s=200)

        _style_axis(ax, linewidth_spines=True)

        ax.legend([_exp_proxy()], ['Exp Success'], loc='best', frameon=False)

        _save_fig(fig, os.path.join(save_dir, f"{plot_name}_color_by_hmm_prob.png"))


def plot_embeddings_color_by_species(df, species_df, save_dir, figsize=(10, 6),
                                     top_n_legends=3, bundle_n=4):
    """
    按 Organism ID 着色的 t-SNE 图：前 3 个物种独立图例，随后 4 个打包，
    其余灰色背景。
    """
    safe_colors = [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
        '#2C3E50', '#E67E22', '#27AE60', '#8E44AD', '#F39C12'
    ]
    os.makedirs(save_dir, exist_ok=True)

    # 数据对齐与合并
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

    # 需要高亮的物种总数 (top_n_legends + bundle_n)
    total_top = top_n_legends + bundle_n
    species_counts = merged_df[merged_df['Organism (ID)'] != 'Unknown']['Organism (ID)'].value_counts()
    top_species_list = species_counts.index[:total_top].tolist()
    vip_species = top_species_list[:top_n_legends]
    other_top_species = top_species_list[top_n_legends:]

    for prefix, plot_name, dim1, dim2 in _iter_embeddings(merged_df):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)
        legend_handles = []
        legend_labels = []

        # 图层 1：背景（Other Species & Unknown）
        mask_bg = (~merged_df['Organism (ID)'].isin(top_species_list)) & (merged_df['exp_label'] != 1.0)
        ax.scatter(merged_df.loc[mask_bg, dim1], merged_df.loc[mask_bg, dim2],
                   c=COLOR_BG, s=5, alpha=0.25, label='_nolegend_', zorder=1)

        bg_proxy = mlines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=COLOR_BG_LEGEND, markersize=8)
        legend_handles.append(bg_proxy)
        legend_labels.append('Other Species (Bg)')

        # 中间层：Other Top Species
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

        # VIP 物种层（从丰度最大的开始画）
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

        # 实验验证点
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
# 主流程
# =====================

def run(data_name):
    data_save_path = DATA_SAVE_FP_DICT['data_save_path'][data_name]
    picture_save_path = DATA_SAVE_FP_DICT['picture_save_path'][data_name]
    emb_save_path = os.path.join(picture_save_path, "or_filter_emb_tsne_2d")
    os.makedirs(emb_save_path, exist_ok=True)

    fasta_dir = os.path.join(data_save_path, "fasta_dir")
    df = pd.read_csv(os.path.join(data_save_path, "high_conf_result.tsv"), sep="\t")

    labels = load_experiment_labels()

    # --- 投影（带缓存，避免每次重跑 t-SNE） ---
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

    # --- 聚类标签（依赖 fasta 聚类结果，每次现算） ---
    fasta_file = Path(os.path.join(fasta_dir, "or_filter_seg_seq.fasta")).resolve()
    cluster_tsv = os.path.join(fasta_file.parent, f"{fasta_file.stem}_cluster_result", "clusterRes_cluster.tsv")
    df = add_cluster_label(df, cluster_tsv)

    # --- 绘图 ---
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
