import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf

from helpers import DATA_SAVE_FP_DICT


def plot_go_enrichment(target_df, background_df, save_dir, 
                       go_col="Gene Ontology (molecular function)",  
                       top_n=10,
                       figsize = (10, 6),
                       x_limit=None,     # X轴范围限制，用于将两张图强行对
    ):

    # -------------------------
    # Local font config (for GO plot only)
    # -------------------------
    FS_GO_AXIS_LABEL = 30     # x/y 轴标题
    FS_GO_TICK = 20           # x/y 轴刻度
    FS_GO_BAR_TEXT = 15       # 条形右侧数值
    FS_GO_CBAR_TICK = 30      # colorbar 刻度

    os.makedirs(save_dir, exist_ok=True)
    
    if 'acc' not in target_df.columns or 'Entry' not in background_df.columns:
        raise ValueError("target_df 必须包含 'acc' 列，background_df 必须包含 'Entry' 列")
        
    target_acc_list = target_df['acc'].dropna().unique().tolist()
    
    # 过滤掉没有当前 GO 注释的背景蛋白质
    bg_clean = background_df.dropna(subset=[go_col]).copy()
    

    def clean_go_terms(go_string):
        terms = str(go_string).split(';')
        cleaned_terms = []
        for t in terms:
            name = re.sub(r'\s*\[GO:\d+\]\s*', '', t).strip()
            if name:
                cleaned_terms.append(name)
        return cleaned_terms


    bg_clean['go_list'] = bg_clean[go_col].apply(clean_go_terms)
    background_acc = bg_clean['Entry'].unique().tolist()
    
    target_acc_clean = [acc for acc in target_acc_list if acc in background_acc]
    
    if len(target_acc_clean) == 0:
        print(f"警告: 在给定的目标集中，没有找到包含 {go_col} 注释的蛋白质！")
        return pd.DataFrame()

    # ==========================================
    # 2. 统计频次
    # ==========================================
    bg_go_counts = {}
    for terms in bg_clean['go_list']:
        for term in set(terms):
            bg_go_counts[term] = bg_go_counts.get(term, 0) + 1

    target_clean_df = bg_clean[bg_clean['Entry'].isin(target_acc_clean)]
    target_go_counts = {}
    for terms in target_clean_df['go_list']:
        for term in set(terms):
            target_go_counts[term] = target_go_counts.get(term, 0) + 1

    # ==========================================
    # 3. 超几何分布检验与 FDR 校正
    # ==========================================
    M = len(background_acc) # 总体大小
    N = len(target_acc_clean) # 样本大小
    
    enrichment_results = []
    for go_term, k in target_go_counts.items():
        n = bg_go_counts[go_term]
        if k < 2: continue # 过滤只出现 1 次的偶发词条
        
        # P-value
        p_value = hypergeom.sf(k - 1, M, n, N)
        # Rich Factor
        rich_factor = (k / N) / (n / M) if n > 0 else 0
        
        enrichment_results.append({
            'GO_Term': go_term,
            'Count': k,
            'Bg_Count': n,
            'P_value': p_value,
            'Rich_Factor': rich_factor
        })

    res_df = pd.DataFrame(enrichment_results)
    
    if res_df.empty:
        print("没有计算出任何富集结果。")
        return res_df
        
    # FDR 多重检验校正
    res_df['FDR'] = multipletests(res_df['P_value'], method='fdr_bh')[1]
    res_df['-log10(FDR)'] = -np.log10(res_df['FDR'] + 1e-10) # 加上一个极小值防止 log(0)
    

    # ==========================================
    # 4. 提取 Top N 准备画图
    # ==========================================
    
    # 取 P-value 最小的 Top N
    plot_df = res_df.sort_values('P_value').head(top_n).copy()
    if plot_df.empty:
        print(f"警告: 经过 FDR < 0.001 过滤后，没有显著的 GO terms!")
        return res_df
    
    # 条形图需要按 Count 升序排列，这样画出来最长的在最上面
    plot_df = plot_df.sort_values('Count', ascending=True)
  
    max_count = plot_df["Count"].max()
    # 根据 max_count 自动选择合适的刻度数量和刻度字体大小
    # 数据集自适应：Asgard 最大值更高，刻度更稀疏
    if max_count >= 2200:   # Asgard 常见区间
        x_pad_ratio = 1.12
        x_nbins = 4
        fs_tick = 15
    else:                   # Ocean 常见区间
        x_pad_ratio = 1.14
        x_nbins = 5
        fs_tick = 15

    plt.figure(figsize=figsize, dpi=600)
    sns.set_theme(style="ticks")

    # 创建一个归一化的颜色映射对象 (映射 Rich Factor)
    norm = mcolors.Normalize(vmin=plot_df['Rich_Factor'].min(), vmax=plot_df['Rich_Factor'].max())
    cmap = plt.get_cmap('viridis')

    # 绘制水平条形图
    bars = plt.barh(
        y=plot_df['GO_Term'], 
        width=plot_df['Count'], 
        color=cmap(norm(plot_df['Rich_Factor'])), 
        edgecolor='black',
        linewidth=0.8,
        height=0.7         # 调整柱子的粗细，让它们之间有呼吸感
    )

    # 1) 给右侧留白，避免柱末数字被截断
    x_max = plot_df['Count'].max()
    ax = plt.gca()
    ax.set_xlim(0, x_max * x_pad_ratio)   # 右侧留 18% 空间（可调 1.12~1.25）
    # 2) 控制 x 轴刻度数量，避免拥挤
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins, integer=True))
    # 调整字体和网格
    plt.yticks(fontsize=fs_tick)
    plt.xticks(fontsize=fs_tick)
    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(
    #         width + (x_max * 0.01), # 稍微偏右一点点
    #         bar.get_y() + bar.get_height() / 2, 
    #         f'{int(width)}', 
    #         va='center', 
    #         ha='left', 
    #         fontsize=FS_GO_BAR_TEXT, 
    #         fontweight='bold',
    #         color='black',
    #         clip_on=False
    #     )

    # 添加 Colorbar (代表 Rich Factor)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # 必须加上这句，空数组用于生成 colorbar
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Rich Factor', fontsize=FS_GO_BAR_TEXT, fontweight='bold')
    cbar.ax.tick_params(labelsize=FS_GO_CBAR_TICK)

    # 设置坐标文本
    plt.xlabel('Number of Proteins', fontsize=FS_GO_AXIS_LABEL, fontweight='bold')
    plt.ylabel('GO Molecular Function', fontsize=FS_GO_AXIS_LABEL, fontweight='bold')

    
    # X轴加上虚线网格，并将网格置于底层，不遮挡柱子
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True) 
    
    # 去除上、右边框
    sns.despine(top=True, right=True)

    # 给 colorbar 和主图留更稳的布局空间
    plt.tight_layout(rect=[0, 0, 0.965, 1])
    # 保存图片
    pic_path = os.path.join(save_dir, "go_enrichment.svg")
    plt.savefig(pic_path, format="svg", bbox_inches='tight', dpi=600)



def max_local_kr_fraction(sequence, window=15):
    """
    计算蛋白质序列中局部最高 K/R 密度
    滑动窗口法：找到 K/R 最密集的 window 个氨基酸窗口
    """
    sequence = sequence.upper()
    if not sequence:
        return np.nan

    kr = np.fromiter(
        (aa in {"K", "R"} for aa in sequence),
        dtype=np.int8,
        count=len(sequence),
    )

    # 少于一个完整窗口的蛋白，用整条蛋白计算
    if len(kr) < window:
        return float(kr.mean())

    cumulative = np.concatenate(([0], np.cumsum(kr)))
    window_counts = cumulative[window:] - cumulative[:-window]
    return float(window_counts.max() / window)


def adjusted_ribosomal_analysis_model1(
    target_df,
    background_df,
    seq_col="Sequence",
    go_col="Gene Ontology (molecular function)",
):
    """
    Model 1: ribosomal_go + kr_fraction + log_length
    """
    required = {"Entry", seq_col, go_col}
    missing = required - set(background_df.columns)
    if missing:
        raise ValueError(f"背景表缺少列: {missing}")

    proteins = (
        background_df
        .dropna(subset=[go_col, seq_col])
        .drop_duplicates(subset=["Entry"])
        .copy()
    )

    selected_acc = set(target_df["acc"].dropna())
    proteins["selected"] = proteins["Entry"].isin(selected_acc).astype(int)

    proteins["sequence"] = proteins[seq_col].astype(str).str.upper()
    proteins["length"] = proteins["sequence"].str.len()
    proteins = proteins[proteins["length"] > 0].copy()

    proteins["kr_fraction"] = (
        proteins["sequence"].str.count("[KR]") / proteins["length"]
    )

    def has_ribosomal_go(value):
        terms = [
            term.split("[GO:")[0].strip()
            for term in str(value).split(";")
        ]
        return "structural constituent of ribosome" in terms

    proteins["ribosomal_go"] = (
        proteins[go_col].apply(has_ribosomal_go).astype(int)
    )
    proteins["log_length"] = np.log(proteins["length"])

    model = smf.glm(
        "selected ~ ribosomal_go + kr_fraction + log_length",
        data=proteins,
        family=sm.families.Binomial(),
    ).fit()

    beta = model.params["ribosomal_go"]
    ci_low, ci_high = model.conf_int().loc["ribosomal_go"]

    return {
        "n_annotated_proteins": len(proteins),
        "n_selected_proteins": int(proteins["selected"].sum()),
        "adjusted_OR": float(np.exp(beta)),
        "CI_95_low": float(np.exp(ci_low)),
        "CI_95_high": float(np.exp(ci_high)),
        "p_value": float(model.pvalues["ribosomal_go"]),
        "model": model,
    }


def adjusted_ribosomal_analysis(
    target_df,
    background_df,
    seq_col="Sequence",
    go_col="Gene Ontology (molecular function)",
):
    required = {"Entry", seq_col, go_col}
    missing = required - set(background_df.columns)
    if missing:
        raise ValueError(f"背景表缺少列: {missing}")

    # 与现有 GO 分析一致：只分析有 GO 注释的被筛查蛋白
    proteins = (
        background_df
        .dropna(subset=[go_col, seq_col])
        .drop_duplicates(subset=["Entry"])
        .copy()
    )

    selected_acc = set(target_df["acc"].dropna())
    proteins["selected"] = proteins["Entry"].isin(selected_acc).astype(int)

    proteins["sequence"] = proteins[seq_col].astype(str).str.upper()
    proteins["length"] = proteins["sequence"].str.len()
    proteins = proteins[proteins["length"] > 0].copy()

    proteins["kr_fraction"] = (
        proteins["sequence"].str.count("[KR]") / proteins["length"]
    )
    
    # 计算局部最高 K/R 密度 (15 aa 窗口)
    proteins["max_kr15"] = proteins["sequence"].apply(
        lambda seq: max_local_kr_fraction(seq, window=15)
    )

    # 使用与原 GO 表相同的条目；先按分号拆开，避免模糊匹配
    def has_ribosomal_go(value):
        terms = [
            term.split("[GO:")[0].strip()
            for term in str(value).split(";")
        ]
        return "structural constituent of ribosome" in terms

    proteins["ribosomal_go"] = (
        proteins[go_col].apply(has_ribosomal_go).astype(int)
    )
    proteins["log_length"] = np.log(proteins["length"])

    model = smf.glm(
        "selected ~ ribosomal_go + kr_fraction + max_kr15 + log_length",
        data=proteins,
        family=sm.families.Binomial(),
    ).fit()

    beta = model.params["ribosomal_go"]
    ci_low, ci_high = model.conf_int().loc["ribosomal_go"]

    return {
        "n_annotated_proteins": len(proteins),
        "n_selected_proteins": int(proteins["selected"].sum()),
        "adjusted_OR": float(np.exp(beta)),
        "CI_95_low": float(np.exp(ci_low)),
        "CI_95_high": float(np.exp(ci_high)),
        "p_value": float(model.pvalues["ribosomal_go"]),
        "model": model,
    }


def run_ribosomal_analysis(target_df, background_df, save_dir, data_name,
                           seq_col="Sequence",
                           go_col="Gene Ontology (molecular function)"):
    """跑两个核糖体 GO 逻辑回归模型(Model 1 / Model 2)，打印摘要并存 TSV。"""
    os.makedirs(save_dir, exist_ok=True)

    models = [
        ("Model 1", "ribosomal_go_model1.tsv", adjusted_ribosomal_analysis_model1),
        ("Model 2", "ribosomal_go_model2.tsv", adjusted_ribosomal_analysis),
    ]

    summaries = []
    for model_label, out_name, runner in models:
        res = runner(target_df=target_df, background_df=background_df,
                     seq_col=seq_col, go_col=go_col)
        summary = {
            "dataset": data_name,
            "model": model_label,
            **{k: v for k, v in res.items() if k != "model"},
        }
        summaries.append(summary)
        print(
            f"  {model_label} 调整后核糖体 GO 的 OR = "
            f"{summary['adjusted_OR']:.3f} "
            f"(95% CI {summary['CI_95_low']:.3f}–"
            f"{summary['CI_95_high']:.3f}), "
            f"P = {summary['p_value']:.3g}"
        )
        pd.DataFrame([summary]).to_csv(
            os.path.join(save_dir, out_name), sep="\t", index=False,
        )

    return summaries


def main():
    """对每个数据集读取高置信度集合 + 物种背景表，跑核糖体 GO 分析。

    运行前请先运行 dual_track_filter.py 生成 high_conf_result.tsv。
    """
    for data_name in ["Ocean", "Asgard"]:
        print(f"start GO / ribosomal analysis for {data_name}")

        data_save_path = DATA_SAVE_FP_DICT['data_save_path'][data_name]
        picture_save_path = DATA_SAVE_FP_DICT['picture_save_path'][data_name]
        save_dir = os.path.join(picture_save_path, "or_filter_emb_tsne_2d")

        target_df = pd.read_csv(os.path.join(data_save_path, "high_conf_result.tsv"), sep="\t")
        background_df = pd.read_csv(
            os.path.join(data_save_path, "species_information", "idmapping.tsv"), sep="\t",
        )

        print("背景表列名：", background_df.columns.tolist())
        run_ribosomal_analysis(
            target_df=target_df,
            background_df=background_df,
            save_dir=save_dir,
            data_name=data_name,
        )


if __name__ == "__main__":
    main()

