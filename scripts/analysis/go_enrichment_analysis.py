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
                       x_limit=None,     # X-axis limit, used to align the two plots
    ):

    # -------------------------
    # Local font config (for GO plot only)
    # -------------------------
    FS_GO_AXIS_LABEL = 30     # x/y axis title
    FS_GO_TICK = 20           # x/y tick labels
    FS_GO_BAR_TEXT = 15       # value text at the right of each bar
    FS_GO_CBAR_TICK = 30      # colorbar ticks

    os.makedirs(save_dir, exist_ok=True)
    
    if 'acc' not in target_df.columns or 'Entry' not in background_df.columns:
        raise ValueError("target_df must contain an 'acc' column and background_df must contain an 'Entry' column")
        
    target_acc_list = target_df['acc'].dropna().unique().tolist()
    
    # drop background proteins without the current GO annotation
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
        print(f"Warning: no proteins with the {go_col} annotation were found in the target set!")
        return pd.DataFrame()

    # ==========================================
    # 2. Count frequencies
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
    # 3. Hypergeometric test and FDR correction
    # ==========================================
    M = len(background_acc) # population size
    N = len(target_acc_clean) # sample size
    
    enrichment_results = []
    for go_term, k in target_go_counts.items():
        n = bg_go_counts[go_term]
        if k < 2: continue # drop terms that only appear once
        
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
        print("No enrichment results were computed.")
        return res_df
        
    # FDR multiple-testing correction
    res_df['FDR'] = multipletests(res_df['P_value'], method='fdr_bh')[1]
    res_df['-log10(FDR)'] = -np.log10(res_df['FDR'] + 1e-10) # add a tiny constant to avoid log(0)
    

    # ==========================================
    # 4. Extract the Top N for plotting
    # ==========================================
    
    # take the Top N by smallest P-value
    plot_df = res_df.sort_values('P_value').head(top_n).copy()
    if plot_df.empty:
        print(f"Warning: no significant GO terms after FDR < 0.001 filtering!")
        return res_df
    
    # bars must be sorted ascending by Count so the longest is on top
    plot_df = plot_df.sort_values('Count', ascending=True)
  
    max_count = plot_df["Count"].max()
    # pick tick count and font size based on max_count
    # dataset-adaptive: Asgard has a higher max, so ticks are sparser
    if max_count >= 2200:   # common Asgard range
        x_pad_ratio = 1.12
        x_nbins = 4
        fs_tick = 15
    else:                   # common Ocean range
        x_pad_ratio = 1.14
        x_nbins = 5
        fs_tick = 15

    plt.figure(figsize=figsize, dpi=600)
    sns.set_theme(style="ticks")

    # create a normalized colormap for Rich Factor
    norm = mcolors.Normalize(vmin=plot_df['Rich_Factor'].min(), vmax=plot_df['Rich_Factor'].max())
    cmap = plt.get_cmap('viridis')

    # draw a horizontal bar chart
    bars = plt.barh(
        y=plot_df['GO_Term'], 
        width=plot_df['Count'], 
        color=cmap(norm(plot_df['Rich_Factor'])), 
        edgecolor='black',
        linewidth=0.8,
        height=0.7         # bar thickness
    )

    # 1) leave room on the right so bar-end numbers are not clipped
    x_max = plot_df['Count'].max()
    ax = plt.gca()
    ax.set_xlim(0, x_max * x_pad_ratio)   # leave 18% space on the right (tunable 1.12~1.25)
    # 2) limit x-axis tick count to avoid crowding
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins, integer=True))
    # adjust fonts and grid
    plt.yticks(fontsize=fs_tick)
    plt.xticks(fontsize=fs_tick)
    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(
    #         width + (x_max * 0.01), # slightly to the right
    #         bar.get_y() + bar.get_height() / 2, 
    #         f'{int(width)}', 
    #         va='center', 
    #         ha='left', 
    #         fontsize=FS_GO_BAR_TEXT, 
    #         fontweight='bold',
    #         color='black',
    #         clip_on=False
    #     )

    # add a colorbar for Rich Factor
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # required: empty array to build the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Rich Factor', fontsize=FS_GO_BAR_TEXT, fontweight='bold')
    cbar.ax.tick_params(labelsize=FS_GO_CBAR_TICK)

    # set axis labels
    plt.xlabel('Number of Proteins', fontsize=FS_GO_AXIS_LABEL, fontweight='bold')
    plt.ylabel('GO Molecular Function', fontsize=FS_GO_AXIS_LABEL, fontweight='bold')

    
    # dashed x-grid placed behind the bars
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True) 
    
    # remove top/right spines
    sns.despine(top=True, right=True)

    # stable layout space for colorbar and main plot
    plt.tight_layout(rect=[0, 0, 0.965, 1])
    # save figure
    pic_path = os.path.join(save_dir, "go_enrichment.svg")
    plt.savefig(pic_path, format="svg", bbox_inches='tight', dpi=600)



def max_local_kr_fraction(sequence, window=15):
    """
    Compute the local maximum K/R density of a protein sequence
    using a sliding window: find the window of `window` residues with the most K/R
    """
    sequence = sequence.upper()
    if not sequence:
        return np.nan

    kr = np.fromiter(
        (aa in {"K", "R"} for aa in sequence),
        dtype=np.int8,
        count=len(sequence),
    )

    # for proteins shorter than one window, use the whole sequence
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
        raise ValueError(f"background table missing columns: {missing}")

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
        raise ValueError(f"background table missing columns: {missing}")

    # consistent with the existing GO analysis: only analyze screened proteins with GO annotation
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
    
    # compute the local max K/R density (15 aa window)
    proteins["max_kr15"] = proteins["sequence"].apply(
        lambda seq: max_local_kr_fraction(seq, window=15)
    )

    # use the same terms as the original GO table; split on semicolons to avoid fuzzy matches
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
    """Run the two ribosomal GO logistic-regression models (Model 1 / Model 2), print summaries, and save TSVs."""
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
            f"  {model_label} adjusted ribosomal-GO OR = "
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
    """For each dataset, load the high-confidence set + species background table and run the ribosomal GO analysis.

    Run dual_track_filter.py first to produce high_conf_result.tsv.
    """
    for data_name in ["Ocean", "Asgard"]:
        print(f"start GO / ribosomal analysis for {data_name}")

        data_save_path = DATA_SAVE_FP_DICT['data_save_path'][data_name]
        picture_save_path = DATA_SAVE_FP_DICT['picture_save_path'][data_name]
        save_dir = os.path.join(picture_save_path, "emb_tsne_2d")

        target_df = pd.read_csv(os.path.join(data_save_path, "high_conf_result.tsv"), sep="\t")
        background_df = pd.read_csv(
            os.path.join(data_save_path, "species_information", "idmapping.tsv"), sep="\t",
        )
        plot_go_enrichment(
            target_df=target_df,
            background_df=background_df,
            save_dir=save_dir,
            go_col="Gene Ontology (molecular function)",
            top_n=10,
        )
        print("background table columns:", background_df.columns.tolist())
        run_ribosomal_analysis(
            target_df=target_df,
            background_df=background_df,
            save_dir=save_dir,
            data_name=data_name,
        )


if __name__ == "__main__":
    main()

