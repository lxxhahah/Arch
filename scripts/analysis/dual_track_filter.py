"""
Dual-track filter:
  1) all_result.tsv (all candidate segments) -> candidate_result.tsv (prob >= 0.7)
  2) candidate_result.tsv -> high_conf_result.tsv (dual-track high-confidence set)

Run preprocess/collect_p_and_attn_cons.py first to produce all_result.tsv
(columns: acc / seg / valid_seg / prob / consensus_ratio).

Downstream:
  - latent_space.py reads high_conf_result.tsv for embedding visualization
  - go_analysis.py   reads high_conf_result.tsv for GO / ribosomal analysis
"""
import os

import numpy as np
import pandas as pd

from helpers import DATA_SAVE_FP_DICT


INITIAL_PROB_THRESHOLD = 0.7   # probability threshold for the initial candidate pool
HIGH_PROB_THRESHOLD = 0.95     # "high probability" track in the dual-track filter


def load_all_result(all_result_tsv):
    """Load all candidate segments and compute seg_key = acc + "_" + valid_seg."""
    df = pd.read_csv(all_result_tsv, sep="\t")
    df["seg_key"] = df["acc"] + "_" + df["valid_seg"]
    return df


def build_initial_candidate(df, prob_threshold=INITIAL_PROB_THRESHOLD):
    """Keep segments with recommendation probability >= prob_threshold as the initial candidate pool."""
    return df[df["prob"] >= prob_threshold].copy()


def apply_dual_track_filter(df, high_prob_threshold=HIGH_PROB_THRESHOLD):
    """
    Dual-track filter:
      Track A: extremely high recommendation probability (prob >= high_prob_threshold)
      Track B: moderate probability but consensus ratio not below the pool mean (consensus_ratio >= mean)
    """
    attn_threshold = np.nanmean(df["consensus_ratio"])
    cond_A = df["prob"] >= high_prob_threshold
    cond_B = (df["prob"] < high_prob_threshold) & (df["consensus_ratio"] >= attn_threshold)
    return df[cond_A | cond_B].copy()


def run(data_name):
    data_save_path = DATA_SAVE_FP_DICT["data_save_path"][data_name]
    all_result_tsv = os.path.join(data_save_path, "all_result.tsv")
    candidate_tsv = os.path.join(data_save_path, "candidate_result.tsv")
    high_conf_tsv = os.path.join(data_save_path, "high_conf_result.tsv")

    df = load_all_result(all_result_tsv)
    print(f"[{data_name}] all candidate segments: {len(df):,}")

    candidate = build_initial_candidate(df)
    candidate.to_csv(candidate_tsv, sep="\t", index=False)
    print(f"[{data_name}] initial candidate pool (prob >= {INITIAL_PROB_THRESHOLD}): "
          f"{len(candidate):,} -> {candidate_tsv}")

    high_conf = apply_dual_track_filter(candidate)
    high_conf.to_csv(high_conf_tsv, sep="\t", index=False)
    print(f"[{data_name}] high-confidence set: {len(high_conf):,} -> {high_conf_tsv}")
    return high_conf


def main():
    for data_name in ["Ocean", "Asgard"]:
        run(data_name)


if __name__ == "__main__":
    main()
