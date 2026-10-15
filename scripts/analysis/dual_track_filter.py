"""
双轨筛选 (dual-track filter)：
  1) 全量候选片段 all_result.tsv -> 推荐概率 >= 0.7 的初始候选池 candidate_result.tsv
  2) 初始候选池 -> 双轨筛选 -> 高置信度集合 high_conf_result.tsv

运行前请先用 preprocess/collect_p_and_attn_cons.py 生成 all_result.tsv
(列为 acc / seg / valid_seg / prob / consensus_ratio)。

下游：
  - latent_space.py 消费 high_conf_result.tsv 做 embedding 可视化
  - go_analysis.py   消费 high_conf_result.tsv 做 GO / 核糖体分析
"""
import os

import numpy as np
import pandas as pd

from helpers import DATA_SAVE_FP_DICT


INITIAL_PROB_THRESHOLD = 0.7   # 初始候选池阈值（推荐概率）
HIGH_PROB_THRESHOLD = 0.95     # 双轨筛选中的"高概率"轨道


def load_all_result(all_result_tsv):
    """读取全量候选片段结果，并补算 seg_key = acc + "_" + valid_seg。"""
    df = pd.read_csv(all_result_tsv, sep="\t")
    df["seg_key"] = df["acc"] + "_" + df["valid_seg"]
    return df


def build_initial_candidate(df, prob_threshold=INITIAL_PROB_THRESHOLD):
    """推荐概率 >= prob_threshold 的片段作为初始候选池。"""
    return df[df["prob"] >= prob_threshold].copy()


def apply_dual_track_filter(df, high_prob_threshold=HIGH_PROB_THRESHOLD):
    """
    双轨筛选：
      轨道 A：极高推荐概率 (prob >= high_prob_threshold)
      轨道 B：中等概率但注意力共识度不低于初始候选池均值 (consensus_ratio >= mean)
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
    print(f"[{data_name}] 全量候选片段: {len(df):,}")

    candidate = build_initial_candidate(df)
    candidate.to_csv(candidate_tsv, sep="\t", index=False)
    print(f"[{data_name}] 初始候选池 (prob >= {INITIAL_PROB_THRESHOLD}): "
          f"{len(candidate):,} -> {candidate_tsv}")

    high_conf = apply_dual_track_filter(candidate)
    high_conf.to_csv(high_conf_tsv, sep="\t", index=False)
    print(f"[{data_name}] 高置信度集合: {len(high_conf):,} -> {high_conf_tsv}")
    return high_conf


def main():
    for data_name in ["Ocean", "Asgard"]:
        run(data_name)


if __name__ == "__main__":
    main()
