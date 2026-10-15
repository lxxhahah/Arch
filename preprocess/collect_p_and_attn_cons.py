import os
import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

import torch
import torch.nn.functional as F


def compute_ppmi_score(attention_matrix):
    """
    基于 PPMI (Positive Pointwise Mutual Information) 的评分算法
    Softmax -> Lift -> Log -> ReLU
    """
    # attention_matrix: [num_heads, seq_len] --------- raw score
    num_heads, seq_len = attention_matrix.shape
    assert seq_len <= 1022

    # 1. Softmax
    attn_tensor = torch.tensor(attention_matrix)
    probs = F.softmax(attn_tensor, dim=-1)
    # 2. compute Lift (相对于随机分布 1/L 的倍数) --- y * len(seq)
    lift = probs * seq_len
    # 3. Log 变换 (计算信息量)
    # 加一个极小值防止 log(0)
    log_lift = torch.log(lift + 1e-9)
    # 4. ReLU 截断 (只保留高于随机水平的信号)
    ppmi_matrix = F.relu(log_lift)
    return ppmi_matrix.numpy()


def compute_binary_z_score(attention_matrix, start, end, top_percent=30):
    """
    保留二值化逻辑，引入全长背景 Z-Score
    attention_matrix: ppmi_score
    """
    num_heads, seq_len = attention_matrix.shape

    # 1. 计算每个头的阈值 -----------基于全长计算的，代表了每个 Head 的活跃标准
    raw_thresholds = np.percentile(attention_matrix, 100 - top_percent, axis=1)  # [num_heads]
    thresholds = np.maximum(raw_thresholds, 1e-6)

    # 2. 对全长矩阵进行二值化--------shape: [num_heads, seq_len]
    full_mask = (attention_matrix > thresholds[:, None]).astype(int)

    # 3. 计算全长每个位置的共识率-------axis=0 表示跨 Head 求和/平均-----------shape: [seq_len]
    global_consensus_vector = np.mean(full_mask, axis=0)

    # 4. 提取片段的得分
    segment_ratios = global_consensus_vector[start:end + 1]
    segment_score = np.mean(segment_ratios)

    # 5. 计算 Z-Score(exclude segment itself)
    bg_mask = np.ones(seq_len, dtype=bool)
    bg_mask[start:end + 1] = False
    bg_values = global_consensus_vector[bg_mask]

    if len(bg_values) > 0:
        bg_mu = np.mean(bg_values)
        bg_std = np.std(bg_values) + 1e-9
        z_score = (segment_score - bg_mu) / bg_std
    else:
        z_score = 0

    return z_score, segment_score


def get_valid_seg(start, end, seg_sequence, seg, all_sequence):
    """
    校验 segment 是否越界，越界则裁剪。
    若片段序列长度与实际区间不一致，且 end 超出截断后的全长，则把 end 修正为 len(all_sequence)-1。
    """
    valid_seg = seg
    if len(seg_sequence) != int(end - start + 1):
        if end >= len(all_sequence):
            valid_seg = f"{start}_{len(all_sequence) - 1}"
        else:
            print(f"[{seg}] length mismatch but end < full len {len(all_sequence)}, keep original")
    return valid_seg


def process_accession(args):
    """
    处理单个 accession，返回该 accession 下所有 segment 的记录。
    args: (acc, rec_h5_path, esm_h5_path, attn_h5_path)
    每条记录: acc / seg / valid_seg / prob / consensus_ratio
    """
    acc, rec_h5_path, esm_h5_path, attn_h5_path = args
    records = []
    try:
        with h5py.File(rec_h5_path, 'r', libver='latest', swmr=True) as f, \
             h5py.File(esm_h5_path, 'r', libver='latest', swmr=True) as esm_file, \
             h5py.File(attn_h5_path, 'r', libver='latest', swmr=True) as attn_file:

            if not (acc in f and acc in esm_file):
                return records

            # 全长序列(截断到 1022，与 attention matrix 的 seq_len 一致)
            raw_all_sequence = esm_file[acc]['sequence'][()].decode('utf-8')
            all_sequence = raw_all_sequence[:1022]

            # attention matrix -> PPMI，每个 acc 只算一次
            ppmi_matrix = None
            attn_group = attn_file.get(acc)
            if attn_group is not None and 'select_attention_matrix' in attn_group:
                ppmi_matrix = compute_ppmi_score(attn_group['select_attention_matrix'][:])

            for seg in f[acc].keys():
                start, end = map(int, f[acc][seg]['segment'][:])
                seg_sequence = f[acc][seg]['sequence'][()].decode('utf-8')
                prob = float(f[acc][seg]['probability'][()])

                valid_seg = get_valid_seg(start, end, seg_sequence, seg, all_sequence)

                consensus_ratio = None
                if ppmi_matrix is not None:
                    vs, ve = map(int, valid_seg.split("_"))
                    seq_len = ppmi_matrix.shape[1]
                    
                    if vs < seq_len:
                        ve = min(ve, seq_len - 1)
                        _, consensus_ratio = compute_binary_z_score(ppmi_matrix, vs, ve)

                records.append({
                    'acc': acc,
                    'seg': seg,
                    'valid_seg': valid_seg,
                    'prob': prob,
                    'consensus_ratio': consensus_ratio,
                })
    except Exception as e:
        print(f"process {acc} occur error: {e}")
    return records


def build_result_tsv(rec_h5_path, esm_h5_path, attn_h5_path,
                     output_tsv, data_idx_fp=None, num_processes=8):
    """
    一次性产出包含 acc / seg / valid_seg / prob / consensus_ratio 的 TSV。
    """
    if data_idx_fp is not None and os.path.exists(data_idx_fp):
        with open(data_idx_fp, "r") as f:
            accessions = [idx.strip() for idx in f]
    else:
        with h5py.File(rec_h5_path, 'r') as f:
            accessions = list(f.keys())

    tasks = [(acc, rec_h5_path, esm_h5_path, attn_h5_path) for acc in accessions]

    all_records = []
    with Pool(processes=num_processes) as pool:
        for records in tqdm(
            pool.imap_unordered(process_accession, tasks, chunksize=16),
            total=len(tasks), desc="Collecting",
        ):
            all_records.extend(records)

    df = pd.DataFrame(all_records, columns=["acc", "seg", "valid_seg", "prob", "consensus_ratio"])
    df = df.sort_values(["acc", "seg"]).reset_index(drop=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved {len(df):,} records to {output_tsv}")
    return df


if __name__ == "__main__":
    rec_h5_path = "/path/to/output/all_result_dict.h5"   # Recommendation_system 输出
    esm_h5_path = "/path/to/output/all_esm1b_dict.h5"    # ESM 特征
    attn_h5_path = "/path/to/output/all_attn_dict.h5"    # A2KA 注意力输出
    data_idx_fp = "/path/to/output/all_acc.idx"          # accession 索引(可选)
    output_tsv = "/path/to/output/all_result.tsv"

    build_result_tsv(
        rec_h5_path=rec_h5_path,
        esm_h5_path=esm_h5_path,
        attn_h5_path=attn_h5_path,
        output_tsv=output_tsv,
        data_idx_fp=data_idx_fp,
        num_processes=8,
    )
