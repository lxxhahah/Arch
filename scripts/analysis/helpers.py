import os
from pathlib import Path

import pandas as pd
import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
import yaml

from collections import defaultdict, OrderedDict


# ---- Path config: reads config.yaml (placeholder paths; fill in your own absolute paths)----
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    DATA_SAVE_FP_DICT = yaml.safe_load(_f)

# Experimental validation labels (not paths, kept as a data file committed with the code)
EXPERIMENT_LABELS_TSV = Path(__file__).resolve().parent / "experiment_labels.tsv"


def load_experiment_labels(tsv_path=EXPERIMENT_LABELS_TSV):
    """Load experimental validation labels, returning {dataset: {label: [seg_key, ...]}}."""
    df = pd.read_csv(tsv_path, sep="\t")
    result = {}
    for dataset, dg in df.groupby("dataset"):
        result[dataset] = {label: g["seg_key"].tolist() for label, g in dg.groupby("label")}
    return result




def process_row(idx, acc, seg, h5_path, esm_h5_path, emb_flag = False):
    """
    return  (idx, payload_or_None)
        payload = (embedding, esm, seg_key, seg_info) -------  if not emb_flag: embedding & esm is None
        seg_info = [acc, valid_seg, all_sequence, seg_sequence, prob, seg]

    """
    try:
        with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f, \
             h5py.File(esm_h5_path, 'r', libver='latest', swmr=True) as esm_file:

            if not (acc in f and acc in esm_file):
                return idx, None
            
           
            start, end = f[acc][seg]['segment'][:] # numpy 
            embedding, esm = None, None
            if emb_flag:
                embedding = np.squeeze(f[acc][seg]['embedding'][:])
                # esm embedding: mean pooling
                esm_full = esm_file[acc]['embedding'][:]  # (L, 1280)
                esm = esm_full[start:end+1].mean(0)

            raw_all_sequence = esm_file[acc]["sequence"][()].decode('utf-8') # raw
            all_sequence = raw_all_sequence[:1022]
            seg_sequence = f[acc][seg]["sequence"][()].decode('utf-8')
            prob = float(f[acc][seg]["probability"][()])

            valid_seg = seg
            if len(seg_sequence) != int(end - start + 1):
                flag = False
                if end >= len(all_sequence):
                    flag = True
                    # clip end to the valid (in-bounds) coordinate
                    valid_seg = f"{start}_{len(all_sequence)-1}"

                if not flag:
                    print(f"{valid_seg}, Full len is {len(all_sequence)}")

            seg_key = f"{acc}_{valid_seg}"
            seg_info = [acc, valid_seg, all_sequence, seg_sequence, prob, seg]

            return idx, (embedding, esm, seg_key, seg_info)

    except Exception as e:
        print(f"process {acc}_{seg} occur error: {e}")
        return idx, None


def process_row_wrapper(args):
    # args: (idx, acc, seg, h5_path, esm_h5_path, emb_flag)
    return process_row(*args)


def load_arch_emb_multiprocess(h5_path, esm_h5_path, recommend_df, emb_flag = False, num_processes=None):
    """
    return:
      all_embeddings: (N, 640) or None
      all_esms: (N, 1280) or None
      seg_dict: {seg_key: [acc, valid_seg, all_sequence, seg_sequence, prob, seg]}
    """
    # recommend_df = (recommend_df
    #                 .sort_values(by=['recommend_prob', 'acc', 'seg'],
    #                              ascending=[False, True, True],
    #                              kind='mergesort')
    #                 .reset_index(drop=True))
    recommend_df = recommend_df.reset_index(drop=True)
    n = len(recommend_df)

    tasks = [(i, recommend_df.at[i, 'acc'], recommend_df.at[i, 'seg'], h5_path, esm_h5_path, emb_flag)
             for i in range(n)]

    emb_list = [None] * n
    esm_list = [None] * n
    key_list = [None] * n
    info_list = [None] * n

    with mp.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_row_wrapper, tasks, chunksize=64)
        for idx, payload in tqdm(results_iterator, total=n):
            if payload is None:
                continue
            embedding, esm, seg_key, seg_info = payload
            
            emb_list[idx] = embedding
            esm_list[idx] = esm
            key_list[idx] = seg_key
            info_list[idx] = seg_info


    keep = [i for i in range(n) if info_list[i] is not None]

    if len(keep) == 0:
        raise RuntimeError("No valid samples were read (all returned None). Check whether the HDF5 keys match.")
    print(f"Parallel read done: {len(keep)}/{n} succeeded.")

    all_embeddings, all_esms = None, None

    if emb_flag:
        all_embeddings = np.stack([emb_list[i] for i in keep], axis=0)
        all_esms = np.stack([esm_list[i] for i in keep], axis=0)

    seg_dict = {key_list[i]: info_list[i] for i in keep}


    return all_embeddings, all_esms, seg_dict


def get_check_df(h5_path, esm_h5_path, df, num_processes=None):
    """
        return result_df: "acc", "valid_seg", "prob", "seg_key", "seg"
               all_emb,
               all_esm,
               full_seq_dict
               seg_seq_dict
    """
    if num_processes is None:
        num_processes = os.cpu_count()

    print(f"Reading data in parallel with {num_processes} processes...")   
    all_emb, all_esm, seg_dict = load_arch_emb_multiprocess(h5_path, esm_h5_path, df, emb_flag=True, num_processes=num_processes)

    data_rows = []
    full_seq_dict = {}  # {Accession: Full_Sequence}
    seg_seq_dict = {}   # {Segment_Key: Segment_Sequence}

    for seg_key, info_list in seg_dict.items():
        acc, valid_seg, all_sequence, seg_sequence, prob, seg = info_list

        full_seq_dict[acc] = all_sequence
        seg_seq_dict[seg_key] = seg_sequence

        row = {
            "acc": acc,
            "valid_seg": valid_seg,
            "seg_key": seg_key,
            "prob": prob,
            "seg": seg
        }
        data_rows.append(row)

    result_df = pd.DataFrame(data_rows)

    return result_df, all_emb, all_esm, full_seq_dict, seg_seq_dict


def parse_cluster_tsv(tsv_path, min_members = None, top_n = None):

    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['cluster_rep', 'member'])
    # {cluster_rep_id: [member1, member2, ...]} =====
    
    cluster_dict = defaultdict(list)
    for rep, member in zip(df['cluster_rep'], df['member']):
        cluster_dict[rep].append(member)

    filtered_cluster_dict = {}
    sorted_cluster_dict = OrderedDict(
        sorted(cluster_dict.items(), key=lambda x: len(x[1]), reverse=True)
    )
    if min_members is not None:
        filtered_cluster_dict = {rep: members for rep, members in sorted_cluster_dict.items() if len(members) >= min_members}
    
    elif top_n is not None:
        filtered_cluster_dict = dict(list(sorted_cluster_dict.items())[:top_n])
     
    return sorted_cluster_dict, filtered_cluster_dict




