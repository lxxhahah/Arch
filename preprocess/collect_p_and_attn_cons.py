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
    PPMI (Positive Pointwise Mutual Information) based scoring
    Softmax -> Lift -> Log -> ReLU
    """
    # attention_matrix: [num_heads, seq_len] --------- raw score
    num_heads, seq_len = attention_matrix.shape
    assert seq_len <= 1022

    # 1. Softmax
    attn_tensor = torch.tensor(attention_matrix)
    probs = F.softmax(attn_tensor, dim=-1)
    # 2. compute Lift (multiple of the random distribution 1/L) --- y * len(seq)
    lift = probs * seq_len
    # 3. Log transform (information content)
    # add a tiny constant to avoid log(0)
    log_lift = torch.log(lift + 1e-9)
    # 4. ReLU truncation (keep only signal above the random level)
    ppmi_matrix = F.relu(log_lift)
    return ppmi_matrix.numpy()


def compute_consensus_ratio(attention_matrix, start, end, top_percent=30):
    """
    Compute the segment's consensus ratio: the mean fraction of heads whose PPMI
    exceeds a per-head threshold at the segment positions (range 0-1).
    attention_matrix: ppmi_score, shape [num_heads, seq_len]
    """
    num_heads, seq_len = attention_matrix.shape

    # 1. per-head threshold, computed over the full length; represents each head's activation level
    raw_thresholds = np.percentile(attention_matrix, 100 - top_percent, axis=1)  # [num_heads]
    thresholds = np.maximum(raw_thresholds, 1e-6)

    # 2. binarize the full-length matrix --- shape: [num_heads, seq_len]
    full_mask = (attention_matrix > thresholds[:, None]).astype(int)

    # 3. consensus rate at each position --- axis=0 averages over heads --- shape: [seq_len]
    global_consensus_vector = np.mean(full_mask, axis=0)

    # 4. mean consensus rate over the segment
    segment_ratios = global_consensus_vector[start:end + 1]
    return float(np.mean(segment_ratios))


def get_valid_seg(start, end, seg, all_sequence):
    """Clip a segment's end to the truncated full length and return the coordinate string."""
    if end >= len(all_sequence):
        return f"{start}_{len(all_sequence) - 1}"
    return seg


_worker_state = {}


def _init_worker(rec_h5_path, esm_h5_path, attn_h5_path):
    """Open the three HDF5 files once per worker process (passed to Pool initializer)."""
    _worker_state["rec"] = h5py.File(rec_h5_path, 'r', libver='latest', swmr=True)
    _worker_state["esm"] = h5py.File(esm_h5_path, 'r', libver='latest', swmr=True)
    _worker_state["attn"] = h5py.File(attn_h5_path, 'r', libver='latest', swmr=True)


def process_accession(acc):
    """
    Process a single accession and return (acc, records, error) for all of its segments.
    HDF5 files are opened once per worker by _init_worker and stored in _worker_state.
    each record: acc / seg / valid_seg / prob / consensus_ratio
    """
    f = _worker_state["rec"]
    esm_file = _worker_state["esm"]
    attn_file = _worker_state["attn"]
    records = []
    try:
        if not (acc in f and acc in esm_file):
            return acc, records, None

        # full-length sequence (truncated to 1022, matching the attention matrix seq_len)
        raw_all_sequence = esm_file[acc]['sequence'][()].decode('utf-8')
        all_sequence = raw_all_sequence[:1022]
        seq_len = len(all_sequence)

        # attention matrix -> PPMI, computed once per accession
        ppmi_matrix = None
        attn_group = attn_file.get(acc)
        if attn_group is not None and 'select_attention_matrix' in attn_group:
            ppmi_matrix = compute_ppmi_score(attn_group['select_attention_matrix'][:])

        for seg in f[acc].keys():
            start, end = map(int, f[acc][seg]['segment'][:])
            prob = float(f[acc][seg]['probability'][()])

            valid_seg = get_valid_seg(start, end, seg, all_sequence)

            consensus_ratio = None
            if ppmi_matrix is not None:
                if ppmi_matrix.shape[1] != seq_len:
                    print(f"[{acc}] attention len {ppmi_matrix.shape[1]} != sequence len {seq_len}, skip consensus")
                else:
                    vs, ve = map(int, valid_seg.split("_"))
                    if vs < seq_len:
                        ve = min(ve, seq_len - 1)
                        consensus_ratio = compute_consensus_ratio(ppmi_matrix, vs, ve)

            records.append({
                'acc': acc,
                'seg': seg,
                'valid_seg': valid_seg,
                'prob': prob,
                'consensus_ratio': consensus_ratio,
            })
    except Exception as e:
        return acc, [], f"{type(e).__name__}: {e}"
    return acc, records, None


def build_result_tsv(rec_h5_path, esm_h5_path, attn_h5_path,
                     output_tsv, data_idx_fp=None, num_processes=8):
    """
    Produce a TSV with acc / seg / valid_seg / prob / consensus_ratio in one pass.
    Failed accessions are written to a sidecar .failed.tsv and reported; raises if
    every accession fails.
    """
    if data_idx_fp is not None and os.path.exists(data_idx_fp):
        with open(data_idx_fp, "r") as f:
            accessions = [idx.strip() for idx in f]
    else:
        with h5py.File(rec_h5_path, 'r') as f:
            accessions = list(f.keys())

    all_records = []
    failures = []
    with Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(rec_h5_path, esm_h5_path, attn_h5_path),
    ) as pool:
        for acc, records, err in tqdm(
            pool.imap_unordered(process_accession, accessions, chunksize=16),
            total=len(accessions), desc="Collecting",
        ):
            if err is not None:
                failures.append((acc, err))
            else:
                all_records.extend(records)

    if len(failures) == len(accessions):
        raise RuntimeError(
            f"all {len(accessions)} accessions failed; check the input HDF5 paths and keys"
        )
    if failures:
        failed_fp = os.path.splitext(output_tsv)[0] + ".failed.tsv"
        with open(failed_fp, "w") as f:
            f.write("acc\terror\n")
            for acc, err in failures:
                f.write(f"{acc}\t{err}\n")
        print(f"WARNING: {len(failures)}/{len(accessions)} accessions failed, written to {failed_fp}")
        for acc, err in failures[:10]:
            print(f"  - {acc}: {err}")

    df = pd.DataFrame(all_records, columns=["acc", "seg", "valid_seg", "prob", "consensus_ratio"])
    df = df.sort_values(["acc", "seg"]).reset_index(drop=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved {len(df):,} records to {output_tsv}")
    return df


if __name__ == "__main__":
    rec_h5_path = "/path/to/output/all_result_dict.h5"   # output of the Recommendation_system
    esm_h5_path = "/path/to/output/all_esm1b_dict.h5"    # ESM features
    attn_h5_path = "/path/to/output/all_attn_dict.h5"    # A2KA attention output
    data_idx_fp = "/path/to/output/all_acc.idx"          # accession index (optional)
    output_tsv = "/path/to/output/all_result.tsv"

    build_result_tsv(
        rec_h5_path=rec_h5_path,
        esm_h5_path=esm_h5_path,
        attn_h5_path=attn_h5_path,
        output_tsv=output_tsv,
        data_idx_fp=data_idx_fp,
        num_processes=8,
    )
