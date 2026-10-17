import h5py
import random
import numpy as np
import torch
import torch.nn.functional as F

import multiprocessing as mp
import h5py
from tqdm import tqdm
from datetime import datetime 
import warnings
warnings.filterwarnings("ignore")

def random_stretch(segg, set_len, max_len):
    """
        Randomly expand or shrink the interval to target length set_len while staying within the sequence length
        segg: original interval (start, end)
        set_len: target length
        max_len: 
    """
    # case 1: set_len exceeds max_len
    if set_len > max_len:
        set_len = max_len
    
    fis = segg[0]
    then = segg[1]
    
    # case 2: the initial interval is already >= set_len
    if then - fis >= set_len:
        return (0, max_len)
    
    while int(then - fis) != set_len:
        flag = random.choice([0, 1])
        if flag == 0 and fis > 1:
            fis -= 1
        elif then < max_len:
            then += 1
    
    return (fis, then)


def filter_single(pre_seg):
    """
        Drop single-residue fragments of length 1, keeping only intervals of length >= 2
        pre_seg: [(1,5), (3,3), (7,9)]
    """
    pre_seg_without_single = []
    for th in pre_seg:
        if th[0]!=th[1]:
            pre_seg_without_single.append(th)

    return pre_seg_without_single

def merge_intervals(intervals):
    """
        Merge overlapping or adjacent (distance <= 1) intervals
        intervals: [(1,3), (2,4), (6,7)]
    """
    merge_dis = 1

    if not intervals:
        return []

    # sort by start position
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        # check for overlap or distance < 1
        if current[0] <= previous[1] + merge_dis:
            # merge fragments
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)

    return merged


def generate_segments(sequence_attention, cofactor, set_len, f_c, seqlen, stret_choose):
    """ 
        sequence_attention: 
        cofactor: 0.3 controls the fraction of key positions extracted
        set_len: 5 minimum segment length threshold
        f_c: whether to filter single-residue points ("f" filter, "n" no)
        seqlen: original sequence length
        stret_choose: whether to enable random stretch ("s" enable, "n" no)
    """
    y = F.softmax(torch.tensor(sequence_attention).reshape(-1)).cpu().numpy()
    p_y = np.log(300 * y)
    c = F.relu(torch.tensor(p_y))
    length = min(seqlen, 1022)

    recom_num = max(1, int(cofactor * length))
    top_residues = np.sort(np.argsort(-np.array(y))[0:recom_num])   # index
    #top_residues = [res + 1 for res in top_residues]
    
    prepred_seg = []
    
    merg_l = 2
    begin = top_residues[0]
    for i, index in enumerate(top_residues):
        if i+1<len(top_residues) and np.absolute(top_residues[i]-top_residues[i+1]) > merg_l:
            prepred_seg.append((begin,top_residues[i]))
            begin = top_residues[i+1]
    prepred_seg.append((begin, top_residues[-1]))
    # prepred_seg is a list, (start, end)
    #choose whether to filter single-residue points  
    if f_c == "f":
        prepred_seg = filter_single(prepred_seg)

    # pad segment length
    pred_seg = []
    # choose whether padding is enabled
    if set_len > 2:  # 3
        #randomly extend from the minimum length by the sampling factor
        add_len = random.choices([0,1,2], [0.5,0.3,0.2])[0]
        set_len = set_len + add_len
        for seg in prepred_seg:
            if seg[1]-seg[0] < set_len:
                pred_seg.append(random_stretch(seg, set_len, seqlen))
            else:
                pred_seg.append(seg)
    # random-stretch augmentation
    if stret_choose == "s":
        st_li = []
        for seg in pred_seg:
            add_len = random.choices([3,4,5,6,7], [0.3,0.2,0.2,0.2,0.1])[0]
            set_len = seg[1]-seg[0] + add_len
            st_li.append(random_stretch(seg, set_len, seqlen))
        return merge_intervals(st_li)
    
    return merge_intervals(pred_seg)

class KSG:
    def __init__(self, attn_h5_fp, data_idx_fp, cofactor, set_len, f_c, stret_choose, n_process=1):
        """
        """
        self.h5_path = attn_h5_fp
        with open(data_idx_fp, "r") as f:
            self.keys = [idx.strip() for idx in f]


        self.params = {
            'cofactor': cofactor,
            'set_len': set_len,
            'f_c': f_c,
            'stret_choose': stret_choose
        }
        

        self.n = n_process

   
    def _process_single(self, key):
        """Thread-safe single-task processing."""
        with h5py.File(self.h5_path, "r") as f:
            attn = np.array(f[key]['select_attention'][:])
            seqlen = len(f[key]['sequence'][()])

        segs = generate_segments(
            sequence_attention= attn,
            seqlen=seqlen,
            **self.params)
        
        return key, np.array(segs, dtype=np.int32) 


    def parallel_generate(self, save_h5_fp, batch_sie=1024):
        """
        Returns: {key: [(start1,end1), ...]}
        """

        ctx = mp.get_context('spawn')  # safer process start method
        with ctx.Pool(processes=self.n) as pool:
            results = list(tqdm(
                pool.imap_unordered(self._process_single, self.keys, chunksize=max(1, len(self.keys)//(self.n*4))),  # each process handles multiple keys at once to reduce communication
                total=len(self.keys),
                desc="Generating segments"
            ))
            
           
                
        # deal with results # write to HDF5 in batches
        with h5py.File(save_h5_fp, 'w') as hf:

            hf.attrs.update({
                "processing_date": str(datetime.now()),
                **self.params
            })
           
            # write in batches
            batch = {}
            for i, (key, segs) in enumerate(results):
                batch[key] = segs

                if len(batch) >= batch_sie:
                    self._save_batch(hf, batch)
                    batch = {}
            
            if batch: 
                self._save_batch(hf, batch)


           

    def _save_batch(self, h5_file, batch):
        for key, segs in batch.items():
            grp = h5_file.create_group(key)
            grp.create_dataset(
                "segments",
                data = segs,
                dtype=np.int32
            )
         
if __name__ == "__main__":
    CONFIG = {
        "attn_h5_fp": "/path/to/output/all_attn_dict.h5",
        "data_idx_fp": "/path/to/output/all_acc.idx",
        "cofactor": 0.3,
        "set_len": 5,
        "f_c": "n",
        "stret_choose": "n",
        "n_process": 10
    }

    ksg = KSG(**CONFIG)
    save_h5_fp = "/path/to/output/all_generate_segs_dict.h5"
    ksg.parallel_generate(save_h5_fp)
  