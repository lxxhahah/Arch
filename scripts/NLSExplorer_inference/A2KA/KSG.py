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
        随机扩展或收缩区间至目标长度set_len, 同时确保不超出序列总长度
        segg: 原始区间 (start, end)
        set_len: 目标长度
        max_len: 
    """
    # 检查第一种情况：set_len 大于 max_len
    if set_len > max_len:
        set_len = max_len
    
    fis = segg[0]
    then = segg[1]
    
    # 检查第二种情况：segg 初始区间的长度已经等于或大于 set_len
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
        过滤掉长度为1的单个位点片段, 只保留长度≥2的区间
        pre_seg: [(1,5), (3,3), (7,9)]
    """
    pre_seg_without_single = []
    for th in pre_seg:
        if th[0]!=th[1]:
            pre_seg_without_single.append(th)

    return pre_seg_without_single

def merge_intervals(intervals):
    """
        合并重叠或邻近(距离≤1)的区间
        intervals: [(1,3), (2,4), (6,7)]
    """
    merge_dis = 1

    if not intervals:
        return []

    # 起始位置排序
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        # 检查是否有重合或距离小于1
        if current[0] <= previous[1] + merge_dis:
            # 融合片段
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)

    return merged


def generate_segments(sequence_attention, cofactor, set_len, f_c, seqlen, stret_choose):
    """ 
        sequence_attention: 
        cofactor: 0.3 控制提取关键位置的比例
        set_len: 5 最小片段长度阈值
        f_c: 是否过滤单点位（"f"表示过滤, "n")
        seqlen: 原始序列长度
        stret_choose: 是否启用随机扩展（"s"表示启用, "n")
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
    #选择是否过滤掉单个位点  
    if f_c == "f":
        prepred_seg = filter_single(prepred_seg)

    # 片段长度补齐
    pred_seg = []
    # 选择是否有补齐机制
    if set_len > 2:  # 3
        #在最短长度的基础上根据采样系数随机延长
        add_len = random.choices([0,1,2], [0.5,0.3,0.2])[0]
        set_len = set_len + add_len
        for seg in prepred_seg:
            if seg[1]-seg[0] < set_len:
                pred_seg.append(random_stretch(seg, set_len, seqlen))
            else:
                pred_seg.append(seg)
    # 随机扩展增强
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
        """线程安全的单任务处理"""
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
        返回: {key: [(start1,end1), ...]}
        """

        ctx = mp.get_context('spawn')  # 更安全的进程启动方式
        with ctx.Pool(processes=self.n) as pool:
            results = list(tqdm(
                pool.imap_unordered(self._process_single, self.keys, chunksize=max(1, len(self.keys)//(self.n*4))),  # 每个进程一次处理多个key, 减少通信
                total=len(self.keys),
                desc="Generating segments"
            ))
            
           
                
        # deal with results # 分批写入HDF5
        with h5py.File(save_h5_fp, 'w') as hf:

            hf.attrs.update({
                "processing_date": str(datetime.now()),
                **self.params
            })
           
            # 分批写入
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
  