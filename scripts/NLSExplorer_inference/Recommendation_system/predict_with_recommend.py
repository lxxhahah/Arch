import pickle
import torch
from recommend_network import gnn_classifier
from collections import OrderedDict
import math
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import sys

class H5SequenceDataset(Dataset):
    """Memory-mapped HDF5 reader."""
    def __init__(self, emb_seq_path, pre_seg_path, keys):
        self.emb_seq_path = emb_seq_path
        self.pre_seg_path = pre_seg_path

        self.keys = keys
        self.emb_file = None
        self.seg_file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.emb_file is None:
            self.emb_file = h5py.File(self.emb_seq_path, 'r', libver='latest', swmr=True)
        if self.seg_file is None:
            self.seg_file = h5py.File(self.pre_seg_path, 'r', libver='latest', swmr=True)

        acc = self.keys[idx]
        emb = torch.from_numpy(self.emb_file[acc]['embedding'][:])
        seq = self.emb_file[acc]['sequence'].asstr()[()]
        pre_seg =self.seg_file[acc]["segments"][:]

        return acc, emb, pre_seg, seq
    
           


class SegmentPredictor:
    def __init__(self, model, device, max_workers):
        self.device = device
        self.model = model.to(device).eval()
        # add CUDA synchronization
        torch.cuda.synchronize(device)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)


    def predict_single(self, acc, emb, seg, seq):
        """Predict segments for a single protein."""
        with torch.no_grad(), torch.cuda.device(self.device):
            emb = emb.unsqueeze(0).to(self.device, non_blocking = True)  # (1, seq_len, 1280)
            pre, emb_li = self.model(emb, [seg])  # pre: (1, num_seg)
            # emb_li is a list    pre: [1, num_seg] -> num_seg
            sorted_indices = torch.argsort(pre.squeeze(0), descending=True)
            sorted_dict = OrderedDict()

            for idx in sorted_indices:
                idx = idx.item()
                start, end = seg[idx]
                sorted_dict[f"{start}_{end}"] = {
                    "probability": float(pre[0, idx].item()),
                    "embedding": emb_li[idx].detach().cpu().numpy(),
                    "segment": np.array([start, end]), 
                    "sequence": seq[start:end + 1]
                }
        
            return acc, sorted_dict    
          
    def process_batch(self, batch):
        """ 
            batch: [acc_list, emb_list, pre_seg_list, seq_list]
        """
        futures = []
        for acc, emb, seg, seq in zip(*batch):
            fut = self.executor.submit(self.predict_single, acc, emb, seg, seq)
            futures.append(fut)
        
        # wait and return the results
        return [f.result() for f in futures]
      
def process_all(model, device, emb_seq_h5, pre_seg_h5, output_h5_path, keys,
                batch_size=64, num_workers=4):

    dataset = H5SequenceDataset(emb_seq_h5, pre_seg_h5, keys)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: list(zip(*x)),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    predictor = SegmentPredictor(model, device, max_workers=6)

    with h5py.File(output_h5_path, 'w', libver='latest') as hf:
        for batch in tqdm(loader, desc=f"Processing {output_h5_path}"):
            results = predictor.process_batch(batch)
            for acc, sorted_dict in results:
                grp = hf.create_group(acc)
                for seg_id, data in sorted_dict.items():
                    seg_grp = grp.create_group(seg_id)
                    for k, v in data.items():
                        seg_grp.create_dataset(k, data=v)


def read_keys(data_idx_fp):
    with open(data_idx_fp, "r") as f:
        return [idx.strip() for idx in f]


def main():
    device = "cuda:0"              
    torch.cuda.set_device(device)

    model_path = "../checkpoint/RENOZDM10"
    model = gnn_classifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    emb_seq_h5    = "/path/to/output/all_esm1b_dict.h5"
    data_idx_fp   = "/path/to/output/all_acc.idx"
    pre_seg_h5    = "/path/to/output/all_generate_segs_dict.h5"

    output_h5_path = "/path/to/output/all_result_dict.h5"

    keys = read_keys(data_idx_fp)
    print(f"n_keys = {len(keys)}")
    process_all(model, device, emb_seq_h5, pre_seg_h5, output_h5_path, keys)


if __name__ == "__main__":
    main()



   

