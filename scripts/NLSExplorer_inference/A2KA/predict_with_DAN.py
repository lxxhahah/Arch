import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))  
from tqdm import tqdm
import h5py
import pickle
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from DAN import LSTMTagger



def get_distribu_with_select_batch(model, embedding, mode='sum'):
    """
    if embedding: is a list, each element responding to a sequence embedding(seq_len, 1280) 
        embedding = list(embedding) + [torch.randn(1022, 1280)]
        embedding = pad_sequence(embedding, batch_first=True).cuda().float()[:-1,:,:] # (bz, 1022, 1280)

    Args:
        return attn_dis(per-layer aggregated attention scores summed), combined(selected heads' attention scores summed)
        embedding: assume that (bz, 1022, 1280)
        select_li: head selection template [num_layers, num_heads] (0/1 matrix)
        mode: fusion mode ('mean' or 'sum')
    """

    # load the selection template; select_li is a nested list of shape [16, 64]
    # each element is 0 or 1; 1 means keep that head's attention
    f_read = open("../checkpoint/per80_select_li.pkl", 'rb')
    select_li = pickle.load(f_read)
    f_read.close()
    select_li = torch.tensor(select_li).bool().cuda()

    with torch.no_grad():
        model.eval()
        # P, attention_dis, total_attention_store, final_att 
        # where all_heads[layer_idx][head_idx] has shape [batch_size, seq_len, 1]
        dan_prob, att_dis, all_heads, _ = model(embedding)   
        dan_prob = dan_prob.squeeze(-1).cpu().detach().numpy()  # [bz, ]

        # att_dis is a list, each element store each layer responding to combined attention score
        # all_heads
        batch_size, seq_len = embedding.shape[0], embedding.shape[1]
        # store the attention of each selected head
        selected_heads_att = []
        combined = torch.zeros(batch_size, seq_len).cuda()
        selected_counts = 0
        
        for layer in range(select_li.shape[0]): # num_layer -> 0 : num_layer - 1
            for head in range(select_li.shape[1]):
                if select_li[layer][head]:
                    head_att = all_heads[layer][head].squeeze(-1)  # [batch, seq_len]

                    selected_heads_att.append(head_att)
                    combined += head_att
                    selected_counts += 1
        
        if mode == 'mean' and selected_counts > 0:
            combined = combined / selected_counts

    selected_heads_att = torch.stack(selected_heads_att).cpu().detach().numpy()  # [num_selected_heads, bz, seq_len]
    selected_heads_att = selected_heads_att.transpose(1, 0, 2)  # [bz, num_selected_heads, seq_len]
    combined = combined.cpu().detach().numpy()  # [bz, seq_len]
    att_dis = torch.stack(att_dis).sum(0).squeeze(-1).cpu().detach().numpy()  # [bz, seq_len]

    return dan_prob, att_dis, combined, selected_heads_att



class H5SequenceDataset(Dataset):
    """Memory-mapped HDF5 reader."""
    def __init__(self, h5_path, data_idx_fp, batch_size=1024):
        """
            data_idx_fp: get protein accession(keys)
            h5_path: esm_h5-----------------store raw full_seq(its len may exclude 1022, but its embedding.shape <= 1022)
        """
        self.h5_path = h5_path
        with open(data_idx_fp, "r") as f:
            self.keys = [idx.strip() for idx in f]

        self.batch_size = batch_size

        self.h5_file = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)

        group = self.h5_file[self.keys[idx]]
        raw_all_sequence = group['sequence'].asstr()[()]
        all_sequence = raw_all_sequence[:1022]

        emb = torch.from_numpy(group['embedding'][:])    
   
        return [self.keys[idx], all_sequence, emb]
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
           
def collate_fn(batch):
    acc_list, seq_list, emb_list = zip(*batch)
    
    batch_emb = list(emb_list) + [torch.randn(1022, 1280)]
    batch_emb = pad_sequence(batch_emb, batch_first=True).float()[:-1,:,:] # (bz, 1022, 1280)
    return [acc_list, seq_list, batch_emb]

    
def optimized_reader(h5_path, data_idx_fp, batch_size, num_workers=4):
    """Parallel data loader."""
    dataset = H5SequenceDataset(h5_path, data_idx_fp, batch_size)
    print("load dataset")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

def process_large_h5(esm_h5_path, data_idx_fp, attn_h5_path, batch_size, device, model_path):
    """
        data_idx_fp: pdb_list
    """
    with h5py.File(attn_h5_path, 'w') as f:
        pass
    dataloader = optimized_reader(esm_h5_path, data_idx_fp, batch_size)
    print("load dataloader")

   
    device = torch.device(device)
    torch.cuda.set_device(device)
    model = LSTMTagger(1280, 1280).cuda()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("load model")


    for batch in tqdm(dataloader, desc="Processing"):
        batch_acc, batch_seq, batch_emb = batch
        batch_emb = batch_emb.cuda()
        with torch.no_grad():
            # att_dis, combined, selected_heads_att
            # shape: [bz, max_len], [bz, max_len], [bz, num_selected_heads, max_len]
            batch_dan_prob, batch_attn, batch_select_combined, batch_select_heads_att = get_distribu_with_select_batch(model, batch_emb)

        # write results in batches
        with h5py.File(attn_h5_path, 'a', libver='latest') as attn_file:
            for acc, seq, dan_prob, attn, combined, select_heads_attn in zip(
                batch_acc,
                batch_seq,
                batch_dan_prob,
                batch_attn,
                batch_select_combined,
                batch_select_heads_att
            ):
                seq_len = len(seq)
                assert seq_len <= 1022
                group = attn_file.require_group(acc)
                group.create_dataset('sequence', data=seq)
                group.create_dataset('dan_probability', data=float(dan_prob))

                group.create_dataset('attention', data=attn[:seq_len])
                group.create_dataset('select_attention', data=combined[:seq_len])
                # select_heads_attn： [num_selected_heads, seq_len]
                group.create_dataset('select_attention_matrix', data=select_heads_attn[:, :seq_len])  # matrix




if __name__ == "__main__":
    model_path = "../checkpoint/NLS_loc_modeltes"
    device = "cuda:0"
    batch_size=64

    esm_h5_fp = "/path/to/output/all_esm1b_dict.h5"
    data_idx_fp = "/path/to/output/all_acc.idx"
    attn_h5_fp = "/path/to/output/all_attn_dict.h5"
    process_large_h5(esm_h5_fp, data_idx_fp, attn_h5_fp, batch_size, device, model_path)
