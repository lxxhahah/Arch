import os
import h5py
from datetime import datetime  
from multiprocessing import get_context
import subprocess
import glob
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import numpy as np
from esm import FastaBatchedDataset, pretrained




class ESM:
    def __init__(self, fasta_file, tmp_dir, model_location, device, n_process=1, toks_per_batch=4096):
        """
            tmp_dir: store tmp fasta file -> to multiprocess
            model_location: default-esm1b_t33_650M_UR50S  or (esm2_t33_650M_UR50D)
        """
        self.fasta = fasta_file
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        self.model_location = model_location
        self.n = n_process
        self.toks_per_batch = toks_per_batch
        self.device = device if isinstance(device, str) else str(device)
    

    def split_fasta_file(self):
        """Split the FASTA into chunks for multiprocessing and return the actual chunk files."""
        # Remove stale split files so glob does not pick them up
        for stale in glob.glob(os.path.join(self.tmp_dir, "fasta_*")):
            os.remove(stale)

        output = subprocess.check_output(["wc", "-l", self.fasta])
        lines = int(output.decode().split()[0])

        # Round lines-per-file up to an even number so a FASTA record is never split in half
        lines_per_file = (lines // (self.n * 2) + 1) * 2
        subprocess.run(
            ["split", "-l", str(lines_per_file), self.fasta, "-d", "-a", "3",
             f"{self.tmp_dir}/fasta_"],
            check=True,
        )

        split_files = sorted(glob.glob(os.path.join(self.tmp_dir, "fasta_*")))
        if not split_files:
            raise RuntimeError("split produced no output files")
        return split_files

    def run_esm(self, fasta_file, h5_fp = None):
        self.device = torch.device(self.device)
        model, alphabet = pretrained.load_model_and_alphabet(self.model_location)
        model.eval()
        model.to(self.device)

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(self.toks_per_batch, extra_toks_per_seq=1)
        data_loader = DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(1022), 
            batch_sampler=batches, pin_memory=True
        )

        hf = None
        if h5_fp is not None:
            hf = h5py.File(h5_fp, 'a')  # append mode
            hf.attrs["model_name"] = self.model_location
            hf.attrs["processing_date"] = str(datetime.now())

        result_dict = {}
        # key: accession, ele is also a dict ["embedding"] ["sequence"]
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                toks = toks.to(device=self.device, non_blocking=True)
                # Get model output from layer 33
                out = model(toks, repr_layers=[33], return_contacts=False)
                # for a batch representation  key: "33" value: [batch_size, max_len, 1280]
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }
                for i, label in enumerate(labels):
                    truncate_len = min(1022, len(strs[i]))
                    # result is for a single sequence representation
                    result = {
                        layer: t[i, 1: truncate_len + 1].clone() # exclude: mean(0)- residual-level embedding
                        for layer, t in representations.items()
                    }


                    embedding = np.array(result[33])
                    sequence = strs[i]

                    if hf is not None:
                        grp = hf.create_group(label)
                        grp.create_dataset("embedding", data=embedding.astype(np.float32))
                        grp.create_dataset("sequence", data=np.string_(sequence))
                    else:
                        result_dict[label] = {
                            "embedding": embedding,
                            "sequence": sequence
                        }

        if hf is not None:
            hf.close()
            return None
        return result_dict

    def run_esm_multiproc(self, save_h5_fp):
        if self.n == 1:
            self.run_esm(self.fasta, save_h5_fp)
            return

        split_files = self.split_fasta_file()

        # Remove stale per-part outputs to avoid duplicate keys in append mode
        for stale in glob.glob(os.path.join(self.tmp_dir, "part_*.h5")):
            os.remove(stale)

        pool_runesm = get_context("spawn").Pool(self.n)
        tmp_h5s = []
        results = []
        for i, fasta_file in enumerate(split_files):
            tmp_h5 = os.path.join(self.tmp_dir, f"part_{i}.h5")
            tmp_h5s.append(tmp_h5)
            results.append(pool_runesm.apply_async(self.run_esm, (fasta_file, tmp_h5)))

        pool_runesm.close()
        # Collect results via get() so any worker exception surfaces here instead of being silently swallowed
        for r in results:
            r.get()
        pool_runesm.join()

        with h5py.File(save_h5_fp, 'w') as hf:
            hf.attrs["model_name"] = self.model_location
            hf.attrs["processing_date"] = str(datetime.now())
            for tmp_h5 in tmp_h5s:
                with h5py.File(tmp_h5, 'r') as part:
                    for accession in part:
                        part.copy(accession, hf)
                os.remove(tmp_h5)


        
def generate_accession_index(fasta_fp, idx_fp):
    """Write one accession per line to idx_fp, parsed from a FASTA file.
    Returns the list of accessions.
    """
    accessions = []
    with open(fasta_fp) as f:
        for line in f:
            if line.startswith(">"):
                accessions.append(line[1:].strip().split()[0])
    with open(idx_fp, "w") as f:
        f.write("\n".join(accessions) + "\n")
    return accessions


if __name__ == "__main__":
    model_location = "esm1b_t33_650M_UR50S"
    device = "cuda:0"
    # set n_process > 1 to enable multiprocessing (memory scales linearly with n)
    n_process = 8
    fasta_fp = "/path/to/proteins.fasta"
    tmp_dir = "/path/to/tmp_dir"
    save_h5_fp = "/path/to/output/all_esm1b_dict.h5"
    idx_fp = "/path/to/output/all_acc.idx"
    # Generate one-accession-per-line index from the input FASTA
    generate_accession_index(fasta_fp, idx_fp)

    esm_runner = ESM(fasta_fp, tmp_dir, model_location, device, n_process=n_process)  
    print("start to run esm")
    esm_runner.run_esm_multiproc(save_h5_fp)

