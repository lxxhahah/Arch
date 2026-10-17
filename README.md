# NLS-type segment discovery in archaeal proteins and downstream analysis

A pipeline for discovering candidate nuclear localization signal (NLS)-type
segments in archaeal proteins from sequence alone, ranking and filtering
them with a dual-track criterion, and running the downstream analyses (latent-space
visualization, GO / ribosomal enrichment).

The pipeline has two stages:

1. **Inference** (ESM embedding → DAN attention → segment generation → segment ranking)
2. **Analysis** (collect to TSV → dual-track filter → visualization / GO analysis)

---

## Environment

Create a Python 3.9 environment and install the dependencies:

```bash
conda create -n nls python=3.9 -y
conda activate nls
pip install -r requirements.txt
```

`torch` is pinned to a CUDA 12.1 build — install from https://pytorch.org if your
CUDA toolkit differs. `mamba-ssm` is optional (see `requirements.txt`).

---

## Model checkpoints (download first)

The model weights are **not** shipped with the code (the `checkpoint/` directory is a
placeholder). Download them from <https://huggingface.co/lxxyyds/Arch_NLS_type> and
place them before running steps 2 and 4:

| File | Used by | Purpose | Download |
|------|---------|---------|----------|
| `NLS_loc_modeltes` | step 2 | DAN model weights | [checkpoint/NLS_loc_modeltes](https://huggingface.co/lxxyyds/Arch_NLS_type/resolve/main/checkpoint/NLS_loc_modeltes) |
| `per80_select_li.pkl` | step 2 | BAU (basic attention unit) selection template | [checkpoint/per80_select_li.pkl](https://huggingface.co/lxxyyds/Arch_NLS_type/resolve/main/checkpoint/per80_select_li.pkl) |
| `RENOZDM10` | step 4 | recommendation network weights | [checkpoint/RENOZDM10](https://huggingface.co/lxxyyds/Arch_NLS_type/resolve/main/checkpoint/RENOZDM10) |

The scripts reference them via a `../checkpoint/` relative path; place them accordingly
or adjust the path in each script.

---

## Pipeline overview

```
proteins.fasta
      │  (1) ESM embedding extraction
      ▼
all_esm1b_dict.h5  +  all_acc.idx
      │
      ├──(2) DAN attention extraction──► all_attn_dict.h5
      │                                        │
      │                                 (3) KSG segment generation
      │                                        ▼
      │                            all_generate_segs_dict.h5
      │                                        │
      └──────────(4) segment recommendation────┘
                                               ▼
                                    all_result_dict.h5
                                               │
           (5) collect recommendation probability + compute attention consensus ratio
                                               ▼
                                      all_result.tsv
                                               │
                                    (6) dual-track filter
                                               ▼
                                  candidate_result.tsv
                                  high_conf_result.tsv
                                        │
              (7) latent space visualization & GO enrichment analysis
```

---

## Quick reference

| # | Script | Input | Output |
|---|--------|-------|--------|
| 1 | `preprocess/esm_embedding_extract.py` | `proteins.fasta` | `all_esm1b_dict.h5`, `all_acc.idx` |
| 2 | `scripts/NLSExplorer_inference/A2KA/predict_with_DAN.py` | `all_esm1b_dict.h5`, `all_acc.idx`, `checkpoint/NLS_loc_modeltes`, `checkpoint/per80_select_li.pkl` | `all_attn_dict.h5` |
| 3 | `scripts/NLSExplorer_inference/A2KA/KSG.py` | `all_attn_dict.h5`, `all_acc.idx` | `all_generate_segs_dict.h5` |
| 4 | `scripts/NLSExplorer_inference/Recommendation_system/predict_with_recommend.py` | `all_esm1b_dict.h5`, `all_generate_segs_dict.h5`, `all_acc.idx`, `checkpoint/RENOZDM10` | `all_result_dict.h5` |
| 5 | `preprocess/collect_p_and_attn_cons.py` | `all_result_dict.h5`, `all_esm1b_dict.h5`, `all_attn_dict.h5`, `all_acc.idx` (optional) | `all_result.tsv` |
| 6 | `scripts/analysis/dual_track_filter.py` | `all_result.tsv` | `candidate_result.tsv`, `high_conf_result.tsv` |
| 7a | `scripts/analysis/latent_space.py` | `high_conf_result.tsv`, h5 files, `experiment_labels.tsv`, external annotation files | `high_conf_with_tsne.tsv`, t-SNE PNGs |
| 7b | `scripts/analysis/go_enrichment_analysis.py` | `high_conf_result.tsv`, `idmapping.tsv` | `ribosomal_go_model1.tsv`, `ribosomal_go_model2.tsv` |

---

## Step-by-step details

### Step 1 — ESM1b embedding extraction

Embed each protein with ESM1b (layer 33, 1280-dim, truncated to 1022 residues).

**Input**

- `proteins.fasta` — protein FASTA.

**Output**

- `all_esm1b_dict.h5` — one group per accession:
  - `embedding` — `[L, 1280]` float32, `L = min(1022, len(sequence))`
  - `sequence` — the residue string
- `all_acc.idx` — one accession per line (used as the key list by every later step)

**Run**

```bash
python preprocess/esm_embedding_extract.py
```

### Step 2 — DAN attention extraction

Run the DAN model over the ESM embeddings to get per-residue attention and the
BAU (basic attention unit) selection matrix.

**Input**

- `all_esm1b_dict.h5`, `all_acc.idx`
- `checkpoint/NLS_loc_modeltes` — DAN weights
- `checkpoint/per80_select_li.pkl` — BAU (basic attention unit) selection template `[16 layers, 64 BAUs]` (0/1)

**Output**

- `all_attn_dict.h5` — one group per accession:
  - `sequence` — residue string (≤ 1022)
  - `dan_probability` — scalar
  - `attention` — `[seq_len]`
  - `select_attention` — `[seq_len]` (selected BAUs, summed)
  - `select_attention_matrix` — `[num_selected_BAUs, seq_len]`

**Run**

```bash
python scripts/NLSExplorer_inference/A2KA/predict_with_DAN.py
```

### Step 3 — Key Segment Generation (KSG)

Generate candidate NLS segment intervals from the attention signal.

**Input**

- `all_attn_dict.h5` (`select_attention` per accession)
- `all_acc.idx`
- Parameters: `cofactor`, `set_len`, `f_c`, `stret_choose`

**Output**

- `all_generate_segs_dict.h5` — one group per accession:
  - `segments` — `[n_segments, 2]` int32, each row is a `[start, end]` interval

**Run**

```bash
python scripts/NLSExplorer_inference/A2KA/KSG.py
```

### Step 4 — Segment recommendation / ranking

Score every generated segment with the recommendation system and rank them.

**Input**

- `all_esm1b_dict.h5` (embeddings + sequences)
- `all_generate_segs_dict.h5` (candidate intervals)
- `all_acc.idx`
- `checkpoint/RENOZDM10` — recommendation network weights

**Output**

- `all_result_dict.h5` — `acc → seg → ...`, where each `seg` group is keyed by the
  **raw** interval string `"{start}_{end}"` and contains:
  - `probability` — recommendation score
  - `embedding` — recommendation embedding (640-dim; used by the latent-space analysis)
  - `segment` — `[start, end]`
  - `sequence` — `seq[start:end+1]`

**Run**

```bash
python scripts/NLSExplorer_inference/Recommendation_system/predict_with_recommend.py
```

### Step 5 — Collect to TSV + consensus ratio

Flatten the recommendation HDF5 into one TSV row per segment, and add the
attention consensus ratio. This is the hand-off between inference and analysis.

**Input**

- `all_result_dict.h5` (recommendation output)
- `all_esm1b_dict.h5` (for full sequences)
- `all_attn_dict.h5` (for `select_attention_matrix` → attention consensus ratio)
- `all_acc.idx` (optional; defaults to the keys of `all_result_dict.h5`)

**Output**

- `all_result.tsv` — columns:

  | column | meaning |
  |--------|---------|
  | `acc` | accession |
  | `seg` | **raw** interval key `"{start}_{end}"` as stored in the HDF5 (may be out-of-bounds) |
  | `valid_seg` | the same interval with `end` clipped to `len(sequence)-1` |
  | `prob` | recommendation probability |
  | `consensus_ratio` | mean fraction of BAUs above a per-BAU threshold over the segment (0–1) |


**Run**

```bash
python preprocess/collect_p_and_attn_cons.py
```

### Step 6 — Dual-track filter

Keep high-confidence segments by a dual-track rule: **extremely high probability**
(`prob >= 0.95`) **OR** moderate probability with an above-average consensus ratio
(`consensus_ratio >= pool mean`). A first pass keeps `prob >= 0.7` as the candidate pool.

**Input**

- `all_result.tsv`

**Output**

- `candidate_result.tsv` — rows with `prob >= 0.7` (adds a `seg_key` column)
- `high_conf_result.tsv` — the dual-track subset (same columns)

Both add the join key:

```
seg_key = acc + "_" + valid_seg
```

**Run**

```bash
python scripts/analysis/dual_track_filter.py
```

### Step 7a — Latent-space visualization

Load segment embeddings (recommendation + ESM), project with PCA + t-SNE, and draw
scatter plots colored by cluster / HMM alignment probability / species, with
experimentally validated positives highlighted.

**Input**

- `high_conf_result.tsv`
- `all_result_dict.h5` (segment embeddings, path in `config.yaml` → `h5`)
- `all_esm1b_dict.h5` (ESM embeddings, path in `config.yaml` → `esm_h5`)
- `experiment_labels.tsv` (validated positives, keyed by `seg_key`)
- External annotation files (paths under `data_save_path` in `config.yaml`):
  - `fasta_dir/high_conf_seg_seq.fasta` + its cluster result
    `clusterRes_cluster.tsv` (MMseqs2 format)
  - `hhm_dir/align_best_prob.tsv` (HMM-HMM best alignment probability)
  - `species_information/idmapping.tsv` (UniProt `Entry` / `Organism (ID)` table)

**Output**

- t-SNE PNGs:
  - `<rec|esm>_color_by_cluster.png`
  - `<rec|esm>_color_by_hmm_prob.png`
  - `<rec|esm>_color_by_species.png`

**Run**

```bash
python scripts/analysis/latent_space.py
```

### Step 7b — GO / ribosomal analysis

Logistic-regression models testing whether selected proteins are enriched for the
ribosomal GO term, controlling for K/R density and length.

**Input**

- `high_conf_result.tsv` (target set)
- `species_information/idmapping.tsv` (background table with `Entry`, `Sequence`,
  and `Gene Ontology (molecular function)` columns)

**Output**

- `go_enrichment.svg` — GO molecular-function enrichment bar chart (top 10 terms)
- `ribosomal_go_model1.tsv` — `selected ~ ribosomal_go + kr_fraction + log_length`
- `ribosomal_go_model2.tsv` — `selected ~ ribosomal_go + kr_fraction + max_kr15 + log_length`



**Run**

```bash
python scripts/analysis/go_enrichment_analysis.py
```

---

## Configuration

Steps 6 and 7 read all paths from [`scripts/analysis/config.yaml`](scripts/analysis/config.yaml):

- `esm_h5` — path to `all_esm1b_dict.h5` per dataset
- `h5` — path to `all_result_dict.h5` per dataset
- `data_save_path` — directory holding `all_result.tsv` / `candidate_result.tsv` /
  `high_conf_result.tsv` and the external annotation folders
- `picture_save_path` — directory for figures and analysis TSVs

Datasets are keyed by name (`Ocean`, `Asgard` by default); add or remove keys to match
your own data.

## Conventions

- **Coordinates are 0-based and inclusive** — a segment `[start, end]` covers
  `seq[start:end+1]`.
- Sequence length is truncated to **1022** residues everywhere downstream of ESM.
