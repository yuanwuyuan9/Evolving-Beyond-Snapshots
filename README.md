# EST: Temporal Knowledge Graph Forecasting

EST is a temporal knowledge graph (TKG) forecasting implementation for predicting future tails in queries of the form `(s, r, ?, t)`.

## What This Project Does

- Learns entity dynamics with persistent state memory instead of per-snapshot reset.
- Combines structural context and temporal sequence modeling in one pipeline.
- Supports multiple temporal backbones: `rnn`, `transformer`, `mamba`.
- Uses time-aware negative filtering to reduce false negatives during training.

## Framework (Figure 1)

- Original figure file: [`figure1.pdf`](./figure1.pdf)
- Clickable preview:

[![EST Framework](./figure1.png)](./figure1.pdf)

### Figure Explanation

The framework compares two paradigms:

- Stateless (left): representations are repeatedly reconstructed from local windows, which causes long-term information decay.
- Stateful EST (right): entity states are maintained globally and updated over time, connecting structural signals and temporal evolution through persistent memory.

Core flow in EST:

1. Retrieve historical neighbors and temporal deltas.
2. Fuse neighbor representations with persistent entity state.
3. Encode the history sequence with a pluggable temporal backbone.
4. Build query-specific context and score candidate tails.
5. Write context back to fast/slow entity memories.

## Repository Layout

```text
.
├── code/
│   ├── main.py                    # CLI entry
│   ├── data_loader.py             # dataset loader + temporal neighbor retrieval
│   ├── train.py                   # training/evaluation pipeline
│   └── models/
│       ├── tr_mamba.py            # EST core model
│       ├── temporal_encoders.py   # RNN / Transformer / Mamba sequence encoders
│       ├── struct_encoder.py      # structural encoder variants
│       └── time.py                # time-delta projection
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── ICEWS05-15/
│   └── GDELT/
└── requirements.txt
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run from repository root:

```bash
source .venv/bin/activate
python code/main.py \
  --data_dir ./data \
  --dataset ICEWS14 \
  --temporal_encoder transformer \
  --epoch 20 \
  --batch_size 128
```

CPU only:

```bash
python code/main.py --data_dir ./data --dataset ICEWS14 --no_cuda
```

Switch backbone:

```bash
python code/main.py --data_dir ./data --dataset ICEWS14 --temporal_encoder rnn
python code/main.py --data_dir ./data --dataset ICEWS14 --temporal_encoder mamba
```

Disable optional modules for ablation-like runs:

```bash
python code/main.py --data_dir ./data --dataset ICEWS14 --no_struct_encoder
python code/main.py --data_dir ./data --dataset ICEWS14 --no_time_aware_negative
```

## Recommended Training Presets

Balanced baseline:

```bash
python code/main.py \
  --data_dir ./data \
  --dataset ICEWS14 \
  --temporal_encoder transformer \
  --epoch 20 \
  --batch_size 128 \
  --dim 64 \
  --time_emb_dim 32 \
  --history_len 32 \
  --lr 5e-3 \
  --l2 1e-4
```

Lightweight run:

```bash
python code/main.py \
  --data_dir ./data \
  --dataset ICEWS14 \
  --temporal_encoder mamba \
  --epoch 10 \
  --batch_size 128 \
  --history_len 16
```

## Main Arguments

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `ICEWS14` | `ICEWS14`, `ICEWS18`, `ICEWS05-15`, `GDELT` |
| `--data_dir` | `../data` | Dataset root path (`./data` recommended from repo root) |
| `--epoch` | `30` | Number of epochs |
| `--batch_size` | `128` | Mini-batch size |
| `--seed` | `42` | Random seed |
| `--dim` | `64` | Entity/relation embedding dimension |
| `--time_emb_dim` | `128` | Time embedding dimension |
| `--history_len` | `32` | Historical window length |
| `--temporal_encoder` | `transformer` | `mamba`, `rnn`, `transformer` |
| `--use_context` / `--no_context` | on | Enable/disable temporal neighbor context |
| `--use_struct_encoder` / `--no_struct_encoder` | on | Enable/disable structural encoder |
| `--struct_type` | `linear` | `linear`, `rgcn`, `gat` |
| `--state_alpha` | `0.5` | Fast-state EMA update ratio |
| `--state_fuse` | `gate` | `add`, `gate` |
| `--lr` | `5e-3` | Learning rate |
| `--l2` | `1e-4` | Weight decay |
| `--lr_warmup_epochs` | `2` | Warmup epochs |
| `--lr_decay_factor` | `0.1` | Cosine annealing minimum LR ratio |
| `--time_encoding` | `delta` | `id`, `delta`, `id_plus_delta` |
| `--time_aware_negative` / `--no_time_aware_negative` | on | Enable/disable time-aware negative filtering |
| `--scoring_fn` | `distmult` | `distmult`, `mlp`, `complex`, `rotate` |

## Data Format

Each split file (`train.txt`, `valid.txt`, `test.txt`) is tab-separated:

```text
head    relation    tail    timestamp
```

Required files per dataset folder:

- `entity2id.txt`
- `relation2id.txt`
- `train.txt`
- `valid.txt`
- `test.txt`

## Training Logs and Metrics

Training output includes:

- Epoch learning rate and `train_loss`
- Evaluation metrics on train/valid/test:
  - accuracy
  - cross-entropy
  - `MRR`, `MR`, `Hits@1`, `Hits@3`, `Hits@10`
- Final best-test summary after restoring the best validation checkpoint

## Troubleshooting

- If CUDA is unavailable, add `--no_cuda`.
- If training is slow, reduce `--history_len` or switch to `--temporal_encoder mamba`.
- If memory is tight, reduce `--batch_size` first.
