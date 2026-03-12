<div align="center">

# Evolving Beyond Snapshots

### Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2602.12389-b31b1b)](https://arxiv.org/abs/2602.12389)
![Task](https://img.shields.io/badge/Task-TKG%20Forecasting-1f6feb)
![Backbone](https://img.shields.io/badge/Backbones-Transformer%20%7C%20Mamba%20%7C%20LSTM%20%7C%20RNN-0a7f5a)

Official implementation of the paper *Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting*.

</div>

## Overview

EST is a temporal knowledge graph forecasting framework for tail prediction on queries of the form `(s, r, ?, t)`. It unifies structural context modeling, temporal sequence encoding, and persistent entity-state updates in a single training pipeline.

Highlights:

- Persistent entity-state memory beyond snapshot-wise reconstruction.
- Pluggable temporal backbones: `transformer`, `mamba`, `lstm`, `rnn`.
- Optional structural encoder for historical neighbor modeling.
- Past-only negative filtering during training.

## Framework

<p align="center">
  <img src="./framework.png" alt="EST Framework" width="88%">
</p>

<p align="center"><em>EST maintains and updates entity states over time to couple structural evidence with temporal dynamics.</em></p>

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training from the repository root:

```bash
python code/main.py \
  --data_dir ./data \
  --dataset ICEWS14 \
  --temporal_encoder transformer \
  --epoch 20 \
  --batch_size 128
```

Recommended baseline:

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

Common variants:

```bash
python code/main.py --data_dir ./data --dataset ICEWS14 --no_cuda
python code/main.py --data_dir ./data --dataset ICEWS14 --temporal_encoder lstm
python code/main.py --data_dir ./data --dataset ICEWS14 --temporal_encoder mamba
python code/main.py --data_dir ./data --dataset ICEWS14 --no_struct_encoder
python code/main.py --data_dir ./data --dataset ICEWS14 --no_time_aware_negative
```

## Repository Structure

```text
.
├── code/
│   ├── main.py                  # training entry point and CLI configuration
│   ├── data_loader.py           # dataset loading and temporal neighbor retrieval
│   ├── train.py                 # training and evaluation loop
│   └── models/
│       ├── est.py               # EST model definition
│       ├── temporal_encoders.py # Transformer / Mamba / LSTM / RNN backbones
│       ├── struct_encoder.py    # structural encoder variants
│       └── time.py              # time encoding utilities
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── ICEWS05-15/
│   └── GDELT/
└── requirements.txt
```

## Citation

```bibtex
@article{li2026evolving,
  title={Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting},
  author={Li, Siyuan and Wu, Yunjia and Xiao, Yiyong and Huang, Pingyang and Li, Peize and Liu, Ruitong and Wen, Yan and Sun, Te and Pei, Fangyi},
  journal={arXiv preprint arXiv:2602.12389},
  year={2026}
}
```
