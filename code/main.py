
import argparse
import os
from datetime import datetime

                                                            
try:
    from .data_loader import load_data                
    from .train import train                
except Exception:
    from data_loader import load_data
    from train import train

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def print_setting(args):
    print()
    print("=============================================")
    print("Temporal KG Forecasting Configuration")
    print("=============================================")
    print(f"temporal_encoder: {args.temporal_encoder}")
    print(f"dataset: {args.dataset}")
    print(f"data_dir: {args.data_dir}")
    print(f"epochs: {args.epoch}")
    print(f"batch_size: {args.batch_size}")
    print(f"dim: {args.dim}")
    print(f"time_emb_dim: {args.time_emb_dim}")
    print(f"history_len: {args.history_len}")
    print(f"use_struct_encoder: {args.use_struct_encoder}")
    print(f"struct_type: {args.struct_type}")
    print("use_state_writeback: True (always on)")
    print(f"state_alpha: {args.state_alpha}")
    print(f"state_fuse: {args.state_fuse}")
    print("model: tr_mamba")
    print(f"l2: {args.l2}")
    print(f"lr: {args.lr}")
    print(f"lr_warmup_epochs: {args.lr_warmup_epochs}")
    print(f"lr_decay_factor: {args.lr_decay_factor}")
    print(f"use_context: {args.use_context}")
    print(f"time_encoding: {args.time_encoding}")
    print(f"time_aware_negative: {args.time_aware_negative}")
    print(f"scoring_fn: {args.scoring_fn}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Temporal Knowledge Graph Tail Forecasting (TR-Mamba)")
    parser.set_defaults(
        cuda=True,
        use_context=True,
        use_struct_encoder=True,
        time_aware_negative=True,
    )

    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Enable CUDA if available")
    parser.add_argument("--no_cuda", dest="cuda", action="store_false", help="Force running on CPU")

    parser.add_argument("--use_context", dest="use_context", action="store_true", help="Enable temporal neighbor aggregation")
    parser.add_argument("--no_context", dest="use_context", action="store_false", help="Disable temporal neighbor aggregation")

    parser.add_argument("--dataset", type=str, default= 'ICEWS14', help="Dataset name (e.g., ICEWS14, ICEWS05-15, GDELT)")
    parser.add_argument("--data_dir", type=str, default="../data", help="Root directory containing temporal KG datasets")
    parser.add_argument("--epoch", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dim", type=int, default=64, help="Hidden dimension for entity/relation embeddings")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Embedding dimension for temporal encoding")
    parser.add_argument("--history_len", type=int, default=32, help="Number of past events considered for context")
    parser.add_argument(
            "--temporal_encoder",
            type=str,
            default="transformer",
            choices=["mamba", "rnn", "transformer"], 
            help="Backbone for temporal aggregation (for ablations).",
        )
    parser.add_argument(
        "--use_struct_encoder",
        dest="use_struct_encoder",
        action="store_true",
        help="Enable structural encoder on historical neighbors",
    )
    parser.add_argument(
        "--no_struct_encoder",
        dest="use_struct_encoder",
        action="store_false",
        help="Disable structural encoder on historical neighbors",
    )
    parser.add_argument(
        "--struct_type",
        type=str,
        default="linear",
        choices=["linear", "rgcn", "gat"],
        help="Type of structural encoder applied to neighbor embeddings",
    )
    parser.add_argument(
        "--state_alpha",
        type=float,
        default=0.5,
        help="EMA coefficient for updating entity state buffer during writeback",
    )
    parser.add_argument(
        "--state_fuse",
        type=str,
        default="gate",
        choices=["add", "gate"],
        help="How to fuse embedding and state when feeding structural encoder",
    )
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularisation weight")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=2,
        help="Number of warmup epochs for learning rate (linear schedule)",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.1,
        help="Minimum LR ratio for cosine annealing (eta_min = lr * lr_decay_factor)",
    )
    parser.add_argument(
        "--time_encoding",
        type=str,
        default="delta",
        choices=["id", "delta", "id_plus_delta"],
        help="Type of time encoding used in the query representation",
    )
    parser.add_argument(
        "--time_aware_negative",
        dest="time_aware_negative",
        action="store_true",
        help="Enable history-aware (time-consistent) filtering of false negatives during training",
    )
    parser.add_argument(
        "--no_time_aware_negative",
        dest="time_aware_negative",
        action="store_false",
        help="Disable history-aware (time-consistent) filtering of false negatives during training",
    )
    parser.add_argument(
        "--scoring_fn",
        type=str,
        default="distmult",
        choices=["distmult", "mlp", "complex", "rotate"],
        help="Tail scoring function: distmult (default), mlp, complex, or rotate",
    )
    args = parser.parse_args()

    print("Using CUDA" if args.cuda else "Using CPU")

    print_setting(args)
               
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_data(args)
    print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train(args, data)


if __name__ == "__main__":
    main()
