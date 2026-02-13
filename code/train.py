from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from .models.tr_mamba import TRMamba
except Exception:
    from models.tr_mamba import TRMamba


SplitDict = Dict[str, np.ndarray]
RankMetrics = Tuple[float, float, float, float, float]
TrueTailMap = Dict[Tuple[int, int], np.ndarray]


def prepare_split(quads: List[tuple]) -> SplitDict:
    if not quads:
        arr = np.zeros((0, 4), dtype=np.int64)
    else:
        arr = np.asarray(quads, dtype=np.int64)
    return {
        "heads": arr[:, 0] if arr.size else np.zeros(0, dtype=np.int64),
        "tails": arr[:, 1] if arr.size else np.zeros(0, dtype=np.int64),
        "relations": arr[:, 2] if arr.size else np.zeros(0, dtype=np.int64),
        "times": arr[:, 3] if arr.size else np.zeros(0, dtype=np.int64),
        "quad_array": arr,
        "size": arr.shape[0],
    }


def build_true_tails_by_hr(history_quads_sorted: List[tuple]) -> TrueTailMap:
    observed: Dict[Tuple[int, int], set] = defaultdict(set)
    for h, t, r, _ in history_quads_sorted:
        observed[(int(h), int(r))].add(int(t))
    return {
        key: np.fromiter(vals, dtype=np.int64) if vals else np.zeros(0, dtype=np.int64)
        for key, vals in observed.items()
    }


def build_feed_dict(
    split: SplitDict,
    indices: np.ndarray,
    device: torch.device,
    use_context: bool,
    history_len: int,
    neighbor_finder,
) -> Dict[str, torch.Tensor]:
    batch_heads = split["heads"][indices]
    batch_rels = split["relations"][indices]
    batch_times = split["times"][indices]
    batch_labels = split["tails"][indices]

    feed = {
        "heads": torch.from_numpy(batch_heads).long().to(device),
        "relations": torch.from_numpy(batch_rels).long().to(device),
        "times": torch.from_numpy(batch_times).long().to(device),
        "labels": torch.from_numpy(batch_labels).long().to(device),
    }

    if use_context and neighbor_finder is not None and batch_heads.size > 0:
        hist_entities_h, hist_relations_h, hist_deltas_h, hist_mask_h = neighbor_finder.get_temporal_neighbors(
            batch_heads, batch_times, history_len
        )
        feed["history_entities"] = torch.from_numpy(hist_entities_h).long().to(device)
        feed["history_relations"] = torch.from_numpy(hist_relations_h).long().to(device)
        feed["history_deltas"] = torch.from_numpy(hist_deltas_h).float().to(device)
        feed["history_mask"] = torch.from_numpy(hist_mask_h).float().to(device)

    return feed


def train_step_sampled(
    feed: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    model: TRMamba,
    step: int,
    use_time_aware_negative: bool,
    true_tails_by_hr_train: Optional[TrueTailMap],
) -> float:
    model.train()
    optimizer.zero_grad()

    labels_true = feed["labels"]
    batch_size = int(labels_true.size(0))
    if batch_size == 0:
        return 0.0

    scores_full = model.forward_full(feed, step)["scores"]

    if use_time_aware_negative and true_tails_by_hr_train:
        heads_np = feed["heads"].detach().cpu().numpy()
        rels_np = feed["relations"].detach().cpu().numpy()
        labels_np = labels_true.detach().cpu().numpy()
        for i in range(batch_size):
            key = (int(heads_np[i]), int(rels_np[i]))
            tails_arr = true_tails_by_hr_train.get(key)
            if tails_arr is None or tails_arr.size == 0:
                continue
            for tail_id in tails_arr:
                if int(tail_id) != int(labels_np[i]):
                    scores_full[i, int(tail_id)] = -1e9

    target = labels_true.long().to(model.device)
    loss = torch.nn.functional.cross_entropy(scores_full, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate_plausibility(
    split: SplitDict,
    model: TRMamba,
    device: torch.device,
    history_quads_sorted: List[tuple],
    use_context: bool,
    history_len: int,
    neighbor_finder,
) -> Tuple[float, float, RankMetrics]:
    total_samples = split["size"]
    if total_samples == 0:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0)

    total_correct = 0
    total_loss = 0.0
    all_ranks: List[float] = []

    order = np.argsort(split["times"])
    times_all = split["times"][order]

    observed = defaultdict(set)
    hist_ptr = 0
    total_hist = len(history_quads_sorted)

    start = 0
    while start < total_samples:
        current_time = times_all[start]

        while hist_ptr < total_hist and history_quads_sorted[hist_ptr][3] < current_time:
            h, t, r, _ = history_quads_sorted[hist_ptr]
            observed[(h, r)].add(t)
            hist_ptr += 1

        end = start
        while end < total_samples and times_all[end] == current_time:
            end += 1
        batch_indices = order[start:end]

        feed = build_feed_dict(
            split=split,
            indices=batch_indices,
            device=device,
            use_context=use_context,
            history_len=history_len,
            neighbor_finder=neighbor_finder,
        )

        labels_true_t = feed["labels"]
        labels_true = labels_true_t.detach().cpu().numpy()
        batch_size = labels_true.shape[0]
        if batch_size == 0:
            start = end
            continue

        heads_np = feed["heads"].detach().cpu().numpy()
        rels_np = feed["relations"].detach().cpu().numpy()

        with torch.no_grad():
            scores_full = model.forward_full(feed, 0)["scores"]

        target = labels_true_t.long().to(model.device)
        loss = torch.nn.functional.cross_entropy(scores_full, target)
        preds = scores_full.argmax(dim=1)
        correct = (preds == target).sum().item()
        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct)

        scores_np = scores_full.detach().cpu().numpy()
        for i in range(batch_size):
            pos = int(labels_true[i])
            pos_score = scores_np[i, pos]
            filt = observed.get((int(heads_np[i]), int(rels_np[i])), set())
            if filt:
                mask = np.ones(model.n_entity, dtype=bool)
                for ent_id in filt:
                    if ent_id != pos:
                        mask[int(ent_id)] = False
            else:
                mask = np.ones(model.n_entity, dtype=bool)

            greater = (scores_np[i] > pos_score) & mask
            rank = 1.0 + float(greater.sum())
            all_ranks.append(rank)

        while hist_ptr < total_hist and history_quads_sorted[hist_ptr][3] == current_time:
            h, t, r, _ = history_quads_sorted[hist_ptr]
            observed[(h, r)].add(t)
            hist_ptr += 1

        start = end

    avg_acc = total_correct / total_samples if total_samples else 0.0
    avg_loss = total_loss / total_samples if total_samples else 0.0

    metrics: RankMetrics = (0.0, 0.0, 0.0, 0.0, 0.0)
    if all_ranks:
        ranks = np.asarray(all_ranks, dtype=float)
        mrr = float(np.mean(1.0 / ranks))
        mr = float(np.mean(ranks))
        hit1 = float(np.mean(ranks <= 1))
        hit3 = float(np.mean(ranks <= 3))
        hit10 = float(np.mean(ranks <= 10))
        metrics = (mrr, mr, hit1, hit3, hit10)

    return avg_acc, avg_loss, metrics


def train(model_args, data_bundle):
    triplets, n_relations, neighbor_finder, n_entity, time_metadata = data_bundle
    train_quads, valid_quads, test_quads = triplets

    train_split = prepare_split(train_quads)
    valid_split = prepare_split(valid_quads)
    test_split = prepare_split(test_quads)

    model = TRMamba(model_args, n_relations, n_entity, time_metadata)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model_args.lr,
        weight_decay=model_args.l2,
    )

    base_lr = float(getattr(model_args, "lr", 1e-3))
    warmup_epochs = int(getattr(model_args, "lr_warmup_epochs", 0))
    min_lr_ratio = float(getattr(model_args, "lr_decay_factor", 0.1))
    use_context = bool(getattr(model_args, "use_context", True))
    history_len = int(getattr(model_args, "history_len", 32))
    use_time_aware_negative = bool(getattr(model_args, "time_aware_negative", False))

    t_max = max(1, int(model_args.epoch) - max(0, warmup_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=base_lr * min_lr_ratio,
    )

    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    history_train = sorted(train_quads, key=lambda x: x[3])
    history_valid = sorted(train_quads + valid_quads, key=lambda x: x[3])
    history_test = sorted(train_quads + valid_quads + test_quads, key=lambda x: x[3])

    true_tails_by_hr_train: Optional[TrueTailMap] = None
    if use_time_aware_negative:
        true_tails_by_hr_train = build_true_tails_by_hr(history_train)

    best_valid_acc = 0.0
    best_state_dict = None
    final_res = None

    print("start training ...")
    train_size = train_split["size"]

    for step in range(model_args.epoch):
        if warmup_epochs > 0 and step < warmup_epochs:
            warmup_lr = base_lr * float(step + 1) / float(warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            scheduler.step()

        permutation = np.random.permutation(train_size)
        epoch_losses = []

        for start in range(0, train_size, model_args.batch_size):
            batch_indices = permutation[start:start + model_args.batch_size]
            if batch_indices.size == 0:
                continue
            feed = build_feed_dict(
                split=train_split,
                indices=batch_indices,
                device=model.device,
                use_context=use_context,
                history_len=history_len,
                neighbor_finder=neighbor_finder,
            )
            loss = train_step_sampled(
                feed=feed,
                optimizer=optimizer,
                model=model,
                step=step,
                use_time_aware_negative=use_time_aware_negative,
                true_tails_by_hr_train=true_tails_by_hr_train,
            )
            epoch_losses.append(loss)

        model.eval()
        with torch.no_grad():
            train_acc, train_loss_eval, train_rank = evaluate_plausibility(
                split=train_split,
                model=model,
                device=model.device,
                history_quads_sorted=history_train,
                use_context=use_context,
                history_len=history_len,
                neighbor_finder=neighbor_finder,
            )
            valid_acc, valid_loss_eval, valid_rank = evaluate_plausibility(
                split=valid_split,
                model=model,
                device=model.device,
                history_quads_sorted=history_valid,
                use_context=use_context,
                history_len=history_len,
                neighbor_finder=neighbor_finder,
            )
            test_acc, test_loss_eval, test_rank = evaluate_plausibility(
                split=test_split,
                model=model,
                device=model.device,
                history_quads_sorted=history_test,
                use_context=use_context,
                history_len=history_len,
                neighbor_finder=neighbor_finder,
            )

        current_lr = optimizer.param_groups[0].get("lr", base_lr)
        if epoch_losses:
            print(f"epoch {step:2d}   lr: {current_lr:.6f}   train_loss: {np.mean(epoch_losses):.4f}")
        else:
            print(f"epoch {step:2d}   lr: {current_lr:.6f}   train_loss: N/A")

        print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        tmrr, tmr, th1, th3, th10 = train_rank
        vmrr, vmr, vh1, vh3, vh10 = valid_rank
        smrr, smr, sh1, sh3, sh10 = test_rank
        print(
            f"[full]    train acc: {train_acc:.4f}  ce: {train_loss_eval:.4f}  "
            f"mrr: {tmrr:.4f}  mr: {tmr:.1f}  h1: {th1:.4f}  h3: {th3:.4f}  h10: {th10:.4f}"
        )
        print(
            f"[full]    valid acc: {valid_acc:.4f}  ce: {valid_loss_eval:.4f}  "
            f"mrr: {vmrr:.4f}  mr: {vmr:.1f}  h1: {vh1:.4f}  h3: {vh3:.4f}  h10: {vh10:.4f}"
        )
        print(
            f"[full]    test  acc: {test_acc:.4f}  ce: {test_loss_eval:.4f}  "
            f"mrr: {smrr:.4f}  mr: {smr:.1f}  h1: {sh1:.4f}  h3: {sh3:.4f}  h10: {sh10:.4f}"
        )

        current_res = (
            f"acc: {test_acc:.4f}   ce: {test_loss_eval:.4f}   mrr: {smrr:.4f}   "
            f"mr: {smr:.1f}   h1: {sh1:.4f}   h3: {sh3:.4f}   h10: {sh10:.4f}"
        )
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            final_res = current_res

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        test_acc, test_loss_eval, test_rank = evaluate_plausibility(
            split=test_split,
            model=model,
            device=model.device,
            history_quads_sorted=history_test,
            use_context=use_context,
            history_len=history_len,
            neighbor_finder=neighbor_finder,
        )
    smrr, smr, sh1, sh3, sh10 = test_rank
    print(
        f"[full-best] test  acc: {test_acc:.4f}  ce: {test_loss_eval:.4f}  "
        f"mrr: {smrr:.4f}  mr: {smr:.1f}  h1: {sh1:.4f}  h3: {sh3:.4f}  h10: {sh10:.4f}"
    )

    summary = final_res if final_res else "No improvement recorded."
    print(f"final results\n{summary}")
