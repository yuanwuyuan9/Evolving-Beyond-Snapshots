

import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


entity_dict: Dict[str, int] = {}
relation_dict: Dict[str, int] = {}


class TemporalNeighborFinder:


    def __init__(
        self,
        head_histories: Dict[int, Dict[str, np.ndarray]],
        tail_histories: Dict[int, Dict[str, np.ndarray]],
        time_id_to_value: Sequence[float],
    ):
        self._head_histories = head_histories
        self._tail_histories = tail_histories
        self._time_id_to_value = np.asarray(time_id_to_value, dtype=np.float64)

    def get_temporal_neighbors(
        self,
        head_ids: np.ndarray,
        time_ids: np.ndarray,
        history_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        batch_size = len(head_ids)
        neighbors = np.zeros((batch_size, history_len), dtype=np.int64)
        relations = np.zeros((batch_size, history_len), dtype=np.int64)
        time_deltas = np.zeros((batch_size, history_len), dtype=np.float32)
        mask = np.zeros((batch_size, history_len), dtype=np.float32)

        for i, (head, time_id) in enumerate(zip(head_ids, time_ids)):
            history = self._head_histories.get(int(head))
            if history is None or history["times"].size == 0:
                continue

            cutoff = np.searchsorted(history["times"], time_id, side="left")
            if cutoff == 0:
                continue

            slice_start = max(0, cutoff - history_len)
            times_slice = history["times"][slice_start:cutoff]
            rels_slice = history["relations"][slice_start:cutoff]
            tails_slice = history["tails"][slice_start:cutoff]

            length = len(times_slice)
            neighbors[i, -length:] = tails_slice
            relations[i, -length:] = rels_slice
            mask[i, -length:] = 1.0

            query_time_val = self._time_id_to_value[int(time_id)]
            past_time_vals = self._time_id_to_value[times_slice]
            deltas = np.maximum(query_time_val - past_time_vals, 0.0)
            time_deltas[i, -length:] = deltas.astype(np.float32)

        return neighbors, relations, time_deltas, mask

    def get_tail_temporal_neighbors(
        self,
        tail_ids: np.ndarray,
        time_ids: np.ndarray,
        history_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        batch_size = len(tail_ids)
        neighbors = np.zeros((batch_size, history_len), dtype=np.int64)
        relations = np.zeros((batch_size, history_len), dtype=np.int64)
        time_deltas = np.zeros((batch_size, history_len), dtype=np.float32)
        mask = np.zeros((batch_size, history_len), dtype=np.float32)

        for i, (tail, time_id) in enumerate(zip(tail_ids, time_ids)):
            history = self._tail_histories.get(int(tail))
            if history is None or history["times"].size == 0:
                continue

            cutoff = np.searchsorted(history["times"], time_id, side="left")
            if cutoff == 0:
                continue

            slice_start = max(0, cutoff - history_len)
            times_slice = history["times"][slice_start:cutoff]
            rels_slice = history["relations"][slice_start:cutoff]
            heads_slice = history["heads"][slice_start:cutoff]

            length = len(times_slice)
            neighbors[i, -length:] = heads_slice
            relations[i, -length:] = rels_slice
            mask[i, -length:] = 1.0

            query_time_val = self._time_id_to_value[int(time_id)]
            past_time_vals = self._time_id_to_value[times_slice]
            deltas = np.maximum(query_time_val - past_time_vals, 0.0)
            time_deltas[i, -length:] = deltas.astype(np.float32)

        return neighbors, relations, time_deltas, mask


def read_entities(file_name: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            name, index = line.strip().split("\t")
            mapping[name] = int(index)
    return mapping


def read_relations(file_name: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            name, index = line.strip().split("\t")
            mapping[name] = int(index)
    return mapping


def _parse_token(token: str, mapping: Dict[str, int]) -> int:
    if token in mapping:
        return mapping[token]
    return int(token)


def read_quadruples(file_name: str) -> List[Tuple[int, int, int, int]]:
    data: List[Tuple[int, int, int, int]] = []
    with open(file_name, "r", encoding="utf-8") as file:
        for raw in file:
            parts = raw.strip().split("\t")
            if len(parts) < 4:
                continue
            head_token, relation_token, tail_token = parts[:3]
            time_token = parts[3]

            head_idx = _parse_token(head_token, entity_dict)
            relation_idx = _parse_token(relation_token, relation_dict)
            tail_idx = _parse_token(tail_token, entity_dict)
            time_value = int(float(time_token))

            data.append((head_idx, tail_idx, relation_idx, time_value))
    return data


def build_temporal_histories_bi(
    train_data: Iterable[Tuple[int, int, int, int]]
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]]:

    head_history_list: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    tail_history_list: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

    for head_idx, tail_idx, relation_idx, time_id in train_data:
        head_history_list[head_idx].append((time_id, relation_idx, tail_idx))
        tail_history_list[tail_idx].append((time_id, relation_idx, head_idx))

    head_histories: Dict[int, Dict[str, np.ndarray]] = {}
    for entity, events in head_history_list.items():
        events.sort(key=lambda item: item[0])
        times = np.array([event[0] for event in events], dtype=np.int64)
        relations = np.array([event[1] for event in events], dtype=np.int64)
        tails = np.array([event[2] for event in events], dtype=np.int64)
        head_histories[entity] = {"times": times, "relations": relations, "tails": tails}

    tail_histories: Dict[int, Dict[str, np.ndarray]] = {}
    for entity, events in tail_history_list.items():
        events.sort(key=lambda item: item[0])
        times = np.array([event[0] for event in events], dtype=np.int64)
        relations = np.array([event[1] for event in events], dtype=np.int64)
        heads = np.array([event[2] for event in events], dtype=np.int64)
        tail_histories[entity] = {"times": times, "relations": relations, "heads": heads}

    return head_histories, tail_histories


def _collect_time_mapping(*datasets: Sequence[Tuple[int, int, int, int]]) -> Tuple[Dict[int, int], np.ndarray]:
    unique_times = sorted({quad[3] for dataset in datasets for quad in dataset})
    time_to_id = {time: idx for idx, time in enumerate(unique_times)}
    time_id_to_value = np.array(unique_times, dtype=np.float64)
    return time_to_id, time_id_to_value


def _remap_time(dataset: Sequence[Tuple[int, int, int, int]], time_to_id: Dict[int, int]) -> List[Tuple[int, int, int, int]]:
    return [(h, t, r, time_to_id[time]) for (h, t, r, time) in dataset]


def load_data(model_args):
    global entity_dict, relation_dict

    base_dir = getattr(model_args, "data_dir", None)
    if base_dir is None:
        base_dir = "../TKG-data"
    dataset_dir = os.path.join(base_dir, model_args.dataset)
    if not os.path.exists(dataset_dir):
        legacy_dir = os.path.join("../data", model_args.dataset)
        if os.path.exists(legacy_dir):
            dataset_dir = legacy_dir
        else:
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print("reading entity dict and relation dict ...")
    entity_dict = read_entities(os.path.join(dataset_dir, "entity2id.txt"))
    relation_dict = read_relations(os.path.join(dataset_dir, "relation2id.txt"))

    print("reading train, validation, and test data ...")
    train_raw = read_quadruples(os.path.join(dataset_dir, "train.txt"))
    valid_raw = read_quadruples(os.path.join(dataset_dir, "valid.txt"))
    test_raw = read_quadruples(os.path.join(dataset_dir, "test.txt"))

    time_to_id, time_id_to_value = _collect_time_mapping(train_raw, valid_raw, test_raw)
    train_quads = _remap_time(train_raw, time_to_id)
    valid_quads = _remap_time(valid_raw, time_to_id)
    test_quads = _remap_time(test_raw, time_to_id)

    print("processing temporal histories ...")
                                                                                     
                                                                                           
    all_quads = train_quads + valid_quads + test_quads
    head_histories, tail_histories = build_temporal_histories_bi(all_quads)
    neighbor_finder = TemporalNeighborFinder(head_histories, tail_histories, time_id_to_value)

    triplets = [train_quads, valid_quads, test_quads]
    neighbor_params = neighbor_finder if getattr(model_args, "use_context", True) else None

    time_metadata = {
        "time_to_id": time_to_id,
        "time_id_to_value": time_id_to_value,
        "n_timestamps": len(time_id_to_value),
    }

    return triplets, len(relation_dict), neighbor_params, len(entity_dict), time_metadata
