from typing import Dict

import torch
import torch.nn as nn

from .struct_encoder import StructEncoder
from .temporal_encoders import (
    TemporalEncoderMamba,
    TemporalEncoderRNN,
    TemporalEncoderTransformer,
)
from .time import TimeDeltaProjection


class TRMamba(nn.Module):
    def __init__(self, args, n_relations: int, n_entity: int, time_metadata: Dict):
        super().__init__()
        self.use_gpu = bool(getattr(args, "cuda", False)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.time_encoding = getattr(args, "time_encoding", "id")
        self.scoring_fn = getattr(args, "scoring_fn", "distmult")
        self.use_struct_encoder = bool(getattr(args, "use_struct_encoder", False))
        self.struct_type = getattr(args, "struct_type", "linear")
        self.use_state_writeback = True
        self.state_alpha = float(getattr(args, "state_alpha", 0.2))
        self.state_fuse = getattr(args, "state_fuse", "add")
        self.temporal_encoder_type = getattr(args, "temporal_encoder", "mamba")

        self.n_relations = n_relations
        self.n_entity = n_entity
        self.hidden_dim = args.dim
        self.time_emb_dim = getattr(args, "time_emb_dim", args.dim)
        self.history_len = getattr(args, "history_len", 32)
        self.use_context = getattr(args, "use_context", True)

        n_timestamps = time_metadata["n_timestamps"]

        self.entity_emb = nn.Embedding(n_entity, self.hidden_dim)
        self.relation_emb = nn.Embedding(n_relations, self.hidden_dim)
        self.time_emb = nn.Embedding(n_timestamps, self.time_emb_dim)

        if self.use_struct_encoder:
            self.struct_encoder = StructEncoder(
                self.struct_type,
                self.hidden_dim,
                n_relations,
                self.relation_emb,
                state_fuse=self.state_fuse,
            )

        if self.scoring_fn == "mlp":
            self.mlp_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if self.scoring_fn in ("complex", "rotate"):
            self.entity_emb_im = nn.Embedding(n_entity, self.hidden_dim)
            self.head_im_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.scoring_fn == "complex":
            self.relation_emb_im = nn.Embedding(n_relations, self.hidden_dim)
        if self.scoring_fn == "rotate":
            self.relation_phase = nn.Embedding(n_relations, self.hidden_dim)

        self.delta_proj = TimeDeltaProjection(self.time_emb_dim)

        if self.use_state_writeback:
            self.register_buffer("entity_state_fast", torch.zeros(n_entity, self.hidden_dim), persistent=False)
            self.register_buffer("entity_state_slow", torch.zeros(n_entity, self.hidden_dim), persistent=False)
            self.gate_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
            self.gate_scale = nn.Parameter(torch.tensor(5.0), requires_grad=False)

        time_id_to_value = time_metadata["time_id_to_value"]
        time_min = float(time_id_to_value.min()) if time_id_to_value.size > 0 else 0.0
        time_max = float(time_id_to_value.max()) if time_id_to_value.size > 0 else 1.0
        time_scale = max(time_max - time_min, 1.0)
        self.register_buffer("time_id_to_value", torch.from_numpy(time_id_to_value).float(), persistent=False)
        self.register_buffer("time_min", torch.tensor(time_min, dtype=torch.float32), persistent=False)
        self.register_buffer("time_scale", torch.tensor(time_scale, dtype=torch.float32), persistent=False)

        query_input_dim = self.hidden_dim * 2 + self.time_emb_dim
        history_input_dim = self.hidden_dim * 2 + self.time_emb_dim

        self.query_proj = nn.Linear(query_input_dim, self.hidden_dim)
        if self.temporal_encoder_type == "mamba":
            self.temporal_encoder = TemporalEncoderMamba(history_input_dim, self.hidden_dim)
        elif self.temporal_encoder_type == "rnn":
            self.temporal_encoder = TemporalEncoderRNN(history_input_dim, self.hidden_dim)
        elif self.temporal_encoder_type == "transformer":
            self.temporal_encoder = TemporalEncoderTransformer(
                history_input_dim,
                self.hidden_dim,
                n_layers=2,
                n_heads=4,
            )
        else:
            raise ValueError(f"Unsupported temporal_encoder: {self.temporal_encoder_type}")

        self.history_attn = nn.Linear(self.hidden_dim * 2, 1)
        self.hyper_in = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hyper_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.register_buffer("zero_context", torch.zeros(1, self.hidden_dim), persistent=False)

        self.reset_parameters()
        self.to(self.device)

    def _encode_query_time(self, times: torch.Tensor) -> torch.Tensor:
        if self.time_encoding == "id":
            return self.time_emb(times)
        time_vals = self.time_id_to_value[times]
        norm_deltas = (time_vals - self.time_min) / self.time_scale
        delta_feat = self.delta_proj(norm_deltas)
        if self.time_encoding == "delta":
            return delta_feat
        if self.time_encoding == "id_plus_delta":
            return self.time_emb(times) + delta_feat
        return self.time_emb(times)

    def _build_history_context(
        self,
        query_vec: torch.Tensor,
        rel_emb: torch.Tensor,
        history_entities: torch.Tensor,
        history_relations: torch.Tensor,
        history_deltas: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_emb = self.entity_emb(history_entities)
        if self.use_struct_encoder and hasattr(self, "struct_encoder"):
            state_in = None
            if self.use_state_writeback and hasattr(self, "entity_state_slow"):
                state_in = self.entity_state_slow[history_entities]
            neighbor_emb = self.struct_encoder(neighbor_emb, history_relations, state_in)

        relation_hist_emb = self.relation_emb(history_relations)
        delta_feat = self.delta_proj(history_deltas)
        history_input = torch.cat([neighbor_emb, relation_hist_emb, delta_feat], dim=-1)
        cond_in = self.hyper_in(rel_emb)
        cond_gate = self.hyper_gate(rel_emb)
        history_seq = self.temporal_encoder(history_input, history_mask, cond_in, cond_gate)

        _, hist_len, _ = history_seq.shape
        query_expanded = query_vec.unsqueeze(1).expand(-1, hist_len, -1)
        attn_input = torch.cat([history_seq, query_expanded], dim=-1)
        scores = torch.tanh(self.history_attn(attn_input)).squeeze(-1)
        scores = scores.masked_fill(history_mask <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * history_mask
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn = attn / denom
        return torch.sum(history_seq * attn.unsqueeze(-1), dim=1)

    def _score_candidates(self, rep: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        cand_emb_all = self.entity_emb.weight
        r_eff = self.relation_emb(relations)
        eff = rep * r_eff

        if self.scoring_fn == "mlp":
            mlp_in = torch.cat([rep, r_eff], dim=-1)
            rep_mlp = torch.tanh(self.mlp_proj(mlp_in))
            scores = rep_mlp @ cand_emb_all.t()
        elif self.scoring_fn == "complex":
            h_r = rep
            h_i = self.head_im_proj(rep)
            r_r = r_eff
            r_i = self.relation_emb_im(relations)
            t_r = cand_emb_all
            t_i = self.entity_emb_im.weight
            real_term = (h_r * r_r - h_i * r_i) @ t_r.t()
            imag_term = (h_r * r_i + h_i * r_r) @ t_i.t()
            scores = real_term + imag_term
        elif self.scoring_fn == "rotate":
            h_r = rep
            h_i = self.head_im_proj(rep)
            phase = torch.tanh(self.relation_phase(relations))
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            h_r_rot = h_r * cos_phase - h_i * sin_phase
            h_i_rot = h_r * sin_phase + h_i * cos_phase
            t_r = cand_emb_all
            t_i = self.entity_emb_im.weight
            head_norm = (h_r_rot.pow(2) + h_i_rot.pow(2)).sum(dim=1, keepdim=True)
            tail_norm = (t_r.pow(2) + t_i.pow(2)).sum(dim=1)
            dot = h_r_rot @ t_r.t() + h_i_rot @ t_i.t()
            scores = -(head_norm + tail_norm.unsqueeze(0) - 2.0 * dot)
        elif self.scoring_fn == "transe":
            q = rep + r_eff
            diff = q.unsqueeze(1) - cand_emb_all.unsqueeze(0)
            scores = -torch.linalg.norm(diff, dim=-1, ord=2)
        else:
            scores = eff @ cand_emb_all.t()

        return torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    def _maybe_writeback_states(self, heads: torch.Tensor, context_head: torch.Tensor) -> None:
        if not hasattr(self, "entity_state_fast"):
            return
        with torch.no_grad():
            unique_heads, idx = torch.unique(heads, return_inverse=True)

            ctx_sum = torch.zeros(unique_heads.size(0), self.hidden_dim, device=self.device)
            counts = torch.zeros(unique_heads.size(0), device=self.device)
            ctx_sum.index_add_(0, idx, context_head)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            counts = counts.clamp(min=1.0).unsqueeze(-1)
            ctx_mean = ctx_sum / counts

            s_fast = self.entity_state_fast[unique_heads]
            new_fast = (1 - self.state_alpha) * s_fast + self.state_alpha * ctx_mean
            self.entity_state_fast[unique_heads] = new_fast

            s_slow = self.entity_state_slow[unique_heads]
            diff = new_fast - s_slow
            delta = torch.norm(diff, p=2, dim=-1, keepdim=True)
            gate_logits = self.gate_scale * (delta - self.gate_threshold)
            gate = torch.sigmoid(gate_logits)

            new_slow = s_slow + gate * diff
            self.entity_state_slow[unique_heads] = new_slow

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.time_emb.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.history_attn.weight)
        nn.init.zeros_(self.history_attn.bias)
        nn.init.xavier_uniform_(self.hyper_in.weight)
        nn.init.zeros_(self.hyper_in.bias)
        nn.init.xavier_uniform_(self.hyper_gate.weight)
        nn.init.zeros_(self.hyper_gate.bias)

        if hasattr(self, "mlp_proj"):
            nn.init.xavier_uniform_(self.mlp_proj.weight)
            nn.init.zeros_(self.mlp_proj.bias)
        if hasattr(self, "entity_emb_im"):
            nn.init.xavier_uniform_(self.entity_emb_im.weight)
        if hasattr(self, "relation_emb_im"):
            nn.init.xavier_uniform_(self.relation_emb_im.weight)
        if hasattr(self, "relation_phase"):
            nn.init.xavier_uniform_(self.relation_phase.weight)
        if hasattr(self, "head_im_proj"):
            nn.init.xavier_uniform_(self.head_im_proj.weight)
            nn.init.zeros_(self.head_im_proj.bias)
        if hasattr(self, "struct_encoder") and isinstance(self.struct_encoder, StructEncoder):
            for mod in self.struct_encoder.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        for layer in self.delta_proj.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        if hasattr(self, "consolidation_mlp"):
            for layer in self.consolidation_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward_full(self, batch, step) -> Dict[str, torch.Tensor]:
        heads = batch["heads"].to(self.device)
        relations = batch["relations"].to(self.device)
        times = batch["times"].to(self.device)

        batch_size = heads.size(0)
        head_emb = self.entity_emb(heads)
        rel_emb = self.relation_emb(relations)
        time_emb = self._encode_query_time(times)

        query_input = torch.cat([head_emb, rel_emb, time_emb], dim=-1)
        query_vec = torch.tanh(self.query_proj(query_input))

        context_head = self.zero_context.expand(batch_size, -1)
        if self.use_context and "history_entities" in batch:
            history_entities = batch["history_entities"].to(self.device)
            history_relations = batch["history_relations"].to(self.device)
            history_deltas = batch["history_deltas"].to(self.device)
            history_mask = batch["history_mask"].to(self.device)

            context_head = self._build_history_context(
                query_vec=query_vec,
                rel_emb=rel_emb,
                history_entities=history_entities,
                history_relations=history_relations,
                history_deltas=history_deltas,
                history_mask=history_mask,
            )

        scores = self._score_candidates(context_head, relations)

        if self.training and self.use_state_writeback and (step >= 0):
            self._maybe_writeback_states(heads, context_head)

        return {"scores": scores}
