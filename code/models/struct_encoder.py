from typing import Optional

import torch
import torch.nn as nn


class StructEncoder(nn.Module):
    def __init__(
        self,
        struct_type: str,
        hidden_dim: int,
        n_relations: int,
        rel_emb: nn.Embedding,
        state_fuse: str = "add",
    ):
        super().__init__()
        self.struct_type = struct_type
        self.hidden_dim = hidden_dim
        self.rel_emb = rel_emb
        self.state_fuse = state_fuse

        if self.state_fuse == "gate":
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            nn.init.xavier_uniform_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

        if struct_type == "linear":
            self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        elif struct_type == "rgcn":
            self.weight_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.weight_loop = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.rel_emb_rgcn = nn.Embedding(n_relations, hidden_dim)
            nn.init.xavier_uniform_(self.rel_emb_rgcn.weight)
        elif struct_type == "gat":
            self.proj_neigh = nn.Linear(hidden_dim, hidden_dim)
            self.proj_rel = nn.Linear(hidden_dim, hidden_dim)
            self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            raise ValueError(f"Unsupported struct_type: {struct_type}")

    def _fuse_state(self, neighbor_emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.state_fuse != "gate":
            return neighbor_emb + state

        state_magnitude = state.abs().sum(dim=-1, keepdim=True)
        is_active = (state_magnitude > 1e-6).type_as(neighbor_emb)
        h_state = torch.nn.functional.layer_norm(state, state.shape[-1:])
        gate_in = torch.cat([neighbor_emb, h_state], dim=-1)
        z = torch.sigmoid(self.fusion_gate(gate_in))
        fused_out = z * neighbor_emb + (1.0 - z) * h_state
        final_out = is_active * fused_out + (1.0 - is_active) * neighbor_emb
        return final_out

    def forward(
        self,
        neighbor_emb: torch.Tensor,
        relation_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state is not None:
            neighbor_emb = self._fuse_state(neighbor_emb, state)

        if self.struct_type == "linear":
            rel_emb = self.rel_emb(relation_ids)
            out = torch.tanh(self.proj(torch.cat([neighbor_emb, rel_emb], dim=-1)))
            return out
        if self.struct_type == "rgcn":
            rel_emb = self.rel_emb_rgcn(relation_ids)
            out = self.weight_in(neighbor_emb) + self.weight_loop(rel_emb)
            return torch.tanh(out)
        if self.struct_type == "gat":
            rel_emb = self.rel_emb(relation_ids)
            h_n = torch.tanh(self.proj_neigh(neighbor_emb))
            h_r = torch.tanh(self.proj_rel(rel_emb))
            gate = torch.sigmoid(self.gate_proj(torch.cat([neighbor_emb, rel_emb], dim=-1)))
            out = gate * h_n + (1.0 - gate) * h_r
            return out
        return neighbor_emb
