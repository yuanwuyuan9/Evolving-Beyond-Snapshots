from typing import Optional

import torch
import torch.nn as nn


class TemporalEncoderMamba(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, conv_kernel: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)

        padding = (conv_kernel - 1) // 2
        self.dw_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=padding,
            groups=hidden_dim,
        )
        self.pw_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        self.A_log = nn.Parameter(torch.randn(hidden_dim) * -0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.xavier_uniform_(self.dw_conv.weight)
        nn.init.zeros_(self.dw_conv.bias)
        nn.init.xavier_uniform_(self.pw_conv.weight)
        nn.init.zeros_(self.pw_conv.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cond_bias_in: Optional[torch.Tensor] = None,
        cond_bias_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = inputs.shape[:-2]
        seq_len = inputs.shape[-2]

        x = inputs.reshape(-1, seq_len, inputs.size(-1))
        m = mask.reshape(-1, seq_len)

        x_lin = self.in_proj(x)
        g_lin = self.gate_proj(x)

        if cond_bias_in is not None:
            x_lin = x_lin + cond_bias_in.reshape(-1, 1, self.hidden_dim)
        if cond_bias_gate is not None:
            g_lin = g_lin + cond_bias_gate.reshape(-1, 1, self.hidden_dim)

        h = torch.tanh(x_lin)
        m_float = m.unsqueeze(-1).type_as(h)
        h = h * m_float

        h_conv = self.dw_conv(h.transpose(1, 2))
        h_conv = self.pw_conv(h_conv)
        h_conv = h_conv.transpose(1, 2)

        u = torch.nn.functional.silu(h_conv + h)
        g = torch.sigmoid(g_lin)

        a = -torch.nn.functional.softplus(self.A_log)
        b = self.B
        c = self.C

        state = torch.zeros(u.size(0), self.hidden_dim, device=u.device, dtype=u.dtype)
        last = torch.zeros_like(state)
        a_exp = torch.exp(a).unsqueeze(0)
        outputs = []

        for t in range(seq_len):
            mt = m[:, t].unsqueeze(-1)
            ut = u[:, t, :]
            gt = g[:, t, :]

            new_state = gt * (a_exp * state) + (1.0 - gt) * (b.unsqueeze(0) * ut)
            state = torch.where(mt > 0.5, new_state, state)
            y = state * c.unsqueeze(0)
            last = torch.where(mt > 0.5, y, last)
            outputs.append(last.unsqueeze(1))

        y_seq = torch.cat(outputs, dim=1)
        y_seq = self.out_norm(y_seq)
        out = y_seq.reshape(*orig_shape, seq_len, self.hidden_dim)
        return out


class TemporalEncoderRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cond_bias_in: Optional[torch.Tensor] = None,
        cond_bias_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = inputs.shape[:-2]
        seq_len = inputs.shape[-2]
        x = inputs.reshape(-1, seq_len, inputs.size(-1))
        m = mask.reshape(-1, seq_len)

        x_lin = self.in_proj(x)
        g_lin = self.gate_proj(x)
        if cond_bias_in is not None:
            x_lin = x_lin + cond_bias_in.reshape(-1, 1, self.hidden_dim)
        if cond_bias_gate is not None:
            g_lin = g_lin + cond_bias_gate.reshape(-1, 1, self.hidden_dim)

        gate = torch.sigmoid(g_lin)
        x_lin = torch.tanh(x_lin) * gate

        state = torch.zeros(x_lin.size(0), self.hidden_dim, device=x_lin.device, dtype=x_lin.dtype)
        outputs = []

        for t in range(seq_len):
            xt = x_lin[:, t, :]
            mt = m[:, t].unsqueeze(-1)
            new_state = self.gru_cell(xt, state)
            state = torch.where(mt > 0.5, new_state, state)
            outputs.append(state.unsqueeze(1))

        y_seq = torch.cat(outputs, dim=1)
        y_seq = self.out_norm(y_seq)
        out = y_seq.reshape(*orig_shape, seq_len, self.hidden_dim)
        return out


class TemporalEncoderTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        cond_bias_in: Optional[torch.Tensor] = None,
        cond_bias_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del cond_bias_gate  # Kept for interface parity across encoders.

        x = self.in_proj(inputs)
        _, seq_len, _ = x.shape

        if cond_bias_in is not None:
            x = x + cond_bias_in.unsqueeze(1)

        x = x + self.pos_emb[:, :seq_len, :]
        key_padding_mask = (mask == 0).bool()

        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        x = x.transpose(0, 1)
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        out = out.transpose(0, 1)
        out = self.out_norm(out)
        out = out * mask.unsqueeze(-1)
        return out
