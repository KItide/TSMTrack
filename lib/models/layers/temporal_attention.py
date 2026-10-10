from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class BaseTemporalAttention(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, feat):
        raise NotImplementedError


class TemporalSelfAttention(BaseTemporalAttention):
    """
    Three stacked temporal self-attention layers followed by an FFN head.
    Input shape: [T, B, C, H, W].
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()

        num_heads = max(1, min(num_heads, embed_dim))
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads -= 1

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn3 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.gain = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, feat):
        T, B, C, H, W = feat.shape
        tokens = feat.mean(dim=(-2, -1)).permute(1, 0, 2)

        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=feat.device) * float('-inf'), diagonal=1)

            attn_out1, _ = self.attn1(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            tokens = self.norm1(tokens + attn_out1)

            attn_out2, _ = self.attn2(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            tokens = self.norm2(tokens + attn_out2)

            attn_out3, _ = self.attn3(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            tokens = self.norm3(tokens + attn_out3)

        tokens = self.norm_ffn(tokens + self.ffn(tokens))
        gates = torch.tanh(self.gain(tokens)).permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gates)


class Fused_TAV(BaseTemporalAttention):
    """
    Fused temporal attention for RGB and infrared branches.
    Input shape: feat_rgb/feat_ir [T, B, C, H, W].
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim

        num_heads = max(1, min(num_heads, embed_dim))
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads -= 1

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn3 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1_rgb = nn.LayerNorm(embed_dim)
        self.norm1_ir = nn.LayerNorm(embed_dim)
        self.norm2_rgb = nn.LayerNorm(embed_dim)
        self.norm2_ir = nn.LayerNorm(embed_dim)
        self.norm3_rgb = nn.LayerNorm(embed_dim)
        self.norm3_ir = nn.LayerNorm(embed_dim)

        self.norm_ffn_rgb = nn.LayerNorm(embed_dim)
        self.norm_ffn_ir = nn.LayerNorm(embed_dim)

        def build_ffn(dim):
            return nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )

        self.ffn_rgb = build_ffn(embed_dim)
        self.ffn_ir = build_ffn(embed_dim)

        self.gain_rgb = nn.Linear(embed_dim, embed_dim)
        self.gain_ir = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.gain_rgb.weight)
        nn.init.zeros_(self.gain_rgb.bias)
        nn.init.zeros_(self.gain_ir.weight)
        nn.init.zeros_(self.gain_ir.bias)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask_base", mask)

    def _get_fused_mask(self, T, device):
        if T > self.causal_mask_base.shape[0]:
            m = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        else:
            m = self.causal_mask_base[:T, :T].to(device)
        return torch.cat([torch.cat([m, m], dim=1), torch.cat([m, m], dim=1)], dim=0)

    def forward(self, feat_rgb, feat_ir):
        T, B, C, H, W = feat_rgb.shape

        tokens_rgb = feat_rgb.mean(dim=(-2, -1)).transpose(0, 1)
        tokens_ir = feat_ir.mean(dim=(-2, -1)).transpose(0, 1)

        if T > 1:
            tokens = torch.cat([tokens_rgb, tokens_ir], dim=1)
            mask = self._get_fused_mask(T, tokens.device)

            out1, _ = self.attn1(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb1, delta_ir1 = torch.split(out1, T, dim=1)
            tokens_rgb = self.norm1_rgb(tokens_rgb + delta_rgb1)
            tokens_ir = self.norm1_ir(tokens_ir + delta_ir1)

            tokens = torch.cat([tokens_rgb, tokens_ir], dim=1)
            out2, _ = self.attn2(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb2, delta_ir2 = torch.split(out2, T, dim=1)
            tokens_rgb = self.norm2_rgb(tokens_rgb + delta_rgb2)
            tokens_ir = self.norm2_ir(tokens_ir + delta_ir2)

            tokens = torch.cat([tokens_rgb, tokens_ir], dim=1)
            out3, _ = self.attn3(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb3, delta_ir3 = torch.split(out3, T, dim=1)
            tokens_rgb = self.norm3_rgb(tokens_rgb + delta_rgb3)
            tokens_ir = self.norm3_ir(tokens_ir + delta_ir3)

        tokens_rgb = self.norm_ffn_rgb(tokens_rgb + self.ffn_rgb(tokens_rgb))
        tokens_ir = self.norm_ffn_ir(tokens_ir + self.ffn_ir(tokens_ir))

        gates_rgb = torch.tanh(self.gain_rgb(tokens_rgb)).transpose(0, 1).view(T, B, C, 1, 1)
        gates_ir = torch.tanh(self.gain_ir(tokens_ir)).transpose(0, 1).view(T, B, C, 1, 1)

        return feat_rgb * (1.0 + gates_rgb), feat_ir * (1.0 + gates_ir)

    def forward_topk(self, tokens_rgb, tokens_ir, topk_indices):
        """
        Apply temporal attention only on target-related top-k search tokens.

        Vote across all frames' top-k indices to obtain one shared spatial
        position set per batch item, then gather those positions from every
        frame for temporal interaction.

        Args:
            tokens_rgb/tokens_ir: [T, B, N, C]
            topk_indices: [T, B, K], reused from sparse CSS routing
        """
        T, B, N, C = tokens_rgb.shape
        K = topk_indices.shape[-1]

        # Current implementation: vote across frames to obtain one shared
        # spatial index set.
        vote = tokens_rgb.new_zeros(B, N)
        flat_indices = topk_indices.permute(1, 0, 2).reshape(B, -1)
        vote.scatter_add_(
            1,
            flat_indices,
            torch.ones_like(flat_indices, dtype=vote.dtype),
        )
        aligned_indices = torch.topk(vote, k=K, dim=1).indices
        gather_index = aligned_indices.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, C)

        # Alternative implementation: directly reuse the current frame's
        # positions in every frame, without voting.
        # current_topk_indices = topk_indices[-1]
        # gather_index = current_topk_indices.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, C)

        selected_rgb = torch.gather(tokens_rgb, dim=2, index=gather_index)
        selected_ir = torch.gather(tokens_ir, dim=2, index=gather_index)

        seq_rgb = selected_rgb.permute(1, 2, 0, 3).reshape(B * K, T, C)
        seq_ir = selected_ir.permute(1, 2, 0, 3).reshape(B * K, T, C)

        if T > 1:
            tokens = torch.cat([seq_rgb, seq_ir], dim=1)
            mask = self._get_fused_mask(T, tokens.device)

            out1, _ = self.attn1(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb1, delta_ir1 = torch.split(out1, T, dim=1)
            seq_rgb = self.norm1_rgb(seq_rgb + delta_rgb1)
            seq_ir = self.norm1_ir(seq_ir + delta_ir1)

            tokens = torch.cat([seq_rgb, seq_ir], dim=1)
            out2, _ = self.attn2(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb2, delta_ir2 = torch.split(out2, T, dim=1)
            seq_rgb = self.norm2_rgb(seq_rgb + delta_rgb2)
            seq_ir = self.norm2_ir(seq_ir + delta_ir2)

            tokens = torch.cat([seq_rgb, seq_ir], dim=1)
            out3, _ = self.attn3(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
            delta_rgb3, delta_ir3 = torch.split(out3, T, dim=1)
            seq_rgb = self.norm3_rgb(seq_rgb + delta_rgb3)
            seq_ir = self.norm3_ir(seq_ir + delta_ir3)

        seq_rgb = self.norm_ffn_rgb(seq_rgb + self.ffn_rgb(seq_rgb))
        seq_ir = self.norm_ffn_ir(seq_ir + self.ffn_ir(seq_ir))

        gates_rgb = torch.tanh(self.gain_rgb(seq_rgb)).reshape(B, K, T, C).permute(2, 0, 1, 3)
        gates_ir = torch.tanh(self.gain_ir(seq_ir)).reshape(B, K, T, C).permute(2, 0, 1, 3)

        enhanced_rgb = selected_rgb * (1.0 + gates_rgb)
        enhanced_ir = selected_ir * (1.0 + gates_ir)

        out_rgb = tokens_rgb.clone()
        out_ir = tokens_ir.clone()
        out_rgb.scatter_(dim=2, index=gather_index, src=enhanced_rgb)
        out_ir.scatter_(dim=2, index=gather_index, src=enhanced_ir)
        return out_rgb, out_ir



def build_temporal_attention(attention_type, embed_dim, num_heads=8, dropout=0.1):
    attn_type = str(attention_type).lower()
    if attn_type in ('self', 'self_attn', 'temporal_self_attention'):
        return TemporalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    if attn_type in ('fused', 'fused_tav'):
        return Fused_TAV(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    raise ValueError('Unknown temporal attention type: {}'.format(attention_type))
