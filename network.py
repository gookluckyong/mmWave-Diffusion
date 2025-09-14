from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utility: adaLN-style modulation
# ------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, C]; shift/scale: [B, C]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ------------------------------
# Timestep sinusoidal embedding + small MLP
# ------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
        """Create sinusoidal timestep embeddings (PyTorch diffusion standard)."""
        half = dim // 2
        device = timesteps.device
        exponent = -math.log(max_period) * torch.arange(half, device=device) / half
        freq = torch.exp(exponent)
        args = timesteps.float()[:, None] * freq[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

# ------------------------------
# Final layer: unpatching back to [B, C, L]
# ------------------------------
class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # [B, T, patch_size*C]
        B, T, _ = x.shape
        x = x.view(B, T, self.out_channels, self.patch_size).permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.out_channels, T * self.patch_size)
        return x

# ------------------------------
#RDT block 
# ------------------------------
class RDTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-Attention + learnable gate (scalar per block)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_gate = nn.Parameter(torch.tensor(0.0))

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        

        # adaLN-Zero style for 3 modules (self/cross/mlp), each with shift/scale/gate -> 9*C
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))


        self._init_zero()

    def _init_zero(self):
        # Zero the last projections to stabilize (adaLN Zero idea)
        nn.init.zeros_(self.self_attn.out_proj.weight)
        if self.self_attn.out_proj.bias is not None:
            nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        # Zero the last linear of MLP
        last_linear = self.mlp[2]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, C]
        context: torch.Tensor,      # [B, S, C]
        t: torch.Tensor,            # [B, C]
        cross_attn_mask: torch.Tensor | None = None,  # [T, S] bool mask (True=disallow)
    ) -> torch.Tensor:
        # Unpack adaLN parameters
        (
            shift_msa, scale_msa, gate_msa,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(t).chunk(9, dim=1)

        # Self-Attention
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + torch.tanh(gate_msa).unsqueeze(1) * attn_out

        # Cross-Attention with learnable gate + optional local mask
        x_norm_cross = modulate(self.norm_cross(x), shift_cross, scale_cross)
        cross_out, _ = self.cross_attn(x_norm_cross, context, context, attn_mask=cross_attn_mask, need_weights=False)
        # Block-level learnable gate + time-conditioned gate
        cross_scale = torch.tanh(self.cross_gate) * torch.tanh(gate_cross).unsqueeze(1)
        x = x + cross_scale * cross_out
        # FFN
        x_mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + torch.tanh(gate_mlp).unsqueeze(1) * self.mlp(x_mlp_in)
        return x

# ------------------------------
# Full RDT with x/y patch embedders and optional cross-local window
# ------------------------------
class RDT(nn.Module):
    def __init__(
        self,
        sequence_length: int =400,
        in_channels: int =1,
        cond_channels: int =1,
        out_channels: int =1,
        patch_size: int =20,
        hidden_size:int = 256,
        depth:int = 2,
        num_heads:int = 4,
        mlp_ratio: int = 4,
        cross_local_window: int = 1,   # 交叉注意力局部窗口，单位=token（±window）
    ):
        super().__init__()
        assert sequence_length % patch_size == 0, "sequence_length must be divisible by patch_size"
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.cross_local_window = int(cross_local_window)

        # Patch embedders (non-overlapping)
        self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.y_embedder = nn.Conv1d(cond_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = sequence_length // patch_size

        # Positional embeddings for x and y
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RDTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed_y, std=0.02)

        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

    def _build_cross_mask(self, T: int, S: int, device: torch.device) -> torch.Tensor | None:
        if self.cross_local_window is None or self.cross_local_window <= 0:
            return None
        # 允许 T!=S 的情况：对齐到 min(T,S) 的主对角
        t_idx = torch.arange(T, device=device).unsqueeze(1)  # [T,1]
        s_idx = torch.arange(S, device=device).unsqueeze(0)  # [1,S]
        dist = (t_idx - s_idx).abs()  # [T,S]
        mask = dist > self.cross_local_window  # True=禁止关注  在这段代码里 T、S 是 patch 之后的 token 长度，dist = |t_idx - s_idx| 就是两个 token 的索引差。所以：cross_local_window = 1 ⇒ 允许 ±1 个 token 的跨注意力； 那我patch_size=20的话,1个token就是1秒
        return mask

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, lq: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """
        x: [B, C_in, L] noisy sample at time t
        lq: [B, C_cond, L] conditioning (radar phase), passed as 'lq' from diffusion code
        timesteps: [B]
        return: [B, C_out, L]
        """
        assert lq is not None, "Expected conditioning input 'lq' (radar signal) to be provided."

        B, _, L = x.shape
        assert L == self.sequence_length, f"expected L={self.sequence_length}, got {L}"

        # 1) Patch embedding -> [B, T, C]
        x_tok = self.x_embedder(x).transpose(1, 2)
        y_tok = self.y_embedder(lq).transpose(1, 2)

        # 2) Add positional encodings
        x_tok = x_tok + self.pos_embed
        y_tok = y_tok + self.pos_embed_y

        # 3) Timestep embedding
        t = self.t_embedder(timesteps)  # [B, C]

        # 4) Optional cross-attention local mask
        T_len = x_tok.size(1)
        S_len = y_tok.size(1)
        attn_mask = self._build_cross_mask(T_len, S_len, x_tok.device)

        # 5) Transformer blocks
        h = x_tok
        for blk in self.blocks:
            h = blk(h, y_tok, t, attn_mask)

        # 6) Final unpatching
        out = self.final_layer(h, t)  # [B, C_out, L]
        return out
