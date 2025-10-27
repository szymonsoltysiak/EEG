# EEGTransformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralEncoding(nn.Module):
    """Encodes spectral (frequency-domain) features using FFT magnitude + phase."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, T)
        fft = torch.fft.rfft(x, dim=-1)
        mag, phase = torch.abs(fft), torch.angle(fft)
        # mag, phase: (B, C, F)
        spec = torch.cat([mag, phase], dim=-1)  # (B, C, 2*F)
        B, C, F2 = spec.shape

        if self.proj is None:
            in_dim = F2
            self.proj = nn.Linear(in_dim, self.d_model).to(spec.device)

        spec = self.proj(spec)  # (B, C, d_model)
        spec = spec.permute(0, 2, 1).contiguous()  # (B, d_model, C)
        return self.norm(spec)  # (B, d_model, C)

class RelativePositionBias(nn.Module):
    """
    Learnable relative bias table.
    Returns tensor shaped (num_heads, seq_len, seq_len)
    """
    def __init__(self, num_heads, max_distance=256):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # bias_table indexed by relative distance in [-max_distance+1, max_distance-1]
        self.bias_table = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, seq_len, device=None):
        # seq_len: int
        if device is None:
            device = self.bias_table.device
        coords = torch.arange(seq_len, device=device)
        rel_pos = coords[None, :] - coords[:, None]  # (seq_len, seq_len) in [-(seq_len-1),(seq_len-1)]
        # clamp to range representable by bias_table
        clipped = rel_pos.clamp(-self.max_distance + 1, self.max_distance - 1) + (self.max_distance - 1)
        # clipped in [0, 2*max_distance-2]
        bias = self.bias_table[clipped]  # (seq_len, seq_len, num_heads)
        # permute to (num_heads, seq_len, seq_len)
        bias = bias.permute(2, 0, 1).contiguous()
        return bias  # (num_heads, seq_len, seq_len)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.4, max_distance=256):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # qkv projections and out projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pos_bias = RelativePositionBias(nhead, max_distance)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        # linear projections
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for heads: (B, nhead, N, d_k)
        q = q.view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)  # (B, H, N, d_k)
        k = k.view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)
        v = v.view(B, N, self.nhead, self.d_k).permute(0, 2, 1, 3)

        # scaled dot-product
        # attn_logits: (B, H, N, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # add relative position bias: (H, N, N) -> broadcast to (B, H, N, N)
        bias = self.pos_bias(N, device=x.device)  # (H, N, N)
        attn_logits = attn_logits + bias.unsqueeze(0)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # attn @ v -> (B, H, N, d_k)
        out = torch.matmul(attn, v)

        # combine heads: (B, H, N, d_k) -> (B, N, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        # residual + norm
        x = x + out
        x = self.norm1(x)

        # feed-forward
        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)
        return x

class EEGClassifier(nn.Module):
    def __init__(self, num_channels=12, num_classes=4, window_size=128,
                 d_model=64, nhead=2, num_layers=2, dim_feedforward=128,
                 dropout=0.2, use_cls_token=True):
        super().__init__()

        self.use_cls_token = use_cls_token

        # convolutional patch embedding: reduces temporal length -> T'
        self.patch_embed = nn.Conv1d(num_channels, d_model, kernel_size=8, stride=4, padding=4)
        self.spec_encoding = SpectralEncoding(d_model)

        # encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, max_distance=max(8, window_size // 4))
            for _ in range(num_layers)
        ])

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)

        # classification head (2-layer MLP)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        # patch embed
        x = self.patch_embed(x)         # (B, d_model, T')
        x = x.permute(0, 2, 1)          # (B, T', d_model)

        # compute spectral encoding: expects (B, d_model, T') input-ish -> returns (B, d_model, F)
        spec = self.spec_encoding(x.transpose(1, 2))  # (B, d_model, F)
        # interpolate frequency axis to token count
        spec = F.interpolate(spec, size=x.shape[1], mode='linear', align_corners=False)  # (B, d_model, T')
        spec = spec.transpose(1, 2)  # (B, T', d_model)

        # combine and normalize
        x = self.norm(x + spec)
        x = F.dropout(x, p=0.4, training=self.training)

        # add cls token
        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)  # (B,1,d)
            x = torch.cat((cls, x), dim=1)  # (B, T'+1, d)

        # encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        rep = x[:, 0, :] if self.use_cls_token else x.mean(dim=1)
        logits = self.head(rep)
        return logits

# Hyperparameters
params = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'num_epochs': 60,
    'num_classes': 4,
    'window_size': 128,
    'num_channels': 16,
    'step_size': 10,
    'gamma': 0.7
}
