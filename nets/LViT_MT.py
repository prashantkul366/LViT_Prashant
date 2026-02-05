# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule
from einops import rearrange, repeat
import math 
from kan_fJNB import KAN
from transformers import AutoTokenizer, AutoModel
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class ClinicalTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")

        for p in self.encoder.parameters():
            p.requires_grad = False  # freeze

    def forward(self, texts):
        """
        texts: list[str], length = B
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(next(self.encoder.parameters()).device)

        out = self.encoder(**tokens)
        return out.last_hidden_state.mean(dim=1)  # (B, 768)

class FiLM(torch.nn.Module):
    def __init__(self, img_channels, text_dim=768):
        super().__init__()
        self.gamma = torch.nn.Linear(text_dim, img_channels)
        self.beta = torch.nn.Linear(text_dim, img_channels)

    def forward(self, x, t):
        gamma = self.gamma(t).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(t).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class TokenMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner = Attention(dim, num_heads, bias)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Token count N must be a perfect square"

        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)   # (B, D, H, W)
        out_2d = self.inner(x_2d)                       # (B, D, H, W)
        out = out_2d.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # no explicit attention weights here
        weights = None
        return out, weights


class FKANMLP(nn.Module):
    """
    Simple KAN-based MLP for token features.
    Input / output: (B, N, C)
    """
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.kan = KAN(
            layers_hidden=[dim, mlp_dim, dim],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        x = self.norm(x)
        x_flat = x.reshape(B * N, C)
        y_flat = self.kan(x_flat)
        y = y_flat.view(B, N, C)
        y = self.dropout(y)
        return y
    
class TransformerMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        mlp_dim = int(dim * mlp_ratio)

        # --- Transformer part ---
        print("At Transformer ")
        self.ln1 = nn.LayerNorm(dim)      # for MDTA
        self.attn = TokenMDTA(dim=dim, num_heads=num_heads, bias=True)

        self.ln2 = nn.LayerNorm(dim)      # for first f-KAN
        self.ffn1 = FKANMLP(dim, mlp_dim)

        # --- Mamba part ---
        print("At MambaVisionMixer ")
        self.ln3 = nn.LayerNorm(dim)      # for VSSM
        self.vssm = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.ln4 = nn.LayerNorm(dim)      # for second f-KAN
        self.ffn2 = FKANMLP(dim, mlp_dim)

    def forward(self, x5d):
        # x5d: (B, C, D, H, W)
        B, C = x5d.shape[:2]
        D, H, W = x5d.shape[2:]
        N = D * H * W
        # print(f"[TMB] x5d in:         {x5d.shape}")

        # ===== flatten to tokens =====
        x = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)
        x_in = x                                # original input tokens
        # print(f"[TMB] tokens x_in:    {x_in.shape}")   # (B, N, C)

        # ================= TRANSFORMER PART =================
        # 1) LN -> MDTA -> add residual (orig input)
        t = self.ln1(x_in)
        # print(f"[TMB] after ln1:      {t.shape}")
        t, _ = self.attn(t)                     # (B, N, C)
        # print(f"[TMB] after attn:     {t.shape}")
        t = x_in + t                            # attn_residual
        # print(f"[TMB] after attn res: {t.shape}")

        # 2) LN -> f-KAN -> add residual (orig input)
        u = self.ln2(t)
        # print(f"[TMB] after ln2:      {u.shape}")
        u = self.ffn1(u)                        # (B, N, C)
        # print(f"[TMB] after fKAN1:    {u.shape}")
        u = u + t                            # f-KAN residual
        # print(f"[TMB] after fKAN1 res:{u.shape}")
        x_tr = x_in + u                         # transformer output
        # print(f"[TMB] x_tr:           {x_tr.shape}")

        # ================== MAMBA PART =====================
        # 3) LN -> VSSM -> add residual (transformer output)
        m = self.ln3(x_tr)
        # print(f"[TMB] after ln3:      {m.shape}")
        m = self.vssm(m)                        # (B, N, C)
        # print(f"[TMB] after VSSM:     {m.shape}")
        m = x_tr + m                            # mamba_residual
        # print(f"[TMB] after VSSM res: {m.shape}")

        # 4) LN -> f-KAN -> add residual (transformer output)
        n = self.ln4(m)
        # print(f"[TMB] after ln4:      {n.shape}")
        n = self.ffn2(n)                        # (B, N, C)
        # print(f"[TMB] after fKAN2:    {n.shape}")
        n = n + m                            # f-KAN residual
        # print(f"[TMB] after fKAN2 res:{n.shape}")
        x_out_tokens = x_tr + n                 # final output tokens
        # print(f"[TMB] x_out_tokens:   {x_out_tokens.shape}")

        # ===== back to 5D =====
        x_out = x_out_tokens.transpose(1, 2).view(B, C, D, H, W)
        # print(f"[TMB] x_out 5D:       {x_out.shape}")
        return x_out
    
class TransformerMambaBlock2D(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 d_state=8, d_conv=3, expand=1):
        super().__init__()

        mlp_dim = int(dim * mlp_ratio)

        self.ln1 = nn.LayerNorm(dim)
        self.attn = TokenMDTA(dim=dim, num_heads=num_heads, bias=True)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn1 = FKANMLP(dim, mlp_dim)

        self.ln3 = nn.LayerNorm(dim)
        self.vssm = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.ln4 = nn.LayerNorm(dim)
        self.ffn2 = FKANMLP(dim, mlp_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # ---- flatten to tokens ----
        tokens = x.view(B, C, N).transpose(1, 2)  # (B, N, C)
        x_in = tokens

        # ---- Transformer Attention ----
        t = self.ln1(x_in)
        t, _ = self.attn(t)
        t = x_in + t

        # ---- FKAN FFN ----
        u = self.ln2(t)
        u = self.ffn1(u)
        u = u + t
        x_tr = x_in + u

        # ---- Mamba Mixer ----
        m = self.ln3(x_tr)
        m = self.vssm(m)
        m = x_tr + m

        # ---- Second FFN ----
        n = self.ln4(m)
        n = self.ffn2(n)
        n = n + m

        out_tokens = x_tr + n

        # ---- reshape back ----
        out = out_tokens.transpose(1, 2).view(B, C, H, W)

        return out

class MambaEncoder2D(nn.Module):
    def __init__(self, in_ch=3, base_dim=64):
        super().__init__()

        self.stem = nn.Conv2d(in_ch, base_dim, 3, padding=1)

        self.stage1 = TransformerMambaBlock2D(dim=base_dim)

        self.down1 = nn.Conv2d(base_dim, base_dim*2, 2, 2)
        self.stage2 = TransformerMambaBlock2D(dim=base_dim*2)

        self.down2 = nn.Conv2d(base_dim*2, base_dim*4, 2, 2)
        self.stage3 = TransformerMambaBlock2D(dim=base_dim*4)

        self.down3 = nn.Conv2d(base_dim*4, base_dim*8, 2, 2)
        self.stage4 = TransformerMambaBlock2D(dim=base_dim*8)

        self.down4 = nn.Conv2d(base_dim*8, base_dim*8, 2, 2)
        self.stage5 = TransformerMambaBlock2D(dim=base_dim*8)

    def forward(self, x):
        x = self.stem(x)

        x1 = self.stage1(x)              # (B,64,224,224)
        x2 = self.stage2(self.down1(x1)) # (B,128,112,112)
        x3 = self.stage3(self.down2(x2)) # (B,256,56,56)
        x4 = self.stage4(self.down3(x3)) # (B,512,28,28)
        x5 = self.stage5(self.down4(x4)) # (B,512,14,14)

        return x1, x2, x3, x4, x5


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def _check_tensor(self, name, x):
        if x is None:
            raise RuntimeError(f"[MambaVisionMixer] {name} is None")
        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"[MambaVisionMixer] Non-finite values in {name}: "
                f"min={x.min().item()}, max={x.max().item()}"
            )
    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"[MambaVisionMixer] Expected (B, L, D), got {hidden_states.shape}"
            )
        
        B, seqlen, D = hidden_states.shape
        if D != self.d_model:
            raise RuntimeError(
                f"[MambaVisionMixer] d_model mismatch: got {D}, expected {self.d_model}"
            )

        self._check_tensor("hidden_states", hidden_states)

        # xz = self.in_proj(hidden_states)
        # xz = rearrange(xz, "b l d -> b d l")
        # x, z = xz.chunk(2, dim=1)
        # A = -torch.exp(self.A_log.float())
        # x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        # z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        # dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # y = selective_scan_fn(x, 
        #                       dt, 
        #                       A, 
        #                       B, 
        #                       C, 
        #                       self.D.float(), 
        #                       z=None, 
        #                       delta_bias=self.dt_proj.bias.float(), 
        #                       delta_softplus=True, 
        #                       return_last_state=None)
        
        # y = torch.cat([y, z], dim=1)
        # y = rearrange(y, "b d l -> b l d")
        # out = self.out_proj(y)
        # return out
        try:
            xz = self.in_proj(hidden_states)          # (B, L, d_inner)
            self._check_tensor("xz", xz)

            xz = rearrange(xz, "b l d -> b d l")      # (B, d_inner, L)
            x, z = xz.chunk(2, dim=1)                # each (B, d_inner/2, L)
            self._check_tensor("x_before_conv", x)
            self._check_tensor("z_before_conv", z)

            A = -torch.exp(self.A_log.float())        # (d_inner/2, d_state)

            x = F.silu(
                F.conv1d(
                    input=x,
                    weight=self.conv1d_x.weight,
                    bias=self.conv1d_x.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )
            z = F.silu(
                F.conv1d(
                    input=z,
                    weight=self.conv1d_z.weight,
                    bias=self.conv1d_z.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )

            self._check_tensor("x_after_conv", x)
            self._check_tensor("z_after_conv", z)

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            self._check_tensor("x_dbl", x_dbl)

            dt, Bmat, Cmat = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
            Bmat = rearrange(Bmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            Cmat = rearrange(Cmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            self._check_tensor("dt", dt)
            self._check_tensor("Bmat", Bmat)
            self._check_tensor("Cmat", Cmat)

            y = selective_scan_fn(
                x,
                dt,
                A,
                Bmat,
                Cmat,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )

            self._check_tensor("y_after_scan", y)

            y = torch.cat([y, z], dim=1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            self._check_tensor("out", out)
            return out

        except RuntimeError as e:
            # Catch low-level CUDA assert and re-raise with context
            if "device-side assert" in str(e).lower():
                raise RuntimeError(
                    "[MambaVisionMixer] CUDA device-side assert inside selective_scan_fn "
                    f"or conv1d.\n"
                    f"hidden_states shape: {hidden_states.shape}, "
                    f"d_model={self.d_model}, d_inner={self.d_inner}, "
                    f"d_state={self.d_state}, seq_len={seqlen}"
                ) from e
            else:
                raise


class LViT_MT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.text_encoder = ClinicalTextEncoder()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        # self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)

        # self.encoder = MambaEncoder2D(in_ch=n_channels)
        self.encoder = MambaEncoder2D(in_ch=n_channels, base_dim=in_channels)


        # self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        # self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        # self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        # self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        print("LViT model with Transformer-Mamba Block as Encoder created successfully!")

    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        print("Input image x :", x.shape)   # (B,3,224,224)

        text_emb = self.text_encoder(text)      # (B, 768)
        print("Text embedding :", text_emb.shape)   # (B,768)
        # text = text_emb.unsqueeze(1)            # (B, 1, 768)
        text = text_emb.unsqueeze(1).repeat(1, 10, 1)   # (B, 10, 768)
        print("Expanded text  :", text.shape)       # (B,10,768)

        # x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        # print(f"x1 after initial conv       : {x1.shape}")

        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        print(f"text4 (for deepest ViT)     : {text4.shape}") 
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        print(f"text3                       : {text3.shape}")
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        print(f"text2                       : {text2.shape}")
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        print(f"text1 (for shallow ViT)     : {text1.shape}")

        x1, x2, x3, x4, x5 = self.encoder(x)

        print("x1 (Mamba stage1):", x1.shape)
        print("x2 (Mamba stage2):", x2.shape)
        print("x3 (Mamba stage3):", x3.shape)
        print("x4 (Mamba stage4):", x4.shape)
        print("x5 (Mamba bottleneck):", x5.shape)

        y1 = self.downVit(x1, x1, text1)
        print(f"y1 (DownViT stage1)         : {y1.shape}")
        # x2 = self.down1(x1)
        # print(f"x2 (CNN down1)              : {x2.shape}")

        y2 = self.downVit1(x2, y1, text2)
        print(f"y2 (DownViT stage2)         : {y2.shape}")
        # x3 = self.down2(x2)
        # print(f"x3 (CNN down2)              : {x3.shape}")

        y3 = self.downVit2(x3, y2, text3)
        print(f"y3 (DownViT stage3)         : {y3.shape}")
        # x4 = self.down3(x3)
        # print(f"x4 (CNN down3)              : {x4.shape}")

        y4 = self.downVit3(x4, y3, text4)
        print(f"y4 (DownViT stage4)         : {y4.shape}")
        # x5 = self.down4(x4)
        # print(f"x5 (CNN bottleneck)         : {x5.shape}")

        y4 = self.upVit3(y4, y4, text4, True)
        print(f"y4 (UpViT stage4)           : {y4.shape}")
        y3 = self.upVit2(y3, y4, text3, True)
        print(f"y3 (UpViT stage3)           : {y3.shape}")
        y2 = self.upVit1(y2, y3, text2, True)
        print(f"y2 (UpViT stage2)           : {y2.shape}")
        y1 = self.upVit(y1, y2, text1, True)
        print(f"y1 (UpViT stage1)           : {y1.shape}")


        x1 = self.reconstruct1(y1) + x1
        print(f"x1 after reconstruct fusion : {x1.shape}")

        x2 = self.reconstruct2(y2) + x2
        print(f"x2 after reconstruct fusion : {x2.shape}")

        x3 = self.reconstruct3(y3) + x3
        print(f"x3 after reconstruct fusion : {x3.shape}")

        x4 = self.reconstruct4(y4) + x4
        print(f"x4 after reconstruct fusion : {x4.shape}")

        x = self.up4(x5, x4)
        print(f"Decoder up4 output          : {x.shape}")
        x = self.up3(x, x3)
        print(f"Decoder up3 output          : {x.shape}")
        x = self.up2(x, x2)
        print(f"Decoder up2 output          : {x.shape}")
        x = self.up1(x, x1)
        print(f"Decoder up1 output          : {x.shape}")

        
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits

    # def forward(self, x, text):
        # x = x.float()  # x [4,3,224,224]

        # text_emb = self.text_encoder(text)      # (B, 768)
        # # text = text_emb.unsqueeze(1)            # (B, 1, 768)
        # text = text_emb.unsqueeze(1).repeat(1, 10, 1)   # (B, 10, 768)

        # x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        # text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        # text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        # text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        # text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        # y1 = self.downVit(x1, x1, text1)
        # x2 = self.down1(x1)
        # y2 = self.downVit1(x2, y1, text2)
        # x3 = self.down2(x2)
        # y3 = self.downVit2(x3, y2, text3)
        # x4 = self.down3(x3)
        # y4 = self.downVit3(x4, y3, text4)
        # x5 = self.down4(x4)
        # y4 = self.upVit3(y4, y4, text4, True)
        # y3 = self.upVit2(y3, y4, text3, True)
        # y2 = self.upVit1(y2, y3, text2, True)
        # y1 = self.upVit(y1, y2, text1, True)
        # x1 = self.reconstruct1(y1) + x1
        # x2 = self.reconstruct2(y2) + x2
        # x3 = self.reconstruct3(y3) + x3
        # x4 = self.reconstruct4(y4) + x4
        # x = self.up4(x5, x4)
        # x = self.up3(x, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)
        # if self.n_classes == 1:
        #     logits = self.last_activation(self.outc(x))
        # else:
        #     logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        # return logits
