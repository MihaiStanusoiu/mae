import torch.nn as nn
from timm.layers import Mlp, DropPath, use_fused_attn
import torch.nn.functional as F
import torch


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            output_fmt=None,
            bias=True,
            strict_img_size=True,
            dynamic_img_pad=False,
    ):
        super().__init__()
        self.patch_size = (16, 16)
        if img_size is not None:
            self.img_size = (224, 224)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        print(x.shape)
        if self.flatten:
            # NCHW -> NLC
            # x = x.flatten(2).transpose(1, 2)
            x = x.flatten(2)
            print(x.shape)
            x = x.transpose(1, 2)
            print(x.shape)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        print('***** attention starts *****')
        B, N, C = x.shape
        print('-- 1. preparation:')
        print('x.shape:', x.shape)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        print('qkv.shape:', qkv.shape)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        print('qkv.shape:', qkv.shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        print('qkv.shape:', qkv.shape)

        q, k, v = qkv.unbind(0)
        print('q.shape, k.shape, v.shape:', q.shape, k.shape, v.shape)
        q, k = self.q_norm(q), self.k_norm(k)

        print('-- 2. computation:')
        q = q * self.scale
        print('q.shape:', q.shape)
        k_transpose = k.transpose(-2, -1)
        print('k_transpose.shape:', k_transpose.shape)
        attn = q @ k_transpose
        print('q @ k_transpose.shape:', attn.shape)
        attn = attn.softmax(dim=-1)
        print('attn.shape (after softmax):', attn.shape)
        attn = self.attn_drop(attn)
        print('attn.shape (after attn_drop):', attn.shape)
        print('v.shape:', v.shape)
        x = attn @ v
        print('x.shape=attn@v.shape:', x.shape)

        print('-- 3. post process:')
        x = x.transpose(1, 2)
        print('x.shape:', x.shape)
        x = x.reshape(B, N, C)
        print('x.shape:', x.shape)
        x = self.proj(x)
        print('x.shape:', x.shape)
        x = self.proj_drop(x)
        print('x.shape:', x.shape)
        print('***** attention ends *****')
        print()
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x