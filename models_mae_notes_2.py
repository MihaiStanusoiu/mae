# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from auxliary import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            # for timm==0.3.2
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            # for newest timm version
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            # for timm==0.3.2
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            # for newest timm version
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        print('***** patchify starts *****')
        print('imgs.shape:', imgs.shape)
        p = self.patch_embed.patch_size[0]
        print('p:', p)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        print('h, w:', h, w)
        # x = imgs.reshape(shape=(imgs.shape[0], 3, 14, 16, 14, 16))
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        print('x.shape:', x.shape)
        x = torch.einsum('nchpwq->nhwpqc', x)
        print('x.shape:', x.shape)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        print('x.shape:', x.shape)
        print('***** patchify ends *****')
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # get ids to restore original sequence
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_kept, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        print('------------- start encoder -------------')
        print('- input image:', x.shape)

        print('\n==> embed patches')
        # embed patches
        x = self.patch_embed(x)
        print('x.shape:', x.shape)

        print('\n==> add position embedding')
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        print('x.shape:', x.shape)

        print('\n==> mask patches')
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        print('x.shape:', x.shape)
        print('mask.shape:', mask.shape)
        print('ids_restore.shape:', ids_restore.shape)

        print('\n==> append class token')
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print('x.shape:', x.shape)

        # apply Transformer blocks
        print('\n==> apply transformer blocks')
        for blk in self.blocks:
            x = blk(x)
            print('x.shape:', x.shape)
        x = self.norm(x)

        print('------------- end encoder -------------')
        print('x.shape:', x.shape)
        print('mask.shape:', mask.shape)
        print('ids_restore.shape:', ids_restore.shape)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        print('\n------------- start decoder -------------')
        print('x.shape:', x.shape)
        print('ids_restore.shape:', ids_restore.shape)

        print('\n==> embed tokens')
        # embed tokens
        x = self.decoder_embed(x)
        print('x.shape:', x.shape)

        print('\n==> append mask tokens to sequence')
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 196-49=147
        print('ids_restore.shape:', ids_restore.shape)
        print('mask_tokens.shape:', mask_tokens.shape)
        print('x[:, 1:, :].shape:', x[:, 1:, :].shape)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        print('x_.shape:', x_.shape)
        ids_restore_1 = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        print('ids_restore_1.shape:', ids_restore_1.shape)
        print(ids_restore_1)
        x_ = torch.gather(x_, dim=1, index=ids_restore_1)  # unshuffle
        print('x_.shape:', x_.shape)

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        print('x.shape:', x.shape)

        print('\n==> add position embedding')
        # add pos embed
        x = x + self.decoder_pos_embed
        print('x.shape:', x.shape)

        print('\n==> apply transformer blocks')
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            print('x.shape:', x.shape)
        x = self.decoder_norm(x)

        print('\n==> predictor projection')
        # predictor projection
        x = self.decoder_pred(x)
        print('x.shape:', x.shape)

        print('\n==> predictor projection')
        # remove cls token
        x = x[:, 1:, :]
        print('x.shape:', x.shape)

        print('------------- end decoder -------------')
        print('x.shape:', x.shape)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        print('\n------------- start loss function -------------')
        print('imgs.shape:', imgs.shape)
        print('pred.shape:', pred.shape)
        print('mask.shape:', mask.shape)

        print('\n==> put original images into patches')
        target = self.patchify(imgs)
        print('target.shape:', target.shape)

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        print('\n==> calculate loss function')
        loss = (pred - target) ** 2
        print('loss.shape:', loss.shape)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        print('loss.shape:', loss.shape)
        print('mask:', mask)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        print('loss:', loss)
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    #could do embed_dim = 20, depth = 1
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=256, depth=6, num_heads=8,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    x = torch.randn([1, 3, 224, 224])
    model(x, 0.75)

    # print('-- model:')
    # print(model)

    print('\n************************')
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis


    flops = FlopCountAnalysis(model, x)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, x)

    # print(f"total flops : {flops.total()}")
    # # print(f"total activations: {acts.total()}")
    # print(f"number of parameter: {param}")
    # flops.set_op_handle("aten::add", torch.jit.)
    flops = flops.total()
    Gflops = flops / 1e9
    Mparam = param / 1e6

    print(f"total Gflops : {Gflops} G")
    # print(f"total activations: {acts.total()}")
    print(f"number of parameter: {Mparam} M")

