# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

#from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

logger = logging.getLogger("dinov2")

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        # import pdb;pdb.set_trace()
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU
    #import numpy.bool
    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha
    from xformers.components.attention import ScaledDotProduct
    from xformers.components import MultiHeadDispatch
    #import numpy.bool
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #if not self.training:
        #
        # self.attn = ScaledDotProduct()
            #self.attn = MultiHeadDispatch(dim_model=EMB, residual_dropout=DROPOUT, num_heads=HEADS, attention=attn)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        if attn_bias is not None:
            attn = attn + attn_bias[:, :, :N]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
        #if True:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, attn_bias)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        if attn_bias is not None:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias[:, :, :N])
        else:
            x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat
    #import numpy.bool
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        def attn_residual_func(x: Tensor, attn_bias) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), attn_bias))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                attn_bias=attn_bias
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, attn_bias))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, attn_bias)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0, attn_bias=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset, attn_bias)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, attn_bias=None):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, attn_bias)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, others=None):
        for b in self:
            if others == None:
                x = b(x)
            else:
                x = b(x, others)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        #init_values=None,  # for layerscale: None or 0 => no layerscale
        init_values=1e-5,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        window_size=37,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        B, C, H, W = x.size()
        pad_h = (self.patch_size - H % self.patch_size)
        pad_w = (self.patch_size - W % self.patch_size)
        if pad_h == self.patch_size:
            pad_h = 0
        if pad_w == self.patch_size:
            pad_w = 0     
        #x = nn.functional.pad(x, (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2))
        if pad_h + pad_w > 0:
            x = torch.nn.functional.interpolate(x, (H+pad_h, W+pad_w), mode='bilinear')

        x = self.prepare_tokens_with_masks(x, masks)

        features = []
        for blk in self.blocks:
            x = blk(x)
        # for idx in range(len(self.blocks[0])):
        #     x = self.blocks[0][idx](x)
        #     if (idx + 1) % (len(self.blocks[0]) // 4) == 0:
        #         features.append(x)

        #return [features, (B, (H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size, H, W)]

        x_norm = self.norm(x)
        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }
        features = []
        features.append(x_norm)
        features.append(x_norm)
        features.append(x_norm)
        features.append(x_norm)
        return [features, (B, (H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size, H, W)]

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
        # if is_training:
        #     return ret
        # else:
        #     return self.head(ret["x_norm_clstoken"])


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 37, stride, 18, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    #def no_weight_decay(self):
        #return ['proj.%d.weight' % i for i in range(4)]


class DinoWindowVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        #init_values=None,  # for layerscale: None or 0 => no layerscale
        init_values=1e-5,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=NestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        window_size=7,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.pos_conv = PosConv(self.embed_dim, self.embed_dim)

        self.window_size = window_size
        #self.conv_block = nn.ModuleList([ConvBlock(embed_dim) for i in range(4)])
        #self.conv_block = nn.ModuleList([nn.Identity() for i in range(4)])

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.nh = -1
        self.nw = -1
        try:
            H = cfg.data_basic['crop_size'][0] 
            W = cfg.data_basic['crop_size'][1] 
            pad_h = (self.patch_size - H % self.patch_size)
            pad_w = (self.patch_size - W % self.patch_size)
            if pad_h == self.patch_size:
                pad_h = 0
            if pad_w == self.patch_size:
                pad_w = 0   
            self.nh = (H + pad_h) // self.patch_size
            self.nw = (W + pad_w) // self.patch_size
            self.prepare_attn_bias((self.nh, self.nw))
        except:
            pass
        self.init_weights()

        self.total_step = 10000 # For PE -> GPE transfer
        self.start_step = 2000
        self.current_step = 20000

    def init_weights(self):
        #trunc_normal_(self.pos_embed, std=0.02)
        #nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)
        for i in range(4):
            try:
                nn.init.constant_(self.conv_block[i].conv2.weight, 0.0)
            except:
                pass

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        #npatch = x.shape[1] - 1
        #N = self.pos_embed.shape[1] - 1
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        #class_pos_embed = pos_embed[:, 0]
        #patch_pos_embed = pos_embed[:, 1:]
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)
        #return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def window_partition(self, x: torch.Tensor, window_size: int, hw: Tuple[int, int], conv_feature=False) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        if conv_feature == False:
            B, N, C = x.shape
            H, W = hw[0], hw[1]

            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        else:
            B, C, H, W = x.shape

            x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)

            windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)            

        #y = torch.cat((x_cls, windows), dim=1)
        return windows   #, (Hp, Wp)


    def window_unpartition(self, 
        windows: torch.Tensor, window_size: int, hw: Tuple[int, int], conv_feature=False
    ) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
            window_size (int): window size.
            pad_hw (Tuple): padded height and width (Hp, Wp).
            hw (Tuple): original height and width (H, W) before padding.

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        H, W = hw

        B = windows.shape[0] // (H * W // window_size // window_size)
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)

        if conv_feature == False:
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp * Wp, -1)
        else:
            C = windows.shape[-1]
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)

        # if Hp > H or Wp > W:
        #     x = x[:, :H, :W, :].contiguous()
        return x

    def prepare_tokens_with_masks(self, x, masks=None, step=-1):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        #x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if step == -1:
            step = self.current_step
        else:
            self.current_step = step
        
        if step < self.start_step:
            coef = 0.0
        elif step < self.total_step:
            coef = (step - self.start_step) / (self.total_step - self.start_step) 
        else:
            coef = 1.0
        
        x = x + (1 - coef) * self.interpolate_pos_encoding(x, w, h) + coef * self.pos_conv(x, (self.nh, self.nw))

        return x

    def prepare_attn_bias(self, shape):
        window_size = self.window_size
        if window_size <= 0:
            return
        
        import xformers.components.attention.attention_patterns as AP
        
        nh, nw = shape
        radius = (window_size-1)//2 
        mask_ori = AP.local_2d_pattern(nh, nw, distance = radius + 0.1, p=torch.inf).cuda()
        
        pad = (8 - (nh * nw) % 8)
        if pad == 8:
            pad = 0
        mask_pad = nn.functional.pad(mask_ori, (0, pad)).contiguous()
        if pad > 0:
            mask = mask_pad[:, :-pad].view(nh, nw, nh, nw)
        else:
            mask = mask_pad[:, :].view(nh, nw, nh, nw)
        
        # angle
        mask[:radius+1,  :radius+1,  :window_size,  :window_size] = True
        mask[:radius+1,  -radius-1:, :window_size,  -window_size:] = True
        mask[-radius-1:, :radius+1,  -window_size:, :window_size] = True
        mask[-radius-1:, -radius-1:, -window_size:, -window_size:] = True

        # edge
        mask[radius+1:-radius-1,  :radius+1,  :,  :] = mask[radius+1:-radius-1,  radius:radius+1,    :,  :]
        mask[radius+1:-radius-1,  -radius-1:, :,  :] = mask[radius+1:-radius-1,  -radius-1:-radius,  :,  :]
        mask[:radius+1,   radius+1:-radius-1, :,  :] = mask[radius:radius+1,   radius+1:-radius-1,   :,  :]
        mask[-radius-1:,  radius+1:-radius-1, :,  :] = mask[-radius-1:-radius, radius+1:-radius-1,   :,  :]

        mask = mask.view(nh*nw, nh*nw)
        bias_pad = torch.log(mask_pad)
        #bias = bias_pad[:, :-pad]
        self.register_buffer('attn_bias', bias_pad)

        return bias_pad

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, **kwargs):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        B, C, H, W = x.size()
        pad_h = (self.patch_size - H % self.patch_size)
        pad_w = (self.patch_size - W % self.patch_size)
        if pad_h == self.patch_size:
            pad_h = 0
        if pad_w == self.patch_size:
            pad_w = 0     
        #x = nn.functional.pad(x, (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2))
        if pad_h + pad_w > 0:
            x = torch.nn.functional.interpolate(x, (H+pad_h, W+pad_w), mode='bilinear')
        
        nh = (H+pad_h)//self.patch_size
        nw = (W+pad_w)//self.patch_size

        if self.window_size > 0:
            if nh == self.nh and nw == self.nw:
                attn_bias = self.attn_bias
            else:
                attn_bias = self.prepare_attn_bias(((H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size))   
                self.nh = nh
                self.nw = nw 
            attn_bias = attn_bias.unsqueeze(0).repeat(B * self.num_heads, 1, 1)
        else:
            attn_bias = None

        x = self.prepare_tokens_with_masks(x, masks)
        #x = self.patch_embed(x)

        features = []
        #x = self.window_partition(x, self.window_size, (H // self.patch_size, W // self.patch_size))
        for blk in self.blocks:
            x = blk(x, attn_bias)
        #x = self.window_unpartition(x, self.window_size, (H // self.patch_size, W // self.patch_size))

        # for idx in range(len(self.blocks[0])):
        #     x = self.blocks[0][idx](x, attn_bias)

        #     if (idx + 1) % (len(self.blocks[0]) // 4) == 0:
        #         x = self.window_unpartition(x, self.window_size, (H // self.patch_size, W // self.patch_size), conv_feature=True)
        #         x = self.conv_block[idx // (len(self.blocks[0]) // 4)](x)
        #         if idx + 1 != len(self.blocks[0]):
        #             x = self.window_partition(x, self.window_size, (H // self.patch_size, W // self.patch_size), conv_feature=True)
        #         else:
        #             b, c, h, w = x.size()
        #             x = x.permute(0, 2, 3, 1).contiguous().view(b, h, w, c)
                #features.append(x)

        #return [features, (B, (H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size, H, W)]

        x_norm = self.norm(x)
        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_patchtokens": x_norm[:, 1:],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }
        features = []
        features.append(x_norm)
        features.append(x_norm)
        features.append(x_norm)
        features.append(x_norm)
        return [features, (B, (H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size, H, W)]

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
        # if is_training:
        #     return ret
        # else:
        #     return self.head(ret["x_norm_clstoken"])



def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=14, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=14, **kwargs):
    model = DinoWindowVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=14, checkpoint=None, **kwargs):
    model = DinoVisionTransformer(
        img_size = 518,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        **kwargs,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'blocks' in key:
                    key_new = 'blocks.0' + key[len('blocks'):]
                else:
                    key_new = key
                new_state_dict[key_new] = value

            model.load_state_dict(new_state_dict, strict=True)
        #del model.norm
        if 'del_mask_token' in kwargs and kwargs['del_mask_token'] == True:
            del model.mask_token
    return model


def vit_giant2(patch_size=14, checkpoint=None, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        img_size = 518,
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        # block_fn=partial(Block, attn_class=MemEffAttention),
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        **kwargs,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'blocks' in key:
                    key_new = 'blocks.0' + key[len('blocks'):]
                else:
                    key_new = key
                new_state_dict[key_new] = value

            import pdb;pdb.set_trace()
            model.load_state_dict(new_state_dict, strict=True)
        #del model.norm
        del model.mask_token

    return model


if __name__ == '__main__':
    try:
        from mmcv.utils import Config
    except:
        from mmengine import Config    
    
    #rgb = torch.rand((2, 3, 518, 518)).cuda()

    #cfg.data_basic['crop_size']['0'] 
    #cfg.data_basic['crop_size']['1'] 
    cfg = Config.fromfile('mu.hu/monodepth/mono/configs/HourglassDecoder/pub12.convlarge.0.3_150.py')

    #rgb = torch.arange(0, 2*3*1036*1036, 1).cuda().float().view(2, 3, 1036, 1036)
    rgb = torch.zeros(1, 3, 1400, 1680).cuda()
    model = vit_large(checkpoint="pretrained_weight_repo/vit/dinov2_vitl14_pretrain.pth", kwarg=cfg).cuda()

    #import timm
    #model2 = timm.models.vision_transformer.vit_large_patch14_dinov2().cuda()
    #timm.models.load_checkpoint(model2, '/cpfs02/shared/public/yvan/pretrained_weight_repo/vit/dinov2_vitl14_pretrain.pth', filter_fn=timm.models.vision_transformer.checkpoint_filter_fn)

    out1 = model(rgb)
    #out2 = model2(rgb)
    temp = 0



# import time
# window_size = 37
# def prepare_window_masks(shape):
#     if window_size <= 0:
#         return None
#     import xformers.components.attention.attention_patterns as AP
    
#     B, nh, nw, _, _ = shape
#     radius = (window_size-1)//2 
#     #time0 = time.time()
#     d = AP.local_nd_distance(nh, nw, distance = radius + 0.1, p=torch.inf).cuda()
#     #mask = AP.local_2d_pattern(nh, nw, distance = radius + 0.1, p=torch.inf).cuda()
#     # mask = mask.view(nh, nw, nh, nw)
#     # #time1 = time.time() - time0
    
#     # # angle
#     # mask[:radius+1,  :radius+1,  :window_size,  :window_size] = True
#     # mask[:radius+1,  -radius-1:, :window_size,  -window_size:] = True
#     # mask[-radius-1:, :radius+1,  -window_size:, :window_size] = True
#     # mask[-radius-1:, -radius-1:, -window_size:, -window_size:] = True
#     # time2 = time.time() - time0 - time1

#     # # edge
#     # mask[radius+1:-radius-1,  :radius+1,  :,  :] = mask[radius+1:-radius-1,  radius:radius+1,    :,  :]
#     # mask[radius+1:-radius-1,  -radius-1:, :,  :] = mask[radius+1:-radius-1,  -radius-1:-radius,  :,  :]
#     # mask[:radius+1,   radius+1:-radius-1, :,  :] = mask[radius:radius+1,   radius+1:-radius-1,   :,  :]
#     # mask[-radius-1:,  radius+1:-radius-1, :,  :] = mask[-radius-1:-radius, radius+1:-radius-1,   :,  :]
#     # time3 = time.time() - time0 - time2
#     # print(time1, time2, time3)

# #     return mask.view(nw*nw, nh*nw).unsqueeze(0).repeat(B, 1)   

# shape = (1, 55, 55, None, None)
# mask = prepare_window_masks(shape)
# # temp = 1