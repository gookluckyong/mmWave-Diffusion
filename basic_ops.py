"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000): #这段函数 timestep_embedding 的作用是：为时间步（timesteps）生成正弦位置编码（Sinusoidal Positional Embeddings），常用于扩散模型（Diffusion Models）、Transformers 等需要表达时间或位置信息的神经网络中。
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. timesteps: 一个一维 Tensor(形状 [N]），表示每个样本的时间步（可以是整数，也可以是小数，比如 torch.linspace(0, T, steps)）
                      These may be fractional.
    :param dim: the dimension of the output. dim: 嵌入的维度(必须为偶数或会在最后补0)
    :param max_period: controls the minimum frequency of the embeddings. max_period: 控制最低频率，影响周期范围
    :return: an [N x dim] Tensor of positional embeddings. 
    """
    half = dim // 2  #先取一半的维度用于 cos 和 sin
    freqs = th.exp( #构造一个指数下降的频率列表   实际上 freqs[i] = 1 / (max_period ** (i / half))，从高频到低频变化 它与 Transformer 中的频率分布一样，是为了让不同维度拥有不同周期的振荡
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half timesteps[:, None] 将 [N] 扩展为 [N, 1]，准备广播 freqs[None] 扩展为 [1, half] 广播后得到 [N, half] 的矩阵，每一行是当前时间步与频率向量的乘积
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1) #cos(args) 和 sin(args) 都是 [N, half]  拼接后得到 [N, dim] 的张量
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)  #如果 dim 是奇数，会在最后补一个 0，让嵌入维度达到 dim
    return embedding  #返回的是形状为 [N, dim] 的嵌入张量，其中每行是对应 timestep 的位置嵌入

