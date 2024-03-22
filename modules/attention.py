from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SelfCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
class TimeFreqCoTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout, gated_ff=True):
        super().__init__()
        
        self.t_self_attn = SelfCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.t_ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.f_self_attn = SelfCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout) 
        self.f_ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, f, t = x.shape        # [b, c, f, t]
        
        x = x.transpose(1, -1)      # [b, t, f, c]
        for i in range(f):
            x_part_pre = x[:, :, i, :]  # [b, t, c]
            x_part = x_part_pre.clone()
            x_part = self.norm1(x_part)
            x_part = self.t_self_attn(x_part) + x_part
            x_part = self.norm2(x_part)
            x_part = self.t_ff(x_part)
            x[:, :, i, :] = x_part
            
        x = x.transpose(1, 2)       # [b, f, t, c]
        for i in range(t):
            x_part_pre = x[:, :, i, :]  # [b, t, c]
            x_part = x_part_pre.clone()
            x_part = self.norm3(x_part)
            x_part = self.f_self_attn(x_part) + x_part
            x_part = self.norm4(x_part)
            x_part = self.f_ff(x_part)
            x[:, :, i, :] = x_part
        x = x.transpose(1, 2).transpose(1, -1)   
        return x
    
class TimeFreqCoTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [TimeFreqCoTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        
    def forward(self, x):
        b, c, f, t = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.proj_out(x)
        return x + x_in
        
            
        
                
        
        
        
