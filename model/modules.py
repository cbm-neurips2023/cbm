import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import AttrDict


class Attention(nn.Module):
    def __init__(self, attention_dim, num_queries, query_dim, num_keys, key_dim, out_dim=None, use_bias=False):
        super(Attention, self).__init__()
        self.temperature = np.sqrt(attention_dim)
        self.use_bias = use_bias

        if out_dim is None:
            out_dim = attention_dim

        b = 1 / self.temperature
        b_v = 1 / np.sqrt(out_dim)
        self.query_weight = nn.Parameter(torch.FloatTensor(num_queries, query_dim, attention_dim).uniform_(-b, b))
        self.query_bias = nn.Parameter(torch.zeros(num_queries, 1, attention_dim))
        self.key_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, attention_dim).uniform_(-b, b))
        self.key_bias = nn.Parameter(torch.zeros(num_keys, 1, attention_dim))
        self.value_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, out_dim).uniform_(-b_v, b_v))
        self.value_bias = nn.Parameter(torch.zeros(num_keys, 1, out_dim))

    def forward_score(self, q, k):
        """
        :param q: (num_queries, bs, query_dim)
        :param k: (num_keys, bs, key_dim)
        :return: logits (bs, num_queries, num_keys)
        """
        query = torch.bmm(q, self.query_weight)                         # (num_queries, bs, attention_dim)
        key = torch.bmm(k, self.key_weight)                             # (num_keys, bs, attention_dim)
        if self.use_bias:
            query += self.query_bias                                    # (num_queries, bs, attention_dim)
            key += self.key_bias                                        # (num_keys, bs, attention_dim)

        query = query.permute(1, 0, 2)                                  # (bs, num_queries, attention_dim)
        key = key.permute(1, 2, 0)                                      # (bs, attention_dim, num_keys)

        logits = torch.bmm(query, key) / self.temperature               # (bs, num_queries, num_keys)
        return logits

    def forward(self, q, k, return_logits=False, gumbel_select=False, tau=1.0):
        """
        :param q: (num_queries, bs, query_dim)
        :param k: (num_keys, bs, key_dim)
        :return:
        """
        value = torch.bmm(k, self.value_weight)                         # (num_keys, bs, attention_value_dim)
        if self.use_bias:
            value += self.value_bias                                    # (num_keys, bs, attention_value_dim)

        logits = self.forward_score(q, k)                               # (bs, num_queries, num_keys)

        if gumbel_select:
            attn = F.gumbel_softmax(logits, dim=-1, hard=True, tau=tau) # (bs, num_queries, num_keys)
        else:
            attn = F.softmax(logits, dim=-1)                            # (bs, num_queries, num_keys)

        attn = attn.permute(1, 2, 0).unsqueeze(dim=-1)                  # (num_queries, num_keys, bs, 1)

        output = (value * attn).sum(dim=1)                              # (num_queries, bs, attention_value_dim)

        if return_logits:
            return output, logits
        else:
            return output


class MHAttention(nn.Module):
    def __init__(self, attention_dim, num_heads, num_queries, query_dim, num_keys, key_dim,
                 out_dim=None, use_bias=False):
        super(MHAttention, self).__init__()

        if out_dim is None:
            out_dim = attention_dim
        self.temperature = np.sqrt(attention_dim)
        self.use_bias = use_bias

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.embed_dim = embed_dim = num_heads * attention_dim

        b = 1 / self.temperature
        b_p = 1 / np.sqrt(out_dim)
        self.query_weight = nn.Parameter(torch.FloatTensor(num_queries, query_dim, embed_dim).uniform_(-b, b))
        self.query_bias = nn.Parameter(torch.zeros(num_queries, 1, embed_dim))
        self.key_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, embed_dim).uniform_(-b, b))
        self.key_bias = nn.Parameter(torch.zeros(num_keys, 1, embed_dim))
        self.value_weight = nn.Parameter(torch.FloatTensor(num_keys, key_dim, embed_dim).uniform_(-b, b))
        self.value_bias = nn.Parameter(torch.zeros(num_keys, 1, embed_dim))
        self.proj_weight = nn.Parameter(torch.FloatTensor(num_queries, embed_dim, out_dim).uniform_(-b_p, b_p))
        self.proj_bias = nn.Parameter(torch.zeros(num_queries, 1, out_dim))

    def forward(self, q, k, return_logits=False, gumbel_select=False, tau=1.0):
        """
        :param q: (num_queries, bs, query_dim)
        :param k: (num_keys, bs, key_dim)
        :return:
        """
        bs = q.shape[1]
        attention_dim = self.attention_dim
        num_heads = self.num_heads
        embed_dim = self.embed_dim
        num_queries = self.num_queries
        num_keys = self.num_keys

        query = torch.bmm(q, self.query_weight)                         # (num_queries, bs, embed_dim)
        key = torch.bmm(k, self.key_weight)                             # (num_keys, bs, embed_dim)
        if self.use_bias:
            query += self.query_bias                                    # (num_queries, bs, embed_dim)
            key += self.key_bias                                        # (num_keys, bs, embed_dim)

        query = query.view(num_queries, bs * num_heads, attention_dim)  # (num_queries, bs * num_heads, attention_dim)
        key = key.view(num_keys, bs * num_heads, attention_dim)         # (num_keys, bs * num_heads, attention_dim)

        query = query.permute(1, 0, 2)                                  # (bs * num_heads, num_queries, attention_dim)
        key = key.permute(1, 2, 0)                                      # (bs * num_heads, attention_dim, num_keys)

        logits = torch.bmm(query, key) / self.temperature               # (bs * num_heads, num_queries, num_keys)

        if gumbel_select:
            attn = F.gumbel_softmax(logits, dim=-1, hard=True, tau=tau) # (bs * num_heads, num_queries, num_keys)
        else:
            attn = F.softmax(logits, dim=-1)                            # (bs * num_heads, num_queries, num_keys)
        attn = attn.permute(1, 2, 0).unsqueeze(dim=-1)                  # (num_queries, num_keys, bs * num_heads, 1)

        value = torch.bmm(k, self.value_weight)                         # (num_keys, bs, embed_dim)
        if self.use_bias:
            value += self.value_bias                                    # (num_keys, bs, embed_dim)
        value = value.view(num_keys, bs * num_heads, attention_dim)     # (num_keys, bs * num_heads, attention_dim)

        output = (value * attn).sum(dim=1)                              # (num_queries, bs * num_heads, attention_dim)
        output = output.view(num_queries, bs, embed_dim)                # (num_queries, bs, embed_dim)
        output = torch.bmm(output, self.proj_weight)                    # (num_queries, bs, out_dim)
        if self.use_bias:
            output += self.proj_bias                                    # (num_queries, bs, out_dim)

        if not return_logits:
            return output

        logits = logits.view(bs, num_heads, num_queries, num_keys)
        return output, logits

class Residual(nn.Module):
    # convolution residual block
    def __init__(self, channel, kernel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel, padding=kernel // 2)
        self.scalar = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        out = self.conv(F.leaky_relu(x, negative_slope=0.02))
        return x + out * self.scalar


class Reshape(nn.Module):
    # reshape last dim to (c, h, w)
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.dim = np.prod(self.shape)

    def forward(self, x):
        assert x.shape[-1] == self.dim
        return x.view(*x.shape[:-1], *self.shape)


def get_backbone(params, encoding, verbose=False):
    encoder_params = params.encoder_params
    decoder_params = params.decoder_params

    image_spec = np.concatenate([obs for obs in params.obs_spec.values() if obs.ndim == 3], axis=0).shape

    c = h = w = dim = None
    if encoding:
        c, h, w = image_spec
        assert h == w
    else:
        dim = params.encoder_params.feature_dim

    def get_shape():
        if encoding:
            if dim is None:
                shape = (c, h, w)
            else:
                shape = dim
        else:
            if c is None:
                shape = dim
            else:
                shape = (c, h, w)
        return shape

    module_list = []
    modules = encoder_params.modules if encoding else decoder_params.modules
    for i, mod_params in enumerate(modules):
        mod_params = AttrDict(mod_params)
        module_type = mod_params.type
        mod_params.pop("type")

        if verbose:
            if i == 0:
                print("encoder" if encoding else "decoder")
            print("{}-th module:".format(i + 1), module_type, mod_params)
            print("input shape:", get_shape())

        if module_type == "conv":
            if mod_params.channel is None:
                assert not encoding and i == len(modules) - 1
                mod_params.channel = image_spec[0]
            module = nn.Conv2d(c, mod_params.channel, mod_params.kernel, mod_params.stride, mod_params.kernel // 2)
            w = w // mod_params.stride
            h = h // mod_params.stride
            c = mod_params.channel
        elif module_type == "residual":
            module = Residual(c, mod_params.kernel)
        elif module_type == "avg_pool":
            module = nn.AvgPool2d(kernel_size=mod_params.kernel)
            w = w // mod_params.kernel
            h = h // mod_params.kernel
        elif module_type == "upsample":
            module = nn.Upsample(scale_factor=mod_params.scale_factor, mode="bilinear", align_corners=False)
            w = w * mod_params.scale_factor
            h = h * mod_params.scale_factor
        elif module_type == "flatten":
            assert dim is None
            module = nn.Flatten(start_dim=-3, end_dim=-1)
            dim = w * h * c
        elif module_type == "reshape":
            assert c is None and h is None and w is None
            assert dim == np.prod(mod_params.shape)
            module = Reshape(mod_params.shape)
            c, h, w = mod_params.shape
        elif module_type == "linear":
            module = nn.Linear(dim, mod_params.dim)
            dim = mod_params.dim
        elif module_type == "layer_norm":
            module = nn.LayerNorm(dim)
        elif module_type == "tanh":
            module = nn.Tanh()
        elif module_type == "relu":
            module = nn.ReLU()
        elif module_type == "leaky_relu":
            module = nn.LeakyReLU(negative_slope=mod_params.alpha)
        else:
            raise NotImplementedError

        module_list.append(module)

        if verbose:
            print("output shape:", get_shape())
            if i == len(modules) - 1:
                print()

    if encoding:
        output_shape = dim
    else:
        output_shape = (c, h, w)

    return nn.Sequential(*module_list), output_shape
