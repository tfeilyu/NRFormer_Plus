import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import *
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class self_AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = self_AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=207):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class LScaledDotProductAttention(Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, groups=1):
        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_k = nn.Linear(d_model // groups, h * d_k // groups)
        self.fc_v = nn.Linear(d_model // groups, h * d_v // groups)
        self.fc_o = nn.Linear(h * d_v // groups, d_model // groups)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        xavier_uniform_(self.fc_q.weight)
        xavier_uniform_(self.fc_k.weight)
        xavier_uniform_(self.fc_v.weight)
        xavier_uniform_(self.fc_o.weight)
        constant_(self.fc_q.bias, 0)
        constant_(self.fc_k.bias, 0)
        constant_(self.fc_v.bias, 0)
        constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s, nq, self.groups, -1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values.view(b_s, nk, self.groups, -1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out.view(b_s, nq, self.groups, -1)).view(b_s, nq, -1)
        return out.permute(1, 0, 2)


class LMultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=.1, batch_first=False, groups=1, device=None, dtype=None):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h,
                                                    h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None,need_weights=False,attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out


class Lightformer(Module):

    __constants__ = ['norm']

    def __init__(self, attention_layer, num_layers, norm=None):
        super(Lightformer, self).__init__()
        self.layers = _get_clones(attention_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,src_v, mask: Optional[Tensor] = None) -> Tensor:
        output = src
        output_v = src_v
        for i, mod in enumerate(self.layers):
            if i % 2 ==0:
                output = mod(output, output_v)
            else:
                output = mod(output, output_v, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LightformerLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LightformerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                             **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward // 2, d_model // 2, **factory_kwargs)  ###

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(LightformerLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_v, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        x_v = src_v
        x = self.norm1(x + self._sa_block(x, x_v, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, x_v,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x_v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        b, l, d = x.size()
        x = self.linear2(self.dropout(self.activation(self.linear1(x))).view(b, l, 2, d*4 // 2))
        x= x.view(b, l, d)
        return self.dropout2(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    return F.gelu

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class LightGFormer(nn.Module):
    def __init__(self, config):
        super(LightGFormer, self).__init__()

        self.heads = 8
        self.layers = config['num_spatial_att_layer']
        self.hid_dim = config['hidden_channels']

        self.attention_layer = LightformerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=config['num_sensors'])

    def forward(self, input, input_v, mask):
        # print('hid_dim: ', self.hid_dim)
        x = input.permute(1, 0, 2)
        x_v = input_v.permute(1, 0, 2)
        x = self.lpos(x)
        x_v = self.lpos(x_v)
        output = self.attention(x, x_v, mask)
        output = output.permute(1, 0, 2)
        return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden

class loc_MLP(nn.Module):
    def __init__(self, num_dim):
        super(loc_MLP, self).__init__()
        self.fc1 = nn.Linear(2, num_dim)
        self.fc2 = nn.Linear(num_dim, num_dim)
        self.fc3 = nn.Linear(num_dim, 32)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.hidden_channels = config['hidden_channels']
        self.num_temporal_att_layer = config['num_temporal_att_layer']

        self.time_series_learning = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_channels, self.hidden_channels, num_heads=4, dropout=0.3)
                for _ in range(self.num_temporal_att_layer)
            ]
        )

    def forward(self, x):
        x = x.transpose(1, 3)
        for attn in self.time_series_learning:
            x = attn(x, dim=1)
        x = x.transpose(1, 3)
        return x


class NRFormer(nn.Module):
    def __init__(self, config, mask_support_adj):
        super(NRFormer, self).__init__()

        self.config = config

        self.noaa_list = []
        if self.config['Is_wind_angle']:
            self.noaa_list.append('wind_angle')
        if self.config['Is_wind_speed']:
            self.noaa_list.append('wind_speed')
        if self.config['Is_air_temperature']:
            self.noaa_list.append('air_temperature')
        if self.config['Is_dew_point']:
            self.noaa_list.append('dew_point')

        self.use_RevIN = config['use_RevIN']
        if config['use_RevIN']:
            self.revin = RevIN(config['num_sensors'])

        # 1. temporal time series learning
        self.start_time_series = nn.Conv2d(in_channels=1, out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)
        self.time_series_learning = SelfAttention(self.config)

        self.end_time_series = nn.Conv2d(in_channels=self.config['hidden_channels'] * self.config['in_length'],
                                      out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)

        # 2. location embedding learning
        if self.config['IsLocationInfo']:
            self.loc_mlp = loc_MLP(self.config['num_loc_mlp_dim'])

        # 3. node and time embedding learning
        # time embeddings
        time_embed_num = 0
        if config['IsDayOfYearEmbedding']:
            time_embed_num += 1
            self.doy_emb = nn.Parameter(torch.empty(366, self.config['hidden_channels']))
            nn.init.xavier_uniform_(self.doy_emb)
        # node embeddings
        # self.node_emb = nn.Parameter(torch.empty(self.num_sensors, self.mlp_node_dim))
        # nn.init.xavier_uniform_(self.node_emb)
        # time series, node, time embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.config['in_length']*3,
            out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)
        # temporal mlp
        self.hidden_dim = self.config['hidden_channels']
        self.temporal_mlp = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.config['num_mlp_layer'])])

        # 4. meteorological mlp
        if len(self.noaa_list)>0:
            self.meteo_start = nn.Conv2d(
                in_channels=len(self.noaa_list) * self.config['in_length'],
                out_channels=self.config['num_noaa_mlp_dim'], kernel_size=(1, 1), bias=True)
            self.meteo_mlp = nn.Sequential(
                *[MultiLayerPerceptron(self.config['num_noaa_mlp_dim'], self.config['num_noaa_mlp_dim']) for _ in range(self.config['num_noaa_mlp_layer'])])
            self.meteo_end = nn.Conv2d(
                in_channels=self.config['num_noaa_mlp_dim'],
                out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)

        # 5. temporal fusion
        fusion_dim = 0
        if len(self.noaa_list)>0: fusion_dim += 32
        if self.config['IsLocationInfo']: fusion_dim += 32
        self.temporal_fusion = nn.Conv2d(in_channels=self.config['hidden_channels']*2+fusion_dim, out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)

        # 6. spatial learning
        mask0 = mask_support_adj[0].detach()
        mask1 = mask_support_adj[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0
        self.LightTransfer = LightGFormer(config)

        # 7. end fusion
        end_dim = self.config['hidden_channels']*2
        self.end_conv1 = nn.Linear(end_dim, self.config['end_channels'])
        self.end_conv2 = nn.Linear(self.config['end_channels'], self.config['out_length'] * self.config['out_channels'])

    def forward(self, inputs, loc_feature):
        # inputs [64, 3, 307, 24]
        batch_size, num_features, num_nodes, his_steps = inputs.shape
        all_t_embedding = []

        if self.use_RevIN:
            x_enc = inputs[:, 0:1, :, :].squeeze(dim=1).transpose(1, 2)
            x_enc = self.revin(x_enc, 'norm')
            x_enc = x_enc.transpose(1, 2).unsqueeze(dim=1)
            if num_features>1:
                inputs = torch.cat((x_enc, inputs[:, 1:, :, :]), dim=1)
            else:
                inputs = x_enc

        # 1. temporal time series learning
        time_series = inputs[:, 0:1, :, :]
        temporal_start = self.start_time_series(time_series)
        temporal_conv = self.time_series_learning(temporal_start)
        temporal_conv = temporal_conv.reshape(batch_size, -1, num_nodes, 1)
        time_series_embedding = self.end_time_series(temporal_conv).squeeze(dim=-1).transpose(1, 2)
        all_t_embedding.append(time_series_embedding)

        # 2. location embedding learning
        if self.config['IsLocationInfo']:
            # loc_fts = torch.from_numpy(loc_feature).to(device=self.config['device'], dtype=torch.float)
            loc_embedding = self.loc_mlp(loc_feature.float())
            loc_embedding = loc_embedding.repeat(batch_size, 1, 1)
            all_t_embedding.append(loc_embedding)

        # 3. node and time embedding learning
        history_data = inputs.transpose(1, 3)
        history_data = history_data[:, :, :, 0:num_features-len(self.noaa_list)]
        # time embeddings
        tem_emb = []
        time_num = 1
        if self.config['IsDayOfYearEmbedding']:
            d_i_w_data = history_data[..., time_num]
            month_in_year_emb = self.doy_emb[(d_i_w_data[:, -1, :] * self.year_size).type(torch.LongTensor)]
            tem_emb.append(month_in_year_emb.transpose(1, 2).unsqueeze(-1))
            time_num += 1
        # node embedding
        # node_emb = []
        # node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # time series and time embedding layer
        input_data = history_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        # concat all embeddings
        hidden = torch.cat([time_series_emb] + tem_emb, dim=1)
        # temporal mlp
        temporal_mlp = self.temporal_mlp(hidden).squeeze(dim=-1).transpose(1, 2).contiguous()
        all_t_embedding.append(temporal_mlp)

        # 4. meteorological mlp
        if len(self.noaa_list)>0:
            meteorological_data = inputs[:, -len(self.noaa_list):, :, :]
            meteorological_data = meteorological_data.transpose(1, 2)
            meteorological_data = meteorological_data.reshape(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            meteorological_mlp = self.meteo_start(meteorological_data)
            meteorological_mlp = self.meteo_mlp(meteorological_mlp)
            meteorological_embedding = self.meteo_end(meteorological_mlp).squeeze(dim=-1).transpose(1, 2).contiguous()
            all_t_embedding.append(meteorological_embedding)

        # 5. temporal fusion
        x_temporal = torch.cat(all_t_embedding, dim=-1).unsqueeze(dim=-1).transpose(1,2)
        x_temporal = self.temporal_fusion(x_temporal).squeeze(dim=-1).transpose(1,2)

        # 6. spatial learning
        x_spatial = self.LightTransfer(x_temporal, temporal_mlp, self.mask)

        # 7. end fusion
        # TODO
        # select the residual embedding
        x = torch.cat([x_temporal]+[x_spatial], dim=-1)
        x = self.end_conv1(x)
        x = F.relu(x)
        x = self.end_conv2(x)
        x = x.unsqueeze(dim=1)

        if self.use_RevIN:
            x = x.squeeze(dim=1).transpose(1, 2).contiguous()
            x = self.revin(x, 'denorm')
            x = x.transpose(1, 2).unsqueeze(dim=1).contiguous()

        return x