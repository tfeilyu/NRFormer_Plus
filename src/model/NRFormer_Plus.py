import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.nn.init import constant_, xavier_uniform_
import copy
from torch.nn import MultiheadAttention
import math


class PGRT2(nn.Module):
    """
    Physics-Guided Radiation Transformer (PGRT) for Nuclear Radiation Spatiotemporal Prediction

    Key Innovations:
    1. Physics-informed neural network component for atmospheric diffusion
    2. Meteorological-driven radiation propagation module
    3. Multi-scale spatial modeling with local-global pathways
    4. Adaptive spatial attention based on monitoring station density
    """

    def __init__(self, config, mask_support_adj, cluster_ids=None):
        super(PGRT2, self).__init__()

        self.config = config
        self.hidden_dim = config['hidden_channels']
        self.num_nodes = config['num_sensors']
        self.in_steps = config['in_length']
        self.out_steps = config['out_length']

        self.noaa_list = []
        if self.config['Is_wind_angle']:
            self.noaa_list.append('wind_angle')
        if self.config['Is_wind_speed']:
            self.noaa_list.append('wind_speed')
        if self.config['Is_air_temperature']:
            self.noaa_list.append('air_temperature')
        if self.config['Is_dew_point']:
            self.noaa_list.append('dew_point')

        # Iteration 1: log-space and residual learning flags
        self.use_log_space = config.get('use_log_space', False)
        self.use_residual = config.get('use_residual', False)

        # Iteration 3: virtual global nodes for spatial attention
        self.num_global_nodes = config.get('num_global_nodes', 0)
        if self.num_global_nodes > 0:
            self.global_tokens = nn.Parameter(torch.randn(1, self.num_global_nodes, self.hidden_dim) * 0.02)
            self.global_tokens_v = nn.Parameter(torch.randn(1, self.num_global_nodes, self.hidden_dim) * 0.02)

        # Iteration 4: Region-aware attention bias
        self.num_region_clusters = config.get('num_region_clusters', 0)
        if self.num_region_clusters > 0 and cluster_ids is not None:
            self.register_buffer('cluster_ids', torch.tensor(cluster_ids, dtype=torch.long))
            self.region_embed = nn.Embedding(self.num_region_clusters, self.hidden_dim)
            nn.init.normal_(self.region_embed.weight, std=0.02)
        else:
            self.num_region_clusters = 0

        # Iteration 2: rain-aware gating
        self.use_rain_gate = config.get('use_rain_gate', False)
        if self.use_rain_gate and 'air_temperature' in self.noaa_list and 'dew_point' in self.noaa_list:
            self.temp_idx = self.noaa_list.index('air_temperature')
            self.dew_idx = self.noaa_list.index('dew_point')
            self.rain_gate_net = nn.Sequential(
                nn.Linear(self.in_steps, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.use_rain_gate = False

        self.tem_num = 3

        # 1. Data Preprocessing Module
        if self.config['use_RevIN']:
            self.revin = RevIN(self.num_nodes, eps=1e-5, affine=True)

        # 2. Feature Extraction
        self.start_time_series = nn.Conv2d(in_channels=1, out_channels=self.config['hidden_channels'],
                                           kernel_size=(1, 1), bias=True)
        self.time_series_learning = SelfAttention(self.config)

        self.end_time_series = nn.Conv2d(in_channels=self.config['hidden_channels'] * self.config['in_length'],
                                         out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)

        # self.radiation_encoder = nn.Conv2d(1, self.hidden_dim, kernel_size=(1, 1))
        if len(self.noaa_list)>0:
            self.meteo_encoder = MeteorologicalEncoder(config, self.noaa_list)

        if self.config['IsLocationEncoder']:
            self.location_encoder = LocationEncoder(config)
            self.tem_num += 1

        if self.config['IsLocationInfo']:
            self.rad_loc_emb_layer = nn.Conv2d(
                in_channels=3 * self.config['in_length'],
                out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)
            self.rad_loc_mlp = nn.Sequential(
                *[MultiLayerPerceptron(self.config['hidden_channels'], self.config['hidden_channels']) for _ in
                  range(3)])
            self.tem_num += 1

        if self.config['IsDayOfYearEmbedding']:
            self.temporal_encoder = TemporalEncoder(config)
            self.tem_num += 1

        # 3. Physics Module
        # Reconstruct [N, N] adjacency from row-wise tensor list
        adj_np = torch.stack(mask_support_adj).detach().cpu().numpy()
        adj_np = (adj_np > 0).astype(float)
        self.physics_type = config.get('physics_type', 'diffusion')
        if self.physics_type == 'regional':
            # Iter6: RegionalCoherenceModule — data says no diffusion (F7)
            self.physics_module = RegionalCoherenceModule(config, self.noaa_list, adj_matrix=adj_np)
        else:
            self.physics_module = AtmosphericDiffusionModule(config, self.noaa_list, adj_matrix=adj_np)

        # 4. Meteorological-Driven Propagation Module
        # self.wind_propagation = WindPropagationModule(config)

        self.temporal_fusion = nn.Conv2d(in_channels=self.config['hidden_channels'] * self.tem_num,
                                         out_channels=self.config['hidden_channels'], kernel_size=(1, 1), bias=True)
        # 6. spatial learning
        mask0 = mask_support_adj[0].detach()
        mask1 = mask_support_adj[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0
        self.LightTransfer = LightGFormer(config)

        # 6. Temporal Dynamics Module
        # self.temporal_dynamics = TemporalDynamicsModule(config)

        # 7. Adaptive Spatial Attention
        # self.adaptive_attention = AdaptiveSpatialAttention(config)

        # 8. Fusion + Output Projection
        hid = self.config['hidden_channels']
        self.fusion_type = config.get('fusion_type', '3way')  # '2way' or '3way'
        # Iter5: spatial attention input swap (match NRFormer's proven Q/K/V order)
        self.spatial_swap = config.get('spatial_swap', False)

        if self.fusion_type == '2way':
            # Match NRFormer: simple concat(x_temporal, x_spatial)
            end_dim = hid * 2
        else:
            # Original 3-way gated fusion
            self.fusion_gate = nn.Sequential(
                nn.Linear(hid * 3, hid * 3),
                nn.Sigmoid()
            )
            self.fusion_norm = nn.LayerNorm(hid * 3)
            end_dim = hid * 3
        self.end_conv1 = nn.Linear(end_dim, self.config['end_channels'])
        self.end_conv2 = nn.Linear(self.config['end_channels'], self.config['out_length'] * self.config['out_channels'])


    def forward(self, inputs, loc_feature):
        """
        Args:
            inputs: [batch_size, 6, num_nodes, in_steps]
                - inputs[:, 0]: radiation values
                - inputs[:, 1]: day of year
                - inputs[:, 2:6]: meteorological features
            loc_feature: [num_nodes, 2] - longitude and latitude

        Returns:
            output: [batch_size, 1, num_nodes, out_steps]
        """
        batch_size, num_features, num_nodes, his_steps = inputs.shape

        # 1. Normalize radiation values
        # Note: if use_log_space=True, radiation is already log1p-transformed in DataProcessing
        radiation = inputs[:, 0:1, :, :].squeeze(1).transpose(1, 2)  # [B, T, N]

        if self.config['use_RevIN']:
            radiation = self.revin(radiation, 'norm')
        radiation_norm = radiation.transpose(1, 2).unsqueeze(1)  # [B, 1, N, T]

        # Iteration 1b: Save last known value for residual prediction
        if self.use_residual:
            last_value = radiation_norm[:, :, :, -1:]  # [B, 1, N, 1] in normalized space

        # 2. Extract features
        # Radiation features
        radiation_start = self.start_time_series(radiation_norm[:, 0:1, :, :])
        radiation_conv = self.time_series_learning(radiation_start)
        radiation_conv = radiation_conv.reshape(batch_size, -1, num_nodes, 1)
        rad_feat = self.end_time_series(radiation_conv).squeeze(dim=-1).transpose(1, 2)
        #
        # rad_feat = self.radiation_encoder(radiation_norm)  # [B, hidden_dim, N, T]
        # rad_feat = rad_feat.mean(dim=-1).transpose(1, 2)  # [B, N, hidden_dim]

        # Location features
        if self.config['IsLocationEncoder']:
            loc_feat = self.location_encoder(loc_feature, batch_size)  # [B, N, hidden_dim]
        if self.config['IsLocationInfo']:
            rad_loc_info = inputs[:, 0:3, :, :].contiguous()
            rad_loc_data = rad_loc_info.view(batch_size, 1, num_nodes, -1).transpose(1, 3)
            rad_loc_feat = self.rad_loc_emb_layer(rad_loc_data)

            temporal_mlp = self.rad_loc_mlp(rad_loc_feat).squeeze(dim=-1).transpose(1, 2).contiguous()

        # Meteorological features
        meteo_data = None
        if len(self.noaa_list)>0:
            meteo_data = inputs[:, -len(self.noaa_list):, :, :]  # wind_angle, wind_speed, air_temp, dew_point
            meteo_feat = self.meteo_encoder(meteo_data)  # [B, N, hidden_dim]


        # doy features
        if self.config['IsDayOfYearEmbedding']:
            if self.config['IsLocationInfo']:
                n = 3
            else:
                n = 1
            doy_info = inputs[:, n:n+1, :, :]  # day of year
            doy_feat = self.temporal_encoder(doy_info)  # [B, N, hidden_dim]

        # 3. Physics-informed atmospheric diffusion
        physics_constraint = self.physics_module(
            radiation_norm, meteo_data, loc_feature
        )  # [B, N, hidden_dim]

        # 4. Wind-driven propagation modeling
        # propagation_feat = self.wind_propagation(
        #     rad_feat, meteo_data, loc_feature
        # )  # [B, N, hidden_dim]

        # Iteration 2: Rain-aware gating — boost physics/meteo features during humid conditions
        if self.use_rain_gate:
            # dryness_index = air_temperature - dew_point (low = humid/rainy)
            air_temp = meteo_data[:, self.temp_idx, :, :]   # [B, N, T]
            dew_point = meteo_data[:, self.dew_idx, :, :]   # [B, N, T]
            dryness = air_temp - dew_point                   # [B, N, T]
            rain_gate = self.rain_gate_net(-dryness)          # [B, N, 1], high when humid
            # Boost physics and meteo features during rain events
            physics_constraint = physics_constraint * (1.0 + rain_gate)
            meteo_feat = meteo_feat * (1.0 + rain_gate)

        # 5. temporal fusion
        emb = [rad_feat, physics_constraint]
        if self.config['IsLocationEncoder']:
            emb += [loc_feat]
        if self.config['IsLocationInfo']:
            emb += [temporal_mlp]
        if len(self.noaa_list) > 0:
            emb += [meteo_feat]
        if self.config['IsDayOfYearEmbedding']:
            emb += [doy_feat]


        x_temporal = torch.cat(emb, dim=-1).unsqueeze(dim=-1).transpose(1, 2)
        x_temporal = self.temporal_fusion(x_temporal).squeeze(dim=-1).transpose(1, 2)

        # 6. spatial learning
        # Iter5: spatial_swap=True matches NRFormer's Q/K=fused, V=raw pattern
        if self.spatial_swap:
            sp_qk, sp_v = x_temporal, rad_feat
        else:
            sp_qk, sp_v = rad_feat, x_temporal

        # Iteration 4: Add region-aware bias to spatial Q/K and V
        if self.num_region_clusters > 0:
            region_bias = self.region_embed(self.cluster_ids)  # [N, hidden_dim]
            sp_qk = sp_qk + region_bias.unsqueeze(0)  # [B, N, H] + [1, N, H]
            sp_v = sp_v + region_bias.unsqueeze(0)

        if self.num_global_nodes > 0:
            K = self.num_global_nodes
            g_tokens = self.global_tokens.expand(batch_size, -1, -1)
            g_tokens_v = self.global_tokens_v.expand(batch_size, -1, -1)
            sp_qk_aug = torch.cat([sp_qk, g_tokens], dim=1)
            sp_v_aug = torch.cat([sp_v, g_tokens_v], dim=1)
            N = num_nodes
            mask_aug = torch.zeros(N + K, N + K, dtype=torch.bool, device=self.mask.device)
            mask_aug[:N, :N] = self.mask
            x_spatial = self.LightTransfer(sp_qk_aug, sp_v_aug, mask_aug)
            x_spatial = x_spatial[:, :num_nodes, :]
        else:
            x_spatial = self.LightTransfer(sp_qk, sp_v, self.mask)

        # 8. Fusion + Output projection
        if self.fusion_type == '2way':
            # Match NRFormer: simple concat
            output = torch.cat([x_temporal, x_spatial], dim=-1)
        else:
            # 3-way gated fusion
            concat = torch.cat([rad_feat, x_temporal, x_spatial], dim=-1)
            gate = self.fusion_gate(concat)
            output = self.fusion_norm(concat * gate)
        output = self.end_conv1(output)
        output = F.relu(output)
        output = self.end_conv2(output)
        output = output.unsqueeze(dim=1)

        # Iteration 1b: Residual prediction — add last known value
        if self.use_residual:
            # output is predicted delta in normalized space: [B, 1, N, out_steps]
            # last_value: [B, 1, N, 1] broadcasts across out_steps
            output = last_value + output

        # Denormalize
        output = output.squeeze(1).transpose(1, 2)  # [B, T, N]
        if self.config['use_RevIN']:
            output = self.revin(output, 'denorm')
        output = output.transpose(1, 2).unsqueeze(1)  # [B, 1, N, T]

        # Note: if use_log_space=True, inverse log transform (expm1) is applied in trainer
        return output


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.hidden_channels = config['hidden_channels']
        self.num_temporal_att_layer = config['num_temporal_att_layer']
        dropout = config.get('temporal_dropout', 0.1)
        ffn_dim = self.hidden_channels * config.get('ffn_ratio', 4)

        # Learned temporal positional encoding
        in_length = config.get('in_length', 24)
        self.temporal_pe = nn.Parameter(torch.randn(1, in_length, 1, self.hidden_channels) * 0.02)

        self.time_series_learning = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_channels, ffn_dim, num_heads=4, dropout=dropout)
                for _ in range(self.num_temporal_att_layer)
            ]
        )

    def forward(self, x):
        x = x.transpose(1, 3)  # [B, T, N, D]
        x = x + self.temporal_pe  # add temporal positional encoding
        for attn in self.time_series_learning:
            x = attn(x, dim=1)
        x = x.transpose(1, 3)
        return x


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

class LightGFormer(nn.Module):
    def __init__(self, config):
        super(LightGFormer, self).__init__()

        self.heads = config.get('spatial_heads', 4)
        self.layers = config['num_spatial_att_layer']
        self.hid_dim = config['hidden_channels']

        self.attention_layer = LightformerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = Lightformer(self.attention_layer, self.layers, self.attention_norm)
        max_nodes = config['num_sensors'] + config.get('num_global_nodes', 0)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=max_nodes)

    def forward(self, input, input_v, mask):
        # print('hid_dim: ', self.hid_dim)
        x = input.permute(1, 0, 2)
        x_v = input_v.permute(1, 0, 2)
        x = self.lpos(x)
        x_v = self.lpos(x_v)
        output = self.attention(x, x_v, mask)
        output = output.permute(1, 0, 2)
        return output

class LMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=.1, batch_first=False, groups=1, device=None, dtype=None):
        super(LMultiHeadAttention, self).__init__()

        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_model // h, d_v=d_model // h,
                                                    h=h, dropout=dropout)

    def forward(self, queries, keys, values, attn_mask=None, key_padding_mask=None,need_weights=False,attention_weights=None):
        out = self.attention(queries, keys, values, attn_mask, attention_weights)
        return out, out

class LScaledDotProductAttention(nn.Module):

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


class LightformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LightformerLayer, self).__init__()
        self.self_attn = LMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                             **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward // 2, d_model // 2, **factory_kwargs)  ###

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
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

def _get_activation_fn(activation):
    return F.gelu

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Lightformer(nn.Module):

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


class LearnedPositionalEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=207):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class MeteorologicalEncoder(nn.Module):
    """Encode meteorological features with physical relationships"""

    def __init__(self, config, noaa_list):
        super(MeteorologicalEncoder, self).__init__()
        self.noaa_list = noaa_list
        self.hidden_dim = config['hidden_channels']

        # Separate encoders for different meteorological variables
        self.wind_num = 0
        if 'wind_angle' in self.noaa_list:
            self.wind_num += 1
        if 'wind_speed' in self.noaa_list:
            self.wind_num += 1
        if self.wind_num > 0:
            self.wind_encoder = nn.Sequential(
                nn.Conv2d(self.wind_num, self.hidden_dim // 2, kernel_size=(1, 1)),  # wind angle & speed
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim // 2, self.hidden_dim // 2, kernel_size=(1, 3), padding=(0, 1))
            )

        self.tem_num = 0
        if 'air_temperature' in self.noaa_list:
            self.tem_num += 1
        if 'dew_point' in self.noaa_list:
            self.tem_num += 1
        if self.tem_num > 0:
            self.temp_encoder = nn.Sequential(
                nn.Conv2d(self.tem_num, self.hidden_dim // 2, kernel_size=(1, 1)),  # air temp & dew point
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim // 2, self.hidden_dim // 2, kernel_size=(1, 3), padding=(0, 1))
            )
        in_length = config.get('in_length', 24)
        if self.wind_num > 0 and self.tem_num > 0:
            dim = 2 * int(self.hidden_dim / 2) * in_length
        else:
            dim = 1 * int(self.hidden_dim / 2) * in_length
        self.meteo_start = nn.Conv2d(in_channels=dim, out_channels=64, kernel_size=(1, 1), bias=True)
        self.meteo_mlp = nn.Sequential(
                *[MultiLayerPerceptron(64, 64) for _ in range(2)])
        self.meteo_end = nn.Conv2d(in_channels=64, out_channels=config['hidden_channels'], kernel_size=(1, 1), bias=True)
        # self.fusion = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )

    def forward(self, meteo_data):
        """
        Args:
            meteo_data: [B, 4, N, T] - wind_angle, wind_speed, air_temp, dew_point
        """
        batch_size, num_features, num_nodes = meteo_data.shape[0], meteo_data.shape[1], meteo_data.shape[2]

        if self.wind_num > 0:
            wind_data = meteo_data[:, 0:self.wind_num, :, :]  # wind features
        if self.tem_num > 0:
            temp_data = meteo_data[:, self.wind_num:, :, :]  # temperature features

        emb = []
        if self.wind_num > 0:
            wind_feat = self.wind_encoder(wind_data)
            emb.append(wind_feat)
        if self.tem_num > 0:
            temp_feat = self.temp_encoder(temp_data)
            emb.append(temp_feat)

        combined = torch.cat(emb, dim=1)
        combined = combined.reshape(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        output = self.meteo_start(combined)
        output = self.meteo_mlp(output)
        output = self.meteo_end(output).squeeze(dim=-1).transpose(1, 2).contiguous()

        return output


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


class LocationEncoder(nn.Module):
    """Encode spatial locations with learnable embeddings"""

    def __init__(self, config):
        super(LocationEncoder, self).__init__()
        self.hidden_dim = config['hidden_channels']

        self.loc_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
        )
        self._loc_buffer = None

    def forward(self, loc_feature, batch_size):
        device = self.loc_mlp[0].weight.device
        if self._loc_buffer is None or self._loc_buffer.device != device:
            if isinstance(loc_feature, torch.Tensor):
                self._loc_buffer = loc_feature.float().to(device)
            else:
                self._loc_buffer = torch.tensor(loc_feature, dtype=torch.float32).to(device)
        loc_embedding = self.loc_mlp(self._loc_buffer)
        return loc_embedding.unsqueeze(0).expand(batch_size, -1, -1)


class TemporalEncoder(nn.Module):
    """Encode temporal information"""

    def __init__(self, config):
        super(TemporalEncoder, self).__init__()
        self.hidden_dim = config['hidden_channels']

        # Day of year embedding
        self.doy_embedding = nn.Embedding(366, self.hidden_dim)
        nn.init.xavier_uniform_(self.doy_embedding.weight)
        # Temporal pattern extraction
        # self.temporal_conv = nn.Conv1d(1, self.hidden_dim, kernel_size=3, padding=1)
        # self.temporal_attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)

    def forward(self, temporal_info):
        batch_size, _, num_nodes, num_steps = temporal_info.shape

        # Day of year embedding
        doy = temporal_info[:, 0, :, -1].long()  # Use last timestep's day of year
        doy_emb = self.doy_embedding(doy)  # [B, N, hidden_dim]

        # Extract temporal patterns from radiation data
        # rad_temp = radiation_data.squeeze(1).transpose(1, 2)  # [B, T, N]
        # rad_temp = rad_temp.transpose(1, 2).reshape(-1, 1, num_steps)  # [B*N, 1, T]

        # temp_feat = self.temporal_conv(rad_temp)  # [B*N, hidden_dim, T]
        # temp_feat = temp_feat.mean(dim=-1).reshape(batch_size, num_nodes, -1)  # [B, N, hidden_dim]

        # Self-attention for temporal dependencies
        # temp_out, _ = self.temporal_attention(temp_feat, temp_feat, temp_feat)

        return doy_emb


class AtmosphericDiffusionModule(nn.Module):
    """Physics-informed module for atmospheric diffusion modeling.

    Implements a neural approximation of the atmospheric diffusion equation:
        dC/dt = D * nabla^2(C)
    where C is radiation concentration, D is the diffusion coefficient,
    and nabla^2 is the graph Laplacian operator.
    """

    def __init__(self, config, noaa_list, adj_matrix=None):
        super(AtmosphericDiffusionModule, self).__init__()
        self.hidden_dim = config['hidden_channels']
        n = len(noaa_list) + 2

        # Pre-compute row-normalized adjacency for graph Laplacian
        if adj_matrix is not None:
            adj = torch.FloatTensor(adj_matrix)
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            adj_norm = adj / deg
            self.register_buffer('adj_norm', adj_norm)
        else:
            self.adj_norm = None

        # Diffusion coefficient estimation from meteo + location
        self.diffusion_net = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive diffusion coefficient D > 0
        )

        # Temporal gradient encoder (dC/dt approximation)
        self.temporal_grad_net = nn.Sequential(
            nn.Linear(config.get('in_length', 24), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Physics constraint encoder: [C, D, nabla^2(C), dC/dt] -> hidden_dim
        self.physics_encoder = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, radiation_data, meteo_data, loc_feature):
        batch_size, _, num_nodes, num_steps = radiation_data.shape
        device = radiation_data.device

        # Prepare meteorological input (time-averaged): [B, N, C_m]
        meteo_avg = meteo_data.mean(dim=-1).transpose(1, 2)  # [B, N, C_m]

        # Prepare location input (ensure float32): [N, 2]
        if isinstance(loc_feature, torch.Tensor):
            loc_tensor = loc_feature.float().to(device)
        else:
            loc_tensor = torch.tensor(loc_feature, dtype=torch.float32, device=device)
        if loc_tensor.dim() == 1:
            loc_tensor = loc_tensor.unsqueeze(-1)
        # Broadcast to [B, N, 2]
        loc_expanded = loc_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        # Estimate spatially-varying diffusion coefficient D: [B, N, 1]
        diff_input = torch.cat([meteo_avg, loc_expanded], dim=-1)  # [B, N, C_m+2]
        diffusion_coeff = self.diffusion_net(diff_input)  # [B, N, 1]

        # Current radiation concentration
        rad_current = radiation_data[:, 0, :, -1]  # [B, N]

        # Compute graph Laplacian: nabla^2(C) = A_norm @ C - C
        laplacian = self.compute_graph_laplacian(rad_current)  # [B, N]

        # Approximate temporal gradient dC/dt from the input sequence
        rad_sequence = radiation_data[:, 0, :, :]  # [B, N, T]
        temporal_grad = self.temporal_grad_net(rad_sequence).squeeze(-1)  # [B, N]

        # Physics triplet: [C, D, nabla^2(C), dC/dt] -> [B, N, 4]
        physics_features = torch.stack([
            rad_current,
            diffusion_coeff.squeeze(-1),
            laplacian,
            temporal_grad
        ], dim=-1)  # [B, N, 4]

        physics_out = self.physics_encoder(physics_features)  # [B, N, hidden_dim]

        # Store diagnostics for external logging
        self._last_diagnostics = {
            'D_mean': diffusion_coeff.mean().item(),
            'D_std': diffusion_coeff.std().item(),
            'D_min': diffusion_coeff.min().item(),
            'D_max': diffusion_coeff.max().item(),
            'laplacian_abs_mean': laplacian.abs().mean().item(),
            'temporal_grad_std': temporal_grad.std().item(),
        }

        return physics_out

    def compute_graph_laplacian(self, values):
        """Compute graph Laplacian using actual spatial adjacency.

        L(x) = A_norm @ x - x  (neighbor mean minus self)

        Args:
            values: [B, N] radiation values
        Returns:
            laplacian: [B, N] graph Laplacian applied to values
        """
        if self.adj_norm is not None:
            # Real graph Laplacian: neighbor_mean - self
            neighbor_mean = torch.matmul(values, self.adj_norm.t())  # [B, N]
            laplacian = neighbor_mean - values
        else:
            # Fallback: global mean centering
            mean_val = values.mean(dim=1, keepdim=True)
            laplacian = values - mean_val
        return laplacian


class WindPropagationModule(nn.Module):
    """Model radiation propagation driven by wind patterns"""

    def __init__(self, config):
        super(WindPropagationModule, self).__init__()
        self.hidden_dim = config['hidden_channels']

        # Wind field encoder
        self.wind_encoder = nn.Sequential(
            nn.Linear(2, 64),  # wind angle and speed
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Propagation dynamics
        self.propagation_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 32 + 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Directional attention
        # self.directional_attention = DirectionalAttention(config)

    def forward(self, spatial_features, meteo_data, loc_feature):
        batch_size, num_nodes, _ = spatial_features.shape

        # Extract wind data
        wind_data = meteo_data[:, 0:2, :, :].mean(dim=-1).transpose(1, 2)  # [B, N, 2]
        wind_features = self.wind_encoder(wind_data)  # [B, N, 32]

        # Prepare location features
        loc_tensor = torch.tensor(loc_feature, dtype=torch.float32).to(spatial_features.device)
        loc_expanded = loc_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        # Propagation features
        prop_input = torch.cat([spatial_features, wind_features, loc_expanded], dim=-1)
        prop_features = self.propagation_net(prop_input)  # [B, N, hidden_dim]

        # Apply directional attention based on wind patterns
        # prop_output = self.directional_attention(prop_features, wind_data, loc_feature)

        return prop_features


class DirectionalAttention(nn.Module):
    """Attention mechanism considering wind direction for propagation"""

    def __init__(self, config):
        super(DirectionalAttention, self).__init__()
        self.hidden_dim = config['hidden_channels']

        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.direction_encoder = nn.Linear(4, 64)  # relative position + wind direction
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features, wind_data, loc_feature):
        batch_size, num_nodes, _ = features.shape

        # Project features
        Q = self.query_proj(features)  # [B, N, hidden_dim]
        K = self.key_proj(features)  # [B, N, hidden_dim]
        V = self.value_proj(features)  # [B, N, hidden_dim]

        # Compute directional attention weights
        attention_weights = []
        loc_tensor = torch.tensor(loc_feature, dtype=torch.float32).to(features.device)

        for i in range(num_nodes):
            # Relative positions
            rel_pos = loc_tensor - loc_tensor[i:i + 1]  # [N, 2]

            # Wind direction alignment
            wind_angle = wind_data[:, i:i + 1, 0:1]  # [B, 1, 1]
            wind_speed = wind_data[:, i:i + 1, 1:2]  # [B, 1, 1]

            # Directional features
            dir_features = torch.cat([
                rel_pos.unsqueeze(0).expand(batch_size, -1, -1),
                wind_angle.expand(batch_size, num_nodes, 1),
                wind_speed.expand(batch_size, num_nodes, 1)
            ], dim=-1)  # [B, N, 4]

            dir_encoded = self.direction_encoder(dir_features)  # [B, N, 64]

            # Compute attention scores
            q_expanded = Q[:, i:i + 1, :].expand(batch_size, num_nodes, -1)
            att_input = torch.cat([q_expanded * K, dir_encoded], dim=-1)
            att_scores = self.attention_mlp(att_input).squeeze(-1)  # [B, N]

            attention_weights.append(att_scores.unsqueeze(1))

        attention_weights = torch.cat(attention_weights, dim=1)  # [B, N, N]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention
        output = torch.bmm(attention_weights, V)

        return output


class GraphConvLayer(nn.Module):
    """Graph convolution layer for local spatial modeling"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x, adj_matrix):
        # x: [B, N, F]
        # adj_matrix: [N, N]

        # Graph convolution: A * X * W
        support = self.linear(x)  # [B, N, F']
        output = torch.bmm(
            adj_matrix.unsqueeze(0).expand(x.shape[0], -1, -1),
            support
        )

        return self.activation(output)


class SpatialTransformer(nn.Module):
    """Transformer for global spatial relationships"""

    def __init__(self, config):
        super(SpatialTransformer, self).__init__()
        self.hidden_dim = config['hidden_channels']

        # Positional encoding for locations
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
        )

        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, features, loc_feature):
        # Add positional encoding
        loc_tensor = torch.tensor(loc_feature, dtype=torch.float32).to(features.device)
        pos_encoding = self.pos_encoder(loc_tensor)
        pos_encoding = pos_encoding.unsqueeze(0).expand(features.shape[0], -1, -1)

        # Combine with features
        transformer_input = features + pos_encoding

        # Apply transformer
        output = self.transformer(transformer_input)

        return output


class RevIN(nn.Module):
    """Reversible Instance Normalization"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
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
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x