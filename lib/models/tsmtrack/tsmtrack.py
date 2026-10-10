import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.tsmtrack.vit_care import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import Mlp
# DAM/CDA disabled: keep the implementation file intact but do not attach it to the model.
# from lib.models.layers.CDA_Module import CDA
from lib.models.layers.attn import Attention_qkv
from lib.models.layers.temporal_attention import build_temporal_attention
from functools import partial
import torch.nn.functional as F

class TSMTrack(nn.Module):

    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER"):
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.box_head = box_head
        # DAM/CDA disabled.
        # self.offset_range_factor = cfg.MODEL.CDA.OFFSET
        # self.track_query_len = cfg.MODEL.CDA.TRACK_QUERY
        self.template_number = cfg.DATA.TEMPLATE.NUMBER
        temporal_cfg = getattr(cfg.MODEL, "TEMPORAL", None)
        temporal_enabled = bool(getattr(temporal_cfg, "ENABLE", False)) if temporal_cfg is not None else False
        self.use_forward_topk = bool(
            getattr(temporal_cfg, "FORWARD_TOPK", False)
        ) if temporal_cfg is not None else False
        self.use_temporal_attention = temporal_enabled and self.use_forward_topk
        self.temporal_attention_type = str(getattr(temporal_cfg, "TYPE", "fused_tav")).lower() if temporal_cfg is not None else ""
        self.sparse_css_keep_ratio = float(getattr(temporal_cfg, "SPARSE_KEEP_RATIO", 0.75)) if temporal_cfg is not None else 0.75
        self.use_sparse_css_routing = bool(
            getattr(temporal_cfg, "SPARSE_ROUTING", True)
        ) if temporal_cfg is not None else True
        self.use_patch_selection = bool(
            getattr(temporal_cfg, "PATCH_SELECTION", False)
        ) if temporal_cfg is not None else False
        if self.use_patch_selection and not self.use_temporal_attention:
            raise ValueError("PATCH_SELECTION=True requires TEMPORAL.ENABLE=True and FORWARD_TOPK=True.")
        if self.use_temporal_attention:
            self.temporal_attention = build_temporal_attention(
                attention_type=self.temporal_attention_type,
                embed_dim=hidden_dim,
                num_heads=getattr(temporal_cfg, "NUM_HEADS", 8),
                dropout=getattr(temporal_cfg, "DROPOUT", 0.1),
            )
        else:
            self.temporal_attention = None

        self.decode_fuse_search = conv(hidden_dim*2, hidden_dim)
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    @staticmethod
    def _minmax_norm(score):
        score_min = score.min(dim=1, keepdim=True).values
        score_max = score.max(dim=1, keepdim=True).values
        return (score - score_min) / (score_max - score_min + 1e-6)

    def _compute_query_scores(self, feat_last_r, feat_last_x, temp_r_str, temp_x_str):
        feat_last_r_norm = F.normalize(feat_last_r, dim=-1)
        feat_last_x_norm = F.normalize(feat_last_x, dim=-1)
        temp_r_str_norm = F.normalize(temp_r_str, dim=-1)
        temp_x_str_norm = F.normalize(temp_x_str, dim=-1)

        score_r = torch.matmul(feat_last_r_norm, temp_r_str_norm.transpose(1, 2)).squeeze(-1)
        score_x = torch.matmul(feat_last_x_norm, temp_x_str_norm.transpose(1, 2)).squeeze(-1)
        return (0.5 * (score_r + score_x)).clamp(-1.0, 1.0)

    def _select_foreground_indices_from_query(self, feat_last_r, feat_last_x, temp_r_str, temp_x_str):
        # Query-guided fallback used only when no temporal neighborhood exists.
        token_score = self._compute_query_scores(feat_last_r, feat_last_x, temp_r_str, temp_x_str)

        num_tokens = token_score.shape[1]
        keep_tokens = max(1, min(num_tokens, int(round(num_tokens * self.sparse_css_keep_ratio))))
        return torch.topk(token_score, k=keep_tokens, dim=1).indices

    @staticmethod
    def _get_frame_motion_weight(motion_weights, frame_idx, batch_size, device):
        if motion_weights is None:
            default_weight = 0.0 if frame_idx == 0 else 0.5
            return torch.full((batch_size, 1), default_weight, device=device)

        if torch.is_tensor(motion_weights):
            weight = motion_weights[frame_idx]
        else:
            weight = motion_weights[frame_idx]
            if not torch.is_tensor(weight):
                weight = torch.tensor(weight, device=device)

        weight = weight.to(device=device, dtype=torch.float32)
        if weight.dim() == 0:
            weight = weight.view(1, 1).expand(batch_size, 1)
        elif weight.dim() == 1:
            weight = weight.view(-1, 1)
        elif weight.dim() > 2:
            weight = weight.view(weight.shape[0], -1)[:, :1]
        return weight.clamp(0.0, 1.0)

    def _select_motion_foreground_indices(
            self, feat_last_r_list, feat_last_x_list, temp_r_str_list, temp_x_str_list, motion_weights=None):
        # Keep patches that change with respect to the previous frame and
        # suppress highly similar static background patches at the same spatial
        # locations. This keeps the routing causal for online tracking.
        num_frames = len(feat_last_r_list)
        if num_frames == 0:
            return []

        if num_frames == 1:
            return [self._select_foreground_indices_from_query(
                feat_last_r_list[0], feat_last_x_list[0], temp_r_str_list[0], temp_x_str_list[0]
            )]

        num_tokens = feat_last_r_list[0].shape[1]
        keep_tokens = max(1, min(num_tokens, int(round(num_tokens * self.sparse_css_keep_ratio))))

        feat_r_seq = torch.stack(feat_last_r_list, dim=0)
        feat_x_seq = torch.stack(feat_last_x_list, dim=0)
        feat_r_seq = F.normalize(feat_r_seq, dim=-1)
        feat_x_seq = F.normalize(feat_x_seq, dim=-1)

        motion_scores = []
        for frame_idx in range(num_frames):
            if frame_idx == 0:
                motion_scores.append(torch.zeros_like(feat_r_seq[frame_idx, :, :, 0]))
                continue

            sim_prev_r = (feat_r_seq[frame_idx] * feat_r_seq[frame_idx - 1]).sum(dim=-1)
            sim_prev_x = (feat_x_seq[frame_idx] * feat_x_seq[frame_idx - 1]).sum(dim=-1)
            prev_similarity = 0.5 * (sim_prev_r + sim_prev_x)
            # Scale the cosine-distance feature-change score from [0, 2] to
            # [0, 1] without per-frame min-max normalization.
            motion_scores.append(
                (0.5 * (1.0 - prev_similarity)).clamp(0.0, 1.0)
            )

        topk_idx_list = []
        for frame_idx, motion_score in enumerate(motion_scores):
            query_score = self._compute_query_scores(
                feat_last_r_list[frame_idx],
                feat_last_x_list[frame_idx],
                temp_r_str_list[frame_idx],
                temp_x_str_list[frame_idx],
            )
            # Element-wise score at the same spatial patch position. Motion
            # boosts only patches with positive template relevance:
            # S_final = S_q + ReLU(S_q) * S_v.
            # token_score = query_score + F.relu(query_score)* motion_score
            token_score = query_score + F.relu(query_score)* motion_score
            topk_idx = torch.topk(token_score, k=keep_tokens, dim=1).indices
            topk_idx_list.append(topk_idx)
        return topk_idx_list

    def _apply_css_with_routing(self, feat_last_r, feat_last_x, temp_r_str, temp_x_str, topk_idx=None):
        if (not self.use_sparse_css_routing) or self.sparse_css_keep_ratio >= 1.0:
            temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_last_r, feat_last_r)
            temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
            att_r = torch.matmul(feat_last_r, temp_attn_r.transpose(1, 2))
            feat_last_r = att_r * feat_last_r

            temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_last_x, feat_last_x)
            temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
            att_x = torch.matmul(feat_last_x, temp_attn_x.transpose(1, 2))
            feat_last_x = att_x * feat_last_x
            return feat_last_r, feat_last_x, None

        if topk_idx is None:
            topk_idx = self._select_foreground_indices_from_query(
                feat_last_r, feat_last_x, temp_r_str, temp_x_str
            )
        gather_index = topk_idx.unsqueeze(-1).expand(-1, -1, feat_last_r.size(-1))

        feat_fg_r = torch.gather(feat_last_r, 1, gather_index)
        feat_fg_x = torch.gather(feat_last_x, 1, gather_index)

        temp_attn_r = temp_r_str + self.CSS_strengthen_r(temp_r_str, feat_fg_r, feat_fg_r)
        temp_attn_r = temp_attn_r + self.CSS_process_r(temp_attn_r)
        att_r = torch.matmul(feat_fg_r, temp_attn_r.transpose(1, 2))
        feat_fg_r = att_r * feat_fg_r

        temp_attn_x = temp_x_str + self.CSS_strengthen_x(temp_x_str, feat_fg_x, feat_fg_x)
        temp_attn_x = temp_attn_x + self.CSS_process_x(temp_attn_x)
        att_x = torch.matmul(feat_fg_x, temp_attn_x.transpose(1, 2))
        feat_fg_x = att_x * feat_fg_x

        feat_out_r = feat_last_r.clone()
        feat_out_x = feat_last_x.clone()
        feat_out_r.scatter_(1, gather_index, feat_fg_r)
        feat_out_x.scatter_(1, gather_index, feat_fg_x)
        return feat_out_r, feat_out_x, topk_idx
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                track_query_before = None,
                motion_weights = None,
                template_bbox = None
                ):

        feature_list = []
        for search_i in search:
            feature = self.extract_search_intermediate(
                template=template,
                search=search_i,
                track_query_before=track_query_before,
                template_bbox=template_bbox,
            )
            feature_list.append(feature)
            track_query_before = feature["track_query_before"]
        return self.forward_from_intermediates(
            feature_list,
            motion_weights=motion_weights,
            only_current=False,
        )

    def extract_search_intermediate(
            self, template, search, track_query_before=None, template_bbox=None):
        x, aux_dict, len_zx = self.backbone(
            z=template,
            x=search,
            track_query_before=track_query_before,
            template_bbox=template_bbox,
        )
        num_search_token = len_zx[1]

        B, N, _ = x.size()
        temp_r = x[:, :N // 2, :]
        temp_x = x[:, N // 2:, :]

        # DAM/CDA disabled. Use the backbone track-query tokens directly instead of
        # generating aligned query tokens through CDA.
        # boss_fea = torch.cat([temp_r[:, :1, :], temp_x[:, :1, :]], dim=1)
        # temp_r_str, temp_x_str = self.CDA(temp_r[:, 1:num_template_token // 2 + 1, :],
        #                                   temp_r[:, num_template_token // 2 + 1:num_template_token + 1, :],
        #                                   temp_x[:, 1:num_template_token // 2 + 1, :],
        #                                   temp_x[:, num_template_token // 2 + 1:num_template_token + 1, :],
        #                                   boss_fea)
        temp_r_str = temp_r[:, :1, :]
        temp_x_str = temp_x[:, :1, :]
        track_query = [temp_r_str.clone().detach(), temp_x_str.clone().detach()]

        return {
            "feat_last_r": temp_r[:, -num_search_token:, :],
            "feat_last_x": temp_x[:, -num_search_token:, :],
            "temp_r_str": temp_r_str,
            "temp_x_str": temp_x_str,
            "aux_dict": aux_dict,
            "track_query_before": track_query,
        }

    def forward_from_intermediates(self, feature_list, motion_weights=None, only_current=True):
        out_dict = []
        feat_last_r_list = [item["feat_last_r"] for item in feature_list]
        feat_last_x_list = [item["feat_last_x"] for item in feature_list]
        temp_r_str_list = [item["temp_r_str"] for item in feature_list]
        temp_x_str_list = [item["temp_x_str"] for item in feature_list]
        aux_dict_list = [item["aux_dict"] for item in feature_list]
        track_query_list = [item["track_query_before"] for item in feature_list]

        should_run_forward_topk = (
            self.temporal_attention is not None
            and self.use_forward_topk
            and len(feat_last_r_list) > 0
        )
        if should_run_forward_topk and self.temporal_attention_type not in ("fused", "fused_tav"):
            raise ValueError("FORWARD_TOPK requires TEMPORAL.TYPE to be 'fused' or 'fused_tav'.")

        if should_run_forward_topk and self.use_patch_selection:
            precomputed_topk_idx_list = self._select_motion_foreground_indices(
                feat_last_r_list,
                feat_last_x_list,
                temp_r_str_list,
                temp_x_str_list,
                motion_weights=motion_weights,
            )
        elif should_run_forward_topk:
            precomputed_topk_idx_list = []
            for feat_last_r in feat_last_r_list:
                batch_size, num_tokens, _ = feat_last_r.shape
                all_patch_indices = torch.arange(
                    num_tokens, device=feat_last_r.device, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)
                precomputed_topk_idx_list.append(all_patch_indices)
        else:
            precomputed_topk_idx_list = [None] * len(feat_last_r_list)
        css_feat_last_r_list = []
        css_feat_last_x_list = []
        routed_topk_idx_list = []
        for feat_last_r, feat_last_x, topk_idx in zip(
                feat_last_r_list, feat_last_x_list, precomputed_topk_idx_list):
            # CSS enhancement ablation: keep top-k routing for temporal attention,
            # but pass the original search tokens forward unchanged.
            css_feat_last_r_list.append(feat_last_r)
            css_feat_last_x_list.append(feat_last_x)
            routed_topk_idx_list.append(topk_idx)

        if should_run_forward_topk:
            if not hasattr(self.temporal_attention, "forward_topk"):
                raise ValueError("Configured temporal attention does not implement forward_topk().")
            feat_last_r_seq = torch.stack(css_feat_last_r_list, dim=0)
            feat_last_x_seq = torch.stack(css_feat_last_x_list, dim=0)
            topk_idx_seq = torch.stack(routed_topk_idx_list, dim=0)
            feat_last_r_seq, feat_last_x_seq = self.temporal_attention.forward_topk(
                feat_last_r_seq, feat_last_x_seq, topk_idx_seq
            )
            css_feat_last_r_list = [feat_last_r_seq[i] for i in range(len(css_feat_last_r_list))]
            css_feat_last_x_list = [feat_last_x_seq[i] for i in range(len(css_feat_last_x_list))]

        output_indices = [len(css_feat_last_r_list) - 1] if only_current else range(len(css_feat_last_r_list))
        for out_idx in output_indices:
            feat_last_r = css_feat_last_r_list[out_idx]
            feat_last_x = css_feat_last_x_list[out_idx]
            aux_dict = aux_dict_list[out_idx]
            track_query_before = track_query_list[out_idx]
            feat_last = torch.cat([feat_last_r, feat_last_x], dim=-1)
            out = self.forward_head(feat_last, None)

            out.update(aux_dict)
            out['track_query_before'] = track_query_before
            out['backbone_feat'] = feat_last
            out_dict.append(out)
        return out_dict

    def _tokens_to_feature_map(self, tokens):
        B, L, C = tokens.shape
        feat_sz = int(math.sqrt(L))
        if feat_sz * feat_sz != L:
            raise ValueError("Search token length must be square, got {}".format(L))
        return tokens.transpose(1, 2).contiguous().view(B, C, feat_sz, feat_sz)

    @staticmethod
    def _feature_map_to_tokens(feat):
        return feat.flatten(2).transpose(1, 2).contiguous()

    def forward_head(self, cat_feature ,gt_score_map=None):
        opt = (cat_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        opt_feat = self.decode_fuse_search(opt_feat)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_tsmtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('TSMTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            cross_loc=cfg.MODEL.BACKBONE.CROSS_LOC,
                                            group_fusion=cfg.MODEL.BACKBONE.GROUP_FUSION,
                                            )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = TSMTrack(
        backbone,
        box_head,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if training and (
            'OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        if 'DropTrack' in cfg.MODEL.PRETRAIN_FILE:
            for k, v in checkpoint["net"].items():
                if k in ['box_head.conv1_ctr.0.weight', 'box_head.conv1_offset.0.weight',
                         'box_head.conv1_size.0.weight']:
                    v = v
                elif 'pos_embed_x' in k:
                    v = resize_pos_embed(v, 16, 16) + checkpoint["net"]['backbone.temporal_pos_embed_x']
                elif 'pos_embed_z' in k:
                    v = resize_pos_embed(v, 8, 8) + checkpoint["net"]['backbone.temporal_pos_embed_z']
                else:
                    v = v
                param_dict_rgbt[k] = v

        else:
            param_dict_rgbt = checkpoint["net"]

        missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys', missing_keys, unexpected_keys)
    return model

def resize_pos_embed(posemb, hight, width):
    posemb_grid = posemb[0, :]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print(
        'Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight,
                                                                                               width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    return posemb_grid
