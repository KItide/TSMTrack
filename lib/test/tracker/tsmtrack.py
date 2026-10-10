import math
import numpy as np
from lib.models.tsmtrack import build_tsmtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class TSMTrack(BaseTracker):
    _sequence_id_counter = 0
    def __init__(self, params):
        super(TSMTrack, self).__init__(params)
        network = build_tsmtrack(params.cfg, training=False)

        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = checkpoint['net']

        # Older checkpoints can omit the dynamic-layer selector. It is retained
        # in the model only for checkpoint compatibility and shift() is not used
        # by the current 3/4/5 group-fusion forward path. Allow exactly these
        # parameters to be absent while keeping strict validation for all active
        # model parameters and all unexpected checkpoint entries.
        allowed_missing_keys = {
            'backbone.moe_proj_weights',
            'backbone.moe_mlp.0.weight',
            'backbone.moe_mlp.0.bias',
            'backbone.moe_mlp.2.weight',
            'backbone.moe_mlp.2.bias',
            'backbone.moe_mlp.4.weight',
            'backbone.moe_mlp.4.bias',
            'backbone.moe_fc.weight',
            'backbone.moe_fc.bias',
        }
        incompatible = network.load_state_dict(state_dict, strict=False)
        disallowed_missing = set(incompatible.missing_keys) - allowed_missing_keys
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'Checkpoint is incompatible with TSMTrack: missing_keys={}, '
                'unexpected_keys={}'.format(
                    sorted(disallowed_missing), sorted(incompatible.unexpected_keys)
                )
            )
        if incompatible.missing_keys:
            print('Initialize unused legacy MoE parameters from the model: {}'.format(
                sorted(incompatible.missing_keys)
            ))
        print('Load pretrained model from: ' + self.params.checkpoint)
        depth = 12
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM(mean= self.cfg.DATA.MEAN, std = self.cfg.DATA.STD)
        self.state = None
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        temporal_cfg = getattr(self.cfg.MODEL, "TEMPORAL", None)
        temporal_enabled = bool(getattr(temporal_cfg, "ENABLE", False)) if temporal_cfg is not None else False
        forward_topk = bool(getattr(temporal_cfg, "FORWARD_TOPK", False)) if temporal_cfg is not None else False
        self.use_temporal_attention = temporal_enabled and bool(forward_topk)
        self.num_search = getattr(self.cfg.DATA.SEARCH, "NUMBER", 1) if self.use_temporal_attention else 1
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS
        self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD
        self.temporal_enter_threshold = float(getattr(self.cfg.TEST, "TEMPORAL_ENTER_THRESHOLD", 0.0))
        self.use_amp = bool(getattr(self.cfg.TEST, "AMP", False))

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.sequence_id = TSMTrack._sequence_id_counter
        TSMTrack._sequence_id_counter += 1

    def initialize(self, image, info: dict):

        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        
        template = self.preprocessor.process(z_patch_arr)
        init_template_bbox = self.transform_bbox_to_crop(
            info['init_bbox'],
            resize_factor,
            template.device,
            crop_type='template',
        )
        with torch.no_grad():
            self.z_dict = [template]* self.num_template
            self.template_bbox = [init_template_bbox]* self.num_template

        self.track_query_before = None
        self.search_feature_buffer = []

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            if self.use_temporal_attention:
                current_center = self._state_center_norm(W, H)
                current_feature = self.network.extract_search_intermediate(
                    template=self.z_dict,
                    search=search,
                    track_query_before=self.track_query_before,
                    template_bbox=self.template_bbox,
                )
                feature_list, motion_weights = self._build_feature_window(current_feature, current_center)
                out_dict = self.network.forward_from_intermediates(
                    feature_list,
                    motion_weights=motion_weights,
                    only_current=True,
                )
            else:
                current_feature = None
                x_dict = [search]
                motion_weights = None
                # merge the template and the search
                # run the transformer
                if self.track_query_before != None:
                    out_dict = self.network.forward(
                        template=self.z_dict,
                        search=x_dict,
                        track_query_before=self.track_query_before,
                        motion_weights=motion_weights,
                        template_bbox=self.template_bbox)
                else:
                    out_dict = self.network.forward(
                        template=self.z_dict,
                        search=x_dict,
                        motion_weights=motion_weights,
                        template_bbox=self.template_bbox)

        out = out_dict[-1]
        self.track_query_before  = out['track_query_before']

        pred_score_map = out['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf_score = None
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):

                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                dynamic_template_bbox = self.transform_bbox_to_crop(
                    self.state,
                    resize_factor,
                    template.device,
                    crop_type='template',
                )
                self.z_dict.append(template)
                self.template_bbox.append(dynamic_template_bbox)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)
                    self.template_bbox.pop(1)

        if self.use_temporal_attention:
            history_score = float(conf_score.item()) if conf_score is not None else None
            self._update_search_history(current_feature, history_score, self._state_center_norm(W, H))

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            if conf_score != None:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":conf_score.cpu().numpy()[0][0]}
            else:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":None
                        }
        else:
            if conf_score != None:
                return {"target_bbox": self.state,
                        "best_score":conf_score.cpu().numpy()[0][0]
                        }
            else:
                return {"target_bbox": self.state,
                        "best_score":None
                        }

    def _build_feature_window(self, current_feature, current_center):
        if self.num_search <= 1:
            return [current_feature], None

        features = [item["feature"] for item in self.search_feature_buffer] + [current_feature]
        centers = [item["center"] for item in self.search_feature_buffer] + [current_center]
        device = current_feature["feat_last_r"].device
        return features, self._build_motion_weights_from_centers(centers, device)

    def _update_search_history(self, feature, score, center):
        if self.num_search <= 1 or score is None or not (score > self.temporal_enter_threshold):
            return

        self.search_feature_buffer.append({
            "feature": self._detach_feature(feature),
            "score": score,
            "center": center,
        })
        if len(self.search_feature_buffer) > self.num_search - 1:
            self.search_feature_buffer.pop(0)

    @staticmethod
    def _detach_feature(feature):
        detached = {}
        for key, value in feature.items():
            if torch.is_tensor(value):
                detached[key] = value.detach()
            elif isinstance(value, (list, tuple)):
                detached[key] = [
                    item.detach() if torch.is_tensor(item) else item
                    for item in value
                ]
            else:
                detached[key] = value
        return detached

    @staticmethod
    def _build_motion_weights_from_centers(centers, device):
        weights = torch.zeros(len(centers), 1, device=device, dtype=torch.float32)
        for i in range(1, len(centers)):
            prev = torch.tensor(centers[i - 1], device=device, dtype=torch.float32)
            curr = torch.tensor(centers[i], device=device, dtype=torch.float32)
            weights[i, 0] = (torch.norm(curr - prev) / (2 ** 0.5)).clamp(0.0, 1.0)
        return weights

    def _state_center_norm(self, width, height):
        x, y, w, h = self.state
        return [(x + 0.5 * w) / max(float(width), 1.0),
                (y + 0.5 * h) / max(float(height), 1.0)]

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TSMTrack
