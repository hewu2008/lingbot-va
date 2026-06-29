# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_zerith_cfg = EasyDict(__name__='Config: VA Zerith')
va_zerith_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_zerith_cfg.wan22_pretrained_model_name_or_path = "/home/jszn/hewu/model_zoo/lingbot-va-base"

va_zerith_cfg.attn_window = 30
va_zerith_cfg.frame_chunk_size = 4
va_zerith_cfg.env_type = 'none'

va_zerith_cfg.height = 256
va_zerith_cfg.width = 256
va_zerith_cfg.action_dim = 30
va_zerith_cfg.action_per_frame = 8
va_zerith_cfg.obs_cam_keys = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_zerith_cfg.guidance_scale = 5
va_zerith_cfg.action_guidance_scale = 1

va_zerith_cfg.num_inference_steps = 5
va_zerith_cfg.video_exec_step = -1
va_zerith_cfg.action_num_inference_steps = 10

va_zerith_cfg.snr_shift = 5.0
va_zerith_cfg.action_snr_shift = 1.0

# Zerith 机器人动作映射到标准30维格式:
# HDF5 action合并顺序: [arm(14) + base(2) + effector(2) + end(14) + head(2) + waist(3)] = 37维
# HDF5 索引:  arm=[0-13], base=[14-15], effector=[16-17], end=[18-31], head=[32-33], waist=[34-36]
#
# 标准30维格式 (action_dim=30): 
#   0-6: 左臂 EEF (7)
#   7-13: 右臂 EEF (7)  
#   14-20: 左臂关节 (7)
#   21-27: 右臂关节 (7)
#   28: 左夹爪 (1)
#   29: 右夹爪 (1)
#
# 注意: used_action_channel_ids 是在30D对齐空间中的索引(0-29), 不是HDF5索引!
# 
# 当前使用: 双臂EEF (0-13) + 夹爪 (28, 29) = 16维
va_zerith_cfg.used_action_channel_ids = list(range(0, 14)) + [28, 29]
inverse_used_action_channel_ids = [len(va_zerith_cfg.used_action_channel_ids)
                                   ] * va_zerith_cfg.action_dim
for i, j in enumerate(va_zerith_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_zerith_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_zerith_cfg.action_norm_method = 'quantiles'
# norm_stat 需要根据实际数据计算
# 当前使用: 双臂EEF (14维, 索引0-13) + 夹爪 (2维, 索引28-29)
# 其余14维 (索引14-27, 即关节) 填充为0
va_zerith_cfg.norm_stat = {
    "q01": (
        [-0.5, -0.5, -0.5, -3.14, -0.5, -0.5, -0.5,  # 左臂 EEF (x, y, z, rx, ry, rz, rw)
         -0.5, -0.5, -0.5, -3.14, -0.5, -0.5, -0.5,  # 右臂 EEF
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 关节 (14维, 待替换)
         0.0, 0.0]  # 夹爪
    ),
    "q99": (
        [0.5, 0.5, 0.5, 3.14, 0.5, 0.5, 0.5,  # 左臂 EEF
         0.5, 0.5, 0.5, 3.14, 0.5, 0.5, 0.5,  # 右臂 EEF
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 关节 (14维, 待替换)
         100.0, 100.0]  # 夹爪
    ),
}