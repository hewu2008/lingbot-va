# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_zerith_cfg import va_zerith_cfg
import os

va_zerith_train_cfg = EasyDict(__name__='Config: VA Zerith train')
va_zerith_train_cfg.update(va_zerith_cfg)

va_zerith_train_cfg.dataset_path = '/home/jszn/hewu/dataset/lerobot_put_screw_into_the_hole_20260519'
va_zerith_train_cfg.empty_emb_path = os.path.join(va_zerith_train_cfg.dataset_path, 'empty_emb.pt')
va_zerith_train_cfg.enable_wandb = True
va_zerith_train_cfg.load_worker = 16
va_zerith_train_cfg.save_interval = 50
va_zerith_train_cfg.gc_interval = 50
va_zerith_train_cfg.cfg_prob = 0.1

# Training parameters
va_zerith_train_cfg.learning_rate = 1e-4
va_zerith_train_cfg.beta1 = 0.9
va_zerith_train_cfg.beta2 = 0.95
va_zerith_train_cfg.weight_decay = 1e-1
va_zerith_train_cfg.warmup_steps = 10
va_zerith_train_cfg.batch_size = 1 
va_zerith_train_cfg.gradient_accumulation_steps = 8
va_zerith_train_cfg.num_steps = 2000