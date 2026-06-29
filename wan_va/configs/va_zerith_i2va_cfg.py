# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_zerith_cfg import va_zerith_cfg

va_zerith_i2va_cfg = EasyDict(__name__='Config: VA Zerith i2va')
va_zerith_i2va_cfg.update(va_zerith_cfg)

va_zerith_i2va_cfg.input_img_path = 'example/zerith'
va_zerith_i2va_cfg.num_chunks_to_infer = 10
va_zerith_i2va_cfg.prompt = 'Put the screw into the hole'
va_zerith_i2va_cfg.infer_mode = 'i2va'