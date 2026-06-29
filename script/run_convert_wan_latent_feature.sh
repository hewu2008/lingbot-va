#!/bin/bash

export PYTHONPATH=`pwd`

python tools/convert_wan_latent_feature.py \
    --model_path /home/jszn/hewu/model_zoo/lingbot-va-base \
    --dataset_dir /home/jszn/hewu/dataset/lerobot_put_screw_into_the_hole_20260519 \
    --dtype bfloat16