#!/bin/bash

# Action space (37)
# arm/position 14 左右臂关节各7维
# base/velocity 2 底盘轮速（差速驱动）
# effector/position 2 左右夹爪各1维
# end/position 14 左右臂EEF各7维
# head/position 2 头部pan + tilt
# waist/position 3 腰部yaw + pitch + z（上下）

# Observation space (51)
# arm/position 14 
# arm.torque 14
# arm.velocity 14
# base.velocity 2
# effector.position 2
# head.position 2
# waist.position 3 


python tools/convert_zerith_record_data.py \
    --hdf5_dir /home/jszn/hewu/dataset/put_screw_into_the_hole_20260519 \
    --output_root /home/jszn/hewu/dataset/ \
    --repo_id lerobot_put_screw_into_the_hole_20260519 \
    --robot_type Zerith_H1