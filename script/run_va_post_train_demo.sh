#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

NGPU=1 CONFIG_NAME='demo_train' bash script/run_va_posttrain.sh