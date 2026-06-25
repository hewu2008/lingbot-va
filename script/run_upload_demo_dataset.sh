#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

# hf repo create hewu2008/pick-n-place-sq-lerobot-v21 --type dataset
hf upload hewu2008/pick-n-place-sq-lerobot-v21 /home/jszn/hewu/dataset/pick-n-place-sq-lerobot-v21 --repo-type dataset