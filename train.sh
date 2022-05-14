#!/bin/sh
set -e
# now=$(date +"%Y%m%d_%H%M%S")

# curvenet
# python3 train.py --config='configs/config.yaml' --exp_name='modelnet_cls_pointgraph_nogggroupsg' --model=pointgraph --seed=6666

# scan
python train.py --config='configs/config_scancls.yaml' 
python train.py --config='configs/config_scancls.yaml'
python train.py --config='configs/config_scancls.yaml'

echo 'The script was executed successfully'