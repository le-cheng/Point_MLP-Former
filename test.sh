#!/bin/sh
set -e

python test.py --resume pretrain/best_model.pth --config='configs/config_scancls.yaml'  --model=model

# echo 'The script was executed successfully'