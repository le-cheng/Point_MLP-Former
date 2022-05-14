#!/bin/sh
set -e

python test.py --resume 2022-05-06/best_model.pth --config='configs/config_scancls.yaml' 

echo 'The script was executed successfully'