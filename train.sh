#!/bin/sh
set -e
# scan
python train.py --config='configs/config_scancls.yaml'

echo 'The script was executed successfully'