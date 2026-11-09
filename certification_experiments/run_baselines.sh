#!/bin/bash
cd "$(dirname "$0")"
for t in 2fc simplecnn lenet5; do python -u baseline.py --task $t --k-cal 3 --k-eval 3 --n-seeds 30 > logs/baseline_$t.log 2>&1; done
python -u baseline.py --task resnet18 --k-cal 2 --k-eval 2 --n-seeds 20 > logs/baseline_resnet18.log 2>&1
