#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f results/lenet5_attack_log.json ]; do sleep 10; done
python -u attack.py --task simplecnn --stat ratio --models faulty_eval0 faulty_eval1 --iters 800 --lr 1e-3 --tag _800 > logs/attack_simplecnn_ratio_800.log 2>&1
python -u attack.py --task 2fc --stat ratio --models faulty_eval0 faulty_eval1 faulty_eval2 --iters 400 --lr 1e-3 --tag _400 > logs/attack_2fc_ratio_400.log 2>&1
