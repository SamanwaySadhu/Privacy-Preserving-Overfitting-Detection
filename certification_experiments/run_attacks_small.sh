#!/bin/bash
cd "$(dirname "$0")"
python -u attack.py --task simplecnn --stat ratio --models faulty_eval0 faulty_eval1 --iters 200 --lr 1e-3 > logs/attack_simplecnn_ratio.log 2>&1
python -u attack.py --task simplecnn --stat log --models faulty_eval0 faulty_eval1 --iters 200 --lr 1e-3 > logs/attack_simplecnn_log.log 2>&1
python -u attack.py --task lenet5 --stat ratio --models faulty_eval0 faulty_eval1 --iters 200 --lr 1e-3 > logs/attack_lenet5_ratio.log 2>&1
python -u attack.py --task lenet5 --stat log --models faulty_eval0 faulty_eval1 --iters 200 --lr 1e-3 > logs/attack_lenet5_log.log 2>&1
