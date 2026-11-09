# Certification of Machine Learning Models via Directional Sharpness
Artifacts accompanying the paper: *Certification of Machine Learning Models via Directional Sharpness*

## Layout

- `df_utils/` — directional-sharpness probing utilities.
- `experiments/` — runnable experiment scripts (training + probing).
- `grid_train/` — grid-search training scripts (`NN.py`, `train.py`).


## Requirements

- Python 3.11
- Core packages:
  - `torch`, `torchvision`
  - `numpy`, `pandas`, `tqdm`
  - `matplotlib`
  - `transformers` (only for the TinyMem-related experiments)


## Setup

```bash
pip install -r requirements.txt
```

## Quick tests

The following scripts can be run without performing a full training.

1. End-to-end smoke test of SAM and SGD training and directional-sharpness measurement:

   ```bash
   python experiments/light_test.py
   ```

   This script trains two VGG-16-BN models on a small CIFAR-10 subset — one with SAM, one with SGD — and measures the directional sharpness of each. Training is deliberately shortened (1 epoch on a 2048-sample subset) to keep it light. 

2. Runtime benchmark (no training required):

   ```bash
   python experiments/speed.py
   ```

   This script corresponds to the experiment in Section 5.2. Benchmarks the wall-clock cost of the directional-sharpness probe against test-accuracy evaluation and gradient-ascent static sharpness, using a VGG-16-BN on CIFAR-10. A randomly initialized model suffices, since only runtime is measured.



## Distinguishability Experiments

Each script under `experiments/` trains (or loads) a benign and a faulty model, evaluates test accuracy, and probes both with directional sharpness. Output goes to `artifacts/<exp>_<timestamp>/`.

### Experiments without external dependencies

- `probe_noisy_pair.py` — clean CIFAR-10 vs random label corruption, both stopped at a matched training accuracy.
- `probe_spurious_pair.py` — clean CIFAR-10 vs a label-colored top-left patch overlaid with probability `spur_prob`.
- `probe_quantization_pair.py` — SAM-trained benign vs SGD-trained faulty, probed at pre-quantization weights and reported alongside post-quantization test accuracy.
- `probe_small_test_pair.py` — same clean/noisy pair as the noisy experiment, plus a 100-sample test subset crafted so both models score the same observed accuracy.
- `probe_mia.py` — black-box membership-inference attack on a single VGG-16-BN trained on CIFAR-10. 
- `speed.py` — wall-clock benchmark: test-accuracy eval vs the directional-sharpness probe (T steps, batch N) vs the gradient-ascent static-sharpness.

### Experiments with external dependencies


#### `probe_backdoor_pair.py` — requires [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox)

1. Clone BackdoorBox and install all dependencies; follow `tests/test_BadNets.py` to train one benign and one backdoor ResNet-18 on CIFAR-10. BackdoorBox writes raw `model.state_dict()` to `experiments/<exp_name>_<timestamp>/ckpt_epoch_N.pth`.
2. Probe both checkpoints:
   ```bash
   python -u experiments/probe_backdoor_pair.py \
       --clean-ckpt    /path/to/benign_exp/ckpt_epoch_N.pth \
       --backdoor-ckpt /path/to/badnet_exp/ckpt_epoch_M.pth \
       --backdoorbox-root /path/to/BackdoorBox \
       --no-input-normalize \
       --probe-steps 10 --probe-batch-size 16
   ```

#### `probe_tinymem_pair.py` — requires TinyMem from [memorization](https://github.com/msakarvadia/memorization)

1. Clone the memorization repo and install all dependencies; generate the data cache with `src/data/old_data.py`.
2. Train a clean and a faulty (`--backdoor 1` or `--duplicate 1`) TinyMem model with `src/memorization_in_toy_models.py`. Checkpoints land at `<ckpt_dir>/{n_layers}_layer_{epoch}_epoch.pth` and are wrapped dicts (`{"model_state_dict": …, "optimizer_state_dict": …, …}`).
3. Probe both checkpoints, passing the same hyperparameters used for training so the TinyMem config and data path resolve correctly:
   ```bash
   python -u experiments/probe_tinymem_pair.py \
       --mem-root /path/to/memorization \
       --clean-ckpt  /path/to/clean/4_layer_99_epoch.pth \
       --faulty-ckpt /path/to/faulty/4_layer_99_epoch.pth \
       --faulty-kind backdoor \
       --data-name mult --n-layers 4 --n-embed 128 --max-ctx 150 \
       --num-7 20000 --num-2 3000 --num-3 3000 --num-4 3000 --num-5 3000 \
       --length 20 --num-test 1000 --num-noise 1000 --seed 0 \
       --probe-steps 10 --probe-batch-size 8
   ```

#### `probe_sapa_resnet_cifar10.py` — requires [SAPA](https://github.com/PengfeiHePower/SAPA)

This script imports SAPA's `forest` library and trains a benign ResNet-18 (SGD) and a faulty ResNet-18 with SAPA's sharpness-aware hidden-trigger backdoor and probe both.

1. Clone SAPA next to the current repo and install its extra dependency and install its dependencies:
   ```bash
   git clone https://github.com/PengfeiHePower/SAPA.git ../SAPA
   ```
   If the SAPA is located in a different folder, pass `--sharp-agent-root /path/to/SAPA/Sharp-Agent`.
2. Train both models and probe them (checkpoints are saved under `artifacts/sapa_resnet_df_<timestamp>/pair_000/`):
   ```bash
   python -u experiments/probe_sapa_resnet_cifar10.py \
       --net ResNet18 --pair-count 1 \
       --sweep-probe-steps 10 --sweep-probe-step-batch-size 8
   ```
