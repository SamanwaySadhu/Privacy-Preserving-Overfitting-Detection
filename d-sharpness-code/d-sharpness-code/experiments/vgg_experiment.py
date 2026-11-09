from __future__ import annotations

from pathlib import Path
import sys

GRID_DIR = Path(__file__).resolve().parents[1] / "grid_train"
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

from NN import NNGridConfig, run_grid


def main() -> None:
    cfg = NNGridConfig(
        out_root="./outputs/nn_grid",
        experiment_name="vgg_grid",
        append_csv_path=None,
        force_append=False,
        datasets=["cifar10"],
        models=["vgg13_bn"],
        seeds=[2025],
        optimizers=["sgd"],
        lrs=[0.1, 0.032, 0.01, 0.05],
        wds=[1e-4, 5e-4],
        batch_sizes=[32, 64, 128],
        dropouts=[0.0, 0.5, 0.25],
        label_smoothing_list=[0.05, 0.01],
        max_epochs=200,
        train_loss_threshold=0.02,
        num_workers=8,
        sam_rho=2.0,
        sam_adaptive=False,
        eval_microbatch=256,
        stop_no_improve_interval=20,
        stop_min_improve=0.01,
    )
    run_grid(cfg)


if __name__ == "__main__":
    main()
