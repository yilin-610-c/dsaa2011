# Supplement Experiments Guide

## 1. Project Architecture

This project follows a script-oriented training pipeline:

- `models/resnet_variants.py`
  - Defines model families and `build_model(model_name)`.
  - Includes baseline and residual variants:
    - `plain_cnn`
    - `resnet18_optb`
    - `resnet18_opta`
    - `preact_resnet18`
    - `wide_resnet14`
    - `resnet34_optb` (supplement)
    - `resnet50_bottleneck` (supplement)
- `experiments/data_utils.py`
  - Handles data loading, split logic, and dataloaders.
  - Adds **offline tensor cache** to avoid decoding PNG bytes every batch.
- `experiments/train_eval.py`
  - Trains one model for one seed.
  - Saves run artifacts: history CSV, checkpoint, metrics JSON.
- `experiments/run_experiments.py`
  - Launches batch experiments over multiple models and seeds.
- `experiments/analyze_results.py`
  - Builds summary table and plots from `results/run_matrix.csv`.

## 2. Data Flow and Caching

Original bottleneck:
- `data.parquet` stores image bytes.
- Decoding PNG bytes in every `__getitem__` is expensive.

Current accelerated flow:
1. Build cache once from parquet:
   - `results/cache/fashion_tensor_cache.pt`
   - Stores:
     - `images_uint8`: `(70000, 28, 28)`
     - `labels`
     - `split_flags` (`train/test`)
2. Training loads tensors directly from cache.
3. Augmentation and normalization are applied on tensors.

This removes repeated PNG decode overhead from every epoch.

## 3. Implemented Models

Main set:
- M1: Plain-CNN
- M2: ResNet-18-OptB
- M3: ResNet-18-OptA
- M4: PreAct-ResNet-18
- M5: Wide-ResNet-14

Supplement set:
- M6: ResNet-34-OptB (depth comparison)
- M7: ResNet-50-Bottleneck (parameter-efficiency comparison)

## 4. Training Protocol

Shared settings for fair comparison:
- Optimizer: SGD (Nesterov)
- Learning rate: 0.05
- Scheduler: CosineAnnealingLR
- Weight decay: 5e-4
- Batch size: 256
- Epochs: 8 (for final comparison)
- Seeds: `42`, `3407`, `2025` (or seed `42` for quick supplement verification)

## 5. Command-Line Usage

### 5.1 Build cache once (recommended before all runs)

```bash
python -m experiments.build_cache \
  --parquet-path data.parquet \
  --cache-path results/cache/fashion_tensor_cache.pt
```

### 5.2 Run one sanity check

```bash
python -m experiments.train_eval \
  --run-id SANITY001 \
  --model-name resnet18_optb \
  --seed 42 \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 4 \
  --parquet-path data.parquet \
  --cache-path results/cache/fashion_tensor_cache.pt \
  --output-dir results/sanity \
  --use-augmentation \
  --show-epoch-progress
```

### 5.3 Run supplement models only (quick: seed 42)

```bash
python -m experiments.run_experiments \
  --supplement-only \
  --include-supplement \
  --seeds 42 \
  --epochs 8 \
  --batch-size 256 \
  --num-workers 4 \
  --use-augmentation \
  --cache-path results/cache/fashion_tensor_cache.pt \
  --show-epoch-progress
```

### 5.4 Run full suite with supplement models

```bash
python -m experiments.run_experiments \
  --include-wide \
  --include-supplement \
  --seeds 42 3407 2025 \
  --epochs 8 \
  --batch-size 256 \
  --num-workers 4 \
  --use-augmentation \
  --cache-path results/cache/fashion_tensor_cache.pt \
  --show-epoch-progress
```

### 5.5 Re-generate summary tables and plots

```bash
python -m experiments.analyze_results \
  --run-matrix results/run_matrix.csv \
  --output-dir results
```

## 6. Performance Troubleshooting

If training is too slow:
- Confirm GPU is visible:
  - `torch.cuda.is_available()` must be `True`.
- Use cache:
  - Make sure `--cache-path` points to an existing cache file.
- Increase dataloader workers on your machine:
  - Start with `--num-workers 4`, then test `8`.
- Keep `batch-size` at 256 (or increase if GPU memory allows).

If you need a quick trend check:
- Run supplement with `--seeds 42 --epochs 3` first, then scale up.
