# MNIST Diffusion HA3 Experiment Code

This directory implements the HA3 plan with a lightweight DDPM-style MNIST baseline.
Use the `resnet_hw` conda environment.

## Recommended Commands

Run a quick code smoke test:

```bash
conda run -n resnet_hw python -m mnist_diffusion.smoke_test
```

Create schedule diagnostics:

```bash
conda run -n resnet_hw python -m mnist_diffusion.diagnostics
```

Train the MNIST evaluator classifier. It must reach at least 98% test accuracy:

```bash
conda run -n resnet_hw python -m mnist_diffusion.train_evaluator
```

Train the four diffusion checkpoints with the same update budget:

```bash
conda run -n resnet_hw python -m mnist_diffusion.train --schedule scaled_linear --training-t 200 --max-train-steps 10000
conda run -n resnet_hw python -m mnist_diffusion.train --schedule scaled_linear --training-t 1000 --max-train-steps 10000
conda run -n resnet_hw python -m mnist_diffusion.train --schedule cosine --training-t 200 --max-train-steps 10000
conda run -n resnet_hw python -m mnist_diffusion.train --schedule cosine --training-t 1000 --max-train-steps 10000
```

Evaluate all eight configurations:

```bash
conda run -n resnet_hw python -m mnist_diffusion.sample_eval --num-samples 2000
```

Use `--num-samples 5000` if time permits, or `--num-samples 1000` as fallback.
Keep the sample count the same across all eight configurations.

Create the far-from-origin analysis:

```bash
conda run -n resnet_hw python -m mnist_diffusion.far_from_origin
```

## Outputs

- `results/diagnostics/alpha_bar_curves.png`
- `results/diagnostics/schedule_terminal_summary.csv`
- `results/checkpoints/*/model.pt`
- `results/evaluator/mnist_classifier.pt`
- `results/eval/results.csv`
- `results/eval/samples_*.png`
- `results/far_from_origin/far_origin_scale_*.png`

Metric names in the report should be `MNIST-IS` and `Classifier Feature FID`
or equivalent `IS-like` / `FID-like` wording, because the evaluator is a
separately trained MNIST classifier rather than ImageNet Inception.

