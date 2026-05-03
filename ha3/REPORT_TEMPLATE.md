# HA3 Report Template: Diffusion Model Experiments on MNIST

## 1. Summary

This report studies a lightweight DDPM-style diffusion baseline on MNIST using the standard epsilon-prediction objective. Four checkpoints are trained with an equal optimization budget, and each checkpoint is evaluated with both DDPM and DDIM sampling, producing eight total configurations.

The experiment varies:

- noise schedule: scaled linear vs cosine
- training diffusion length: `training_T = 200` vs `training_T = 1000`
- sampler and sampling steps: DDPM with full reverse steps vs DDIM with 50 deterministic steps

All other hyperparameters are fixed so that the comparison focuses on schedule, `training_T`, and sampler/`sampling_steps`.

## 2. Methodology

The model predicts the noise term epsilon added in the forward diffusion process:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon,
epsilon ~ N(0, I)
```

The scaled linear beta schedule uses:

```text
beta_start = 1e-4 * 1000 / training_T
beta_end   = 0.02  * 1000 / training_T
```

The cosine schedule is used as the second schedule. For every schedule and `training_T`, `alpha_bar_T` is recorded and the terminal distribution is checked by visualizing and summarizing `x_T`.

DDPM uses the full stochastic reverse process with:

```text
sampling_steps = training_T
```

DDIM uses deterministic accelerated sampling with:

```text
sampling_steps = 50
```

Therefore, DDPM vs DDIM is interpreted as a practical speed-quality comparison, not a purely isolated sampler-only comparison.

## 3. Experimental Configurations

| Schedule | `training_T` | Sampler | `sampling_steps` |
|---|---:|---|---:|
| scaled linear | 200 | DDPM | 200 |
| scaled linear | 200 | DDIM | 50 |
| scaled linear | 1000 | DDPM | 1000 |
| scaled linear | 1000 | DDIM | 50 |
| cosine | 200 | DDPM | 200 |
| cosine | 200 | DDIM | 50 |
| cosine | 1000 | DDPM | 1000 |
| cosine | 1000 | DDIM | 50 |

Training budget for all four checkpoints:

- `max_train_steps = 10000`
- batch size `128`
- AdamW optimizer
- learning rate `2e-4`
- seed `42`

The dataloader is cycled until `global_step == max_train_steps`, so all checkpoints receive the same number of parameter updates.

## 4. Evaluation Metrics

A separate MNIST classifier is trained as the evaluator. The classifier must reach at least 98% test accuracy before computing metrics.

The metrics are MNIST-specific proxy metrics:

- `MNIST-IS`: an IS-like score computed from the MNIST classifier softmax predictions.
- `Classifier Feature FID`: a FID-like feature distance computed from the MNIST classifier's penultimate feature embeddings.

These are not standard ImageNet Inception Score or standard Inception FID. This evaluator is used because ImageNet Inception features are not ideal for grayscale MNIST digits.

## 5. Results

Insert the table from:

```text
results/eval/results.csv
```

Required columns:

- schedule
- `training_T`
- sampler
- `sampling_steps`
- sample count
- runtime
- `MNIST-IS`
- `Classifier Feature FID`
- `alpha_bar_T`

## 6. Analysis

Discuss:

- how scaled linear and cosine schedules differ in `alpha_bar_t` behavior
- whether `training_T = 1000` improves quality over `training_T = 200`
- whether DDIM gives comparable quality with lower runtime
- how terminal `x_T` statistics support the Gaussian-noise assumption

## 7. Far-From-Origin Analysis

Using the best checkpoint, generate samples from:

```text
x_T = scale * z,    z ~ N(0, I)
scales = [1, 2, 4, 8]
```

The same base noise `z` is reused across scales. Large scales place `x_T` outside the typical region of the standard Gaussian prior, so the reverse process begins from an out-of-distribution terminal latent and image quality is expected to degrade.

Include grids from:

```text
results/far_from_origin/
```

## 8. Visualizations

Include:

- `results/diagnostics/alpha_bar_curves.png`
- terminal `x_T` grids from `results/diagnostics/`
- sample grids from `results/eval/`
- far-from-origin grids from `results/far_from_origin/`

