# Computing environment

The frozen manuscript analyses were recorded with the following runtime configuration.

## Core runtime

- Operating system: Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python: 3.12.7
- Device: CPU only
- Network calls during frozen P9/P10 modeling stages: 0

## Model-development stack

- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.2
- scikit-learn 1.9.0
- joblib 1.4.2
- threadpoolctl 3.5.0

The logistic-regression implementation used `sklearn.linear_model.LogisticRegression` with the `lbfgs` solver, an intercept, no class/sample weighting, `max_iter=5000`, and `tol=1e-8`. The frozen C grid was `{0.01, 0.1, 1, 10, 100}`.

The modeling runtime fixed these thread environment variables to one thread:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## Visual-feature stack

- `open_clip_torch` 3.3.0
- PyTorch 2.13.0+cpu
- NumPy 1.26.4
- Pillow 10.4.0
- OpenCLIP model: `ViT-B-32`
- Pretrained tag: `laion2b_s34b_b79k`
- Device: CPU
- Batch size: 16
- Autocast: disabled
- Stochastic augmentation: disabled

The frozen OpenCLIP preprocessing used 224-pixel shortest-side resize, bicubic interpolation, center crop to 224 × 224, and the checkpoint-specific mean/std values encoded in `config/modeling/p9_visual_feature_contract.json`.

## Hardware detail

The frozen provenance records CPU-only execution but do not record a specific processor model or RAM capacity. Those details are therefore intentionally not invented in this repository.
