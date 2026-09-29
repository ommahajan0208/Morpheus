---
goal: Fast configurable CelebA 64x64 GAN comparison of DCGAN, WGAN-Clipping, and WGAN-GP
version: 1.1
date_created: 2026-09-29
last_updated: 2026-09-29
owner: Project team
status: Planned
tags: [feature, deep-learning, gan, celeba, pytorch, reproducibility, notebook]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Train and compare DCGAN, WGAN with critic weight clipping, and WGAN-GP on a configurable deterministic CelebA subset at 64x64. `main.ipynb` is the end-to-end entry point: Run All always reloads `config/run.yaml`, builds or reuses only the matching split/cache, executes the requested experiments through Python modules, then rebuilds the report artifacts. The initial configuration uses 5,000 images and targets a local NVIDIA RTX 4050 through PyTorch CUDA.

## 1. Requirements & Constraints

- **REQ-001**: Use CelebA aligned face images only; resize and center-crop every image to `3x64x64`, then normalize RGB values to `[-1, 1]`.
- **REQ-002**: Build exactly three core models: `dcgan`, `wgan_clip`, and `wgan_gp`.
- **REQ-003**: Create `config/run.yaml` as the single human-edited run configuration. Its initial values are `dataset.num_images: 5000`, `dataset.eval_images: 500`, `dataset.seed: 42`, and `performance.profile: rtx4050_fast`.
- **REQ-004**: Implement `src/config.py::load_run_config`. Every Run All execution of `main.ipynb` must call it afresh and validate `dataset.num_images` as an integer in `[501, available_image_count]`. No notebook cell may hard-code or retain a prior image-count value.
- **REQ-005**: Select exactly `dataset.num_images` valid CelebA filenames after lexical sorting and seed-42 sampling. Persist the manifest as `data/splits/celeba_n<num_images>_seed42.csv`; changing 5,000 to 10,000 therefore always creates/selects a distinct split.
- **REQ-006**: Put exactly `dataset.eval_images` images in the held-out evaluation partition and all remaining manifest images in training. Evaluation images must never enter the training loader.
- **REQ-007**: Use a common 64x64 generator architecture and comparable parameter scale for all three models: latent dimension `128`, generator/critic base channels `64`, and `tanh` generator output.
- **REQ-008**: Use an identical split, seed `[42]`, actual batch size, sample cadence, FID cadence, and 10,000-generator-update budget for all three models in Experiment 1. WGAN variants use `n_critic=5`; DCGAN uses one discriminator update.
- **REQ-009**: Use DCGAN BCE-with-logits non-saturating loss with Adam `lr=0.0002`, `betas=(0.5,0.999)`; use WGAN-Clipping RMSprop `lr=0.00005` with clipping `[-0.01,0.01]`; use WGAN-GP Adam `lr=0.0001`, `betas=(0.0,0.9)`, two-sided `lambda_gp=10`, and no critic BatchNorm.
- **REQ-010**: Calculate WGAN-GP in float32 with autograd enabled. Ordinary model passes use AMP; the gradient-penalty calculation does not.
- **REQ-011**: Use CUDA when available. The `rtx4050_fast` profile enables AMP, TF32 convolution/matmul, `cudnn.benchmark`, pin memory, non-blocking transfers, persistent DataLoader workers, `prefetch_factor=4`, and an initial requested batch size of 64. If OOM occurs, rerun all compared models at 32 and record that value in every configuration snapshot.
- **REQ-012**: Build one reusable uint8 64x64 RGB NumPy memmap cache at `data/cache/celeba64_n<num_images>_seed42_<manifest_sha256>/images.dat`. Reuse it only when image count, manifest hash, transform version, and resolution match; otherwise rebuild it. Normalize batches immediately before GPU transfer.
- **REQ-013**: Run `src/performance.py::benchmark_dataloader` over 100 batches during preflight and select the fastest valid worker count from `[0,2,4]`. Freeze the selected number for every model in the same comparison.
- **REQ-014**: Compute FID at generator updates 2,500, 5,000, 7,500, and 10,000 using the same held-out real images and an equal number of generated images. Cache real Inception features by manifest hash; evaluate generated batches in memory with `fid_batch_size: 128`, not by writing temporary images. Name the default result `FID-500`.
- **REQ-015**: Record critic input-gradient norm mean, standard deviation, p05, p50, and p95 every 100 critic updates. WGAN-GP reuses its existing penalty gradients; WGAN-Clipping runs a diagnostic with `create_graph=False` only at the interval.
- **REQ-016**: Record training seconds only from first training batch through final checkpoint. Exclude data download, memmap build, FID computation, figure/report generation, and notebook execution; record those durations separately in `pipeline_timings.json`.
- **REQ-017**: Save only the latest checkpoint, best-FID checkpoint, and checkpoints at the FID cadence. Save sample grids at the same cadence. Store each configuration snapshot, metrics CSV, artifacts, and metadata under `artifacts/runs/n<num_images>/<experiment>/<model_or_lambda>/seed_42/`.
- **REQ-018**: Run the WGAN-GP lambda ablation at `{1,10,50}` for 3,000 generator updates. Reuse the first 3,000 updates of the main lambda=10 run; run only lambda=1 and lambda=50 separately.
- **REQ-019**: `main.ipynb` is a thin orchestrator only. It imports public functions from `src/` and must not contain model definitions, transforms, losses, training loops, FID logic, or plotting logic.
- **REQ-020**: The notebook sections, in order, are: environment and preflight; load/display current configuration; build/validate manifest and cache; smoke test; main experiment; lambda ablation; analysis/report; artifact summary. Each section calls a documented function in `src/pipeline.py`.
- **CON-001**: Initial development and results remain at 64x64. Do not add 128x128, conditioning, other datasets, penalties, or architecture sweeps.
- **GUD-001**: After changing `dataset.num_images` in `config/run.yaml`, the user runs all cells in `main.ipynb`; only configuration-derived paths are used, so no 5k split, cache, FID reference, or artifact can be accidentally reused for a 10k run.
- **PAT-001**: Treat critic-loss magnitude as a within-model diagnostic. Compare models through FID, samples, elapsed training time, and recorded stability evidence.

## 2. Implementation Steps

### Implementation Phase 1

- **GOAL-001**: Build the config-driven, cached CUDA pipeline and notebook orchestration.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Create `config/run.yaml` and `src/config.py::{RunConfig,load_run_config,write_config_snapshot}`. Include dataset, models, training, evaluation, performance, output, and experiment keys. Reject unknown keys and invalid image counts. |  |  |
| TASK-002 | Implement `src/data/celeba.py::{build_manifest,build_dataloaders}`. Derive every split path from `config.dataset.num_images`; never read a hard-coded `celeba_5k` path. |  |  |
| TASK-003 | Implement `src/data/cache.py::build_or_open_uint8_memmap_cache` and metadata validation. It performs crop/resize once per matching manifest, records cache-build time, and serves training/evaluation arrays without repeated JPEG decode each epoch. |  |  |
| TASK-004 | Implement `src/performance.py::{configure_cuda,benchmark_dataloader}`. Configure AMP/TF32/cuDNN and benchmark worker candidates before the smoke test; write selected settings to the run snapshot. |  |  |
| TASK-005 | Implement `src/models/dcgan.py::{Generator,Discriminator}`, `src/models/critics.py::WGANCritic`, and `src/training/objectives.py::{dcgan_losses,wgan_losses,gradient_penalty,critic_input_gradient_stats}` using REQ-007 through REQ-010. |  |  |
| TASK-006 | Implement `src/training/trainer.py::GANTrainer` and `src/train.py::run_training`. Add AMP boundaries, update schedules, finite-metric checks, resume support, interval-only checkpoints/samples, and buffered metric writes. |  |  |
| TASK-007 | Implement `src/pipeline.py::{preflight,prepare_data,run_smoke_tests,run_main_comparison,run_lambda_ablation,build_report,summarize_artifacts}`. Each function accepts a `RunConfig`; no pipeline function reads mutable notebook state. |  |  |
| TASK-008 | Create `main.ipynb` with the exact REQ-020 sections. The first executable cell sets the repository root, reloads `config/run.yaml`, displays `num_images`, split/cache paths, CUDA device, profile, and selected worker count. The final cell prints clickable artifact paths. |  |  |
| TASK-009 | Create `tests/test_config.py`, `tests/test_data.py`, `tests/test_models.py`, and `tests/test_objectives.py`; test config reload after changing a temporary YAML from 5,000 to 10,000 and assert differing manifest/cache/run paths. |  |  |

**Completion criteria:** Editing only `dataset.num_images` from 5,000 to 10,000 and then executing notebook Run All produces a validated 10,000-image manifest/cache path; all three CUDA smoke tests complete without NaN/Inf.

### Implementation Phase 2

- **GOAL-002**: Execute the optimized controlled main comparison and gradient analysis.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | Implement `src/evaluation/fid.py::compute_cached_fid` to cache real features by manifest hash and evaluate generated tensors in memory. It must use the configurable evaluation count and cadence in REQ-014. |  |  |
| TASK-011 | Implement `src/training/loggers.py::RunLogger` with buffered CSV writes. Log update, training seconds, losses, scores, GP loss, interval gradient statistics, peak GPU memory, and FID. Log blanks for non-applicable metrics. |  |  |
| TASK-012 | Use the notebook main-experiment section to run DCGAN, WGAN-Clipping, and WGAN-GP from the same loaded configuration. Resolve OOM once before the first complete run; use the chosen actual batch size for all three. |  |  |
| TASK-013 | Implement `src/evaluation/validate_run.py::validate_main_run`. Require 10,000 generator updates, checkpoints/samples/FID at every cadence, finite metrics, and a matching configuration snapshot/manifest hash. |  |  |
| TASK-014 | Implement `src/analysis/{plot_main_comparison,plot_gradient_stability,build_main_table}.py`. Generate faces, FID vs updates, loss curves, training-time chart, gradient median plus p05-p95 chart, and `main_comparison.csv`. |  |  |
| TASK-015 | Persist `pipeline_timings.json` with cache, preflight, per-model training, per-FID, analysis, and report times. Use it to identify whether the remaining bottleneck is WGAN-GP gradient penalty, FID, or data loading. |  |  |

**Completion criteria:** Three valid 10,000-step runs exist for the active image count; all main figures/table exist; and gradient analysis uses logged values rather than reconstructed gradients.

### Implementation Phase 3

- **GOAL-003**: Run the small lambda ablation and create a reproducible report package.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-016 | Run lambda=1 and lambda=50 WGAN-GP jobs for 3,000 generator updates using the active configuration and main-run batch size; extract lambda=10 results through update 3,000 from the main run. |  |  |
| TASK-017 | Implement `src/analysis/plot_lambda_ablation.py` to write `lambda_ablation.png` and `lambda_ablation.csv` with lambda, FID at 3,000, training seconds, final GP, gradient median/p95, and stability label. |  |  |
| TASK-018 | Create `reports/celeba_wgan_gp_results.md` with Setup, Current Configuration, Dataset and Split, Main Comparison, Gradient Analysis, Lambda Ablation, Timings, Limitations, and Reproduction sections. Explicitly label FID-500 as a small-sample comparative score. |  |  |
| TASK-019 | Implement `src/analysis/rebuild_all.py` and call it from the notebook analysis section. It rebuilds every figure/table solely from the active `artifacts/runs/n<num_images>/` directory. |  |  |
| TASK-020 | Run `python -m pytest tests -q` and a clean notebook Run All validation. Pass only when every linked report artifact exists and the displayed configuration image count matches all artifact paths. |  |  |

**Completion criteria:** Lambda values 1, 10, and 50 have comparable data; report claims match logs; clean rebuild succeeds; and a new 10k Run All cannot mix outputs from the prior 5k run.

## 3. Alternatives

- **ALT-001**: Keep 5,000 hard-coded in code/notebook. Rejected because users cannot safely switch to 10,000 and can accidentally reuse invalid splits or FID references.
- **ALT-002**: Put the whole pipeline in notebook cells. Rejected because it creates duplicated logic that cannot be properly tested, reused, resumed, or invoked from the command line.
- **ALT-003**: Decode original JPEGs on every epoch. Rejected because 10,000 images repeatedly crop/resize/decode across many epochs; a validated uint8 memmap makes that one-time work.
- **ALT-004**: Compute FID every 1,000 updates and write generated images to disk. Rejected because it adds repeated Inception and file-system overhead without materially improving the four-point training curve.
- **ALT-005**: Log WGAN-Clipping gradient norms at every critic update. Rejected because it would add unnecessary autograd work; the 100-update interval retains the stability signal at negligible cost.

## 4. Dependencies

- **DEP-001**: CelebA aligned-and-cropped image archive, placed under configured `data/celeba/` according to its license.
- **DEP-002**: CUDA-enabled PyTorch compatible with the RTX 4050 and an installed NVIDIA driver.
- **DEP-003**: `torch`, `torchvision`, `numpy`, `pyyaml`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `pytest`, and one Inception/FID implementation with feature caching.
- **DEP-004**: At least 10 GB free disk space for CelebA, dataset caches, checkpoints, figures, and reports.

## 5. Files

- **FILE-001**: `main.ipynb` - config-reloading end-to-end orchestration notebook.
- **FILE-002**: `config/run.yaml` and `src/config.py` - editable run values, validation, and immutable snapshots.
- **FILE-003**: `src/pipeline.py` and `src/performance.py` - notebook-callable pipeline and RTX 4050 performance controls.
- **FILE-004**: `src/data/celeba.py` and `src/data/cache.py` - dynamic split, loaders, and keyed uint8 memmap cache.
- **FILE-005**: `src/models/`, `src/training/`, `src/evaluation/`, and `src/analysis/` - model, training, evaluation, logging, report, and plotting modules.
- **FILE-006**: `data/splits/celeba_n<num_images>_seed42.csv`, `data/cache/`, and `artifacts/runs/n<num_images>/` - configuration-keyed generated data/results.
- **FILE-007**: `tests/test_config.py`, `tests/test_data.py`, `tests/test_models.py`, and `tests/test_objectives.py` - config and pipeline verification.

## 6. Testing

- **TEST-001**: Change a temporary config from `num_images: 5000` to `10000`, reload it twice, and assert only the matching manifest/cache/output roots are returned.
- **TEST-002**: Verify memmap invalidation whenever its stored manifest hash, transform version, resolution, or image count differs.
- **TEST-003**: Verify every model has a CPU forward pass and every objective produces finite expected-shape tensors.
- **TEST-004**: CUDA smoke-test all models with current configuration, AMP enabled, and GP forced to float32.
- **TEST-005**: Verify cached FID uses equal real/generated image counts and never writes generated evaluation images to disk.
- **TEST-006**: Verify notebook Run All calls all eight pipeline stages in order and contains no model/training implementation symbols.
- **TEST-007**: Verify a clean rebuild creates every report figure/table from a single active image-count artifact root.

## 7. Risks & Assumptions

- **RISK-001**: FID-500 has high variance and a 5k/10k subset may overfit. Mitigation: fixed split, fixed evaluation count, qualitative samples, and conservative single-seed conclusions.
- **RISK-002**: WGAN-GP is inherently slower because it requires second-order autograd behavior for gradient penalty. Mitigation: preserve its float32 calculation for correctness while removing avoidable data/FID/logging overhead.
- **RISK-003**: RTX 4050 VRAM and laptop thermals vary. Mitigation: shared OOM fallback, checkpoint/resume, measured worker selection, and separate phase timings.
- **RISK-004**: Changing `num_images` can leave older artifacts on disk. Mitigation: every cache, manifest, FID feature set, and run path includes image count and manifest hash validation.
- **ASSUMPTION-001**: CelebA is downloaded before the first preflight run.
- **ASSUMPTION-002**: System RAM can hold worker buffers and a 10k uint8 64x64 cache; the cache remains memory-mapped rather than fully loaded if RAM is constrained.

## 8. Related Specifications / Further Reading

- [Improved Training of Wasserstein GANs - Gulrajani et al.](https://arxiv.org/abs/1704.00028)
- [Wasserstein GAN - Arjovsky, Chintala, Bottou](https://arxiv.org/abs/1701.07875)
- [CelebA dataset project page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [PyTorch performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)