# CelebA GAN comparison

This project compares DCGAN, WGAN with weight clipping, and WGAN-GP at 64x64.

## Research paper

The project follows Gulrajani et al., *Improved Training of Wasserstein GANs* (2017), especially its comparison between WGAN weight clipping and WGAN-GP. A local copy is available at [docs/references/improved-training-wasserstein-gans-gulrajani-2017.pdf](docs/references/improved-training-wasserstein-gans-gulrajani-2017.pdf). The public version is available at https://arxiv.org/abs/1704.00028.

## Quick start

1. Place the CelebA image directory below `data/celeba/`.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Edit `dataset.num_images` in `config/run.yaml` to `5000` or `10000`.
4. Open and run all cells in `main.ipynb`.

The notebook reloads the YAML at the start of every Run All. Changing image count creates a separate manifest, cache, and result root under `artifacts/runs/n<image_count>/`; results cannot silently mix between 5k and 10k runs.

`python -m src.train --mode smoke` is a smaller command-line check that uses the same implementation.

## Performance notes

The first run creates a center-cropped/resized uint8 cache. Later epochs avoid repeated JPEG decoding. The RTX 4050 profile enables CUDA AMP, TF32, pinned memory, worker benchmarking, in-memory FID batches, and sparse checkpoint/sample/FID cadence. WGAN-GP remains slower than the other models because its gradient penalty requires autograd through critic inputs; that calculation intentionally remains float32.
