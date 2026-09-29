"""Training implementation shared by notebook and command-line entry points."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
import random

import numpy as np
import torch
from torchvision.utils import save_image

from src.config import RunConfig, write_config_snapshot
from src.evaluation.fid import CachedFIDEvaluator
from src.models.critics import WGANCritic
from src.models.dcgan import Discriminator, Generator
from src.performance import maybe_compile
from src.training.loggers import RunLogger
from src.training.objectives import critic_input_gradient_stats, dcgan_losses, gradient_penalty, wgan_losses


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weights_init(module) -> None:
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(module.bias.data)


def _autocast(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")
        if device.type == "cuda"
        else nullcontext()
    )


class GANTrainer:
    def __init__(self, config: RunConfig, model_name: str, run_dir: Path, device: torch.device,
                 compile_models: bool = True):
        self.config, self.model_name, self.run_dir, self.device = config, model_name, run_dir, device
        set_seed(config.seed)
        channels, latent = config.get("models.base_channels"), config.get("models.latent_dim")

        gen_raw = Generator(latent, channels).to(device)
        adv_raw = (Discriminator(channels) if model_name == "dcgan" else WGANCritic(channels)).to(device)
        gen_raw.apply(_weights_init); adv_raw.apply(_weights_init)

        # torch.compile fuses CUDA kernels for a significant throughput boost after
        # the first-batch warm-up.  Disabled for smoke tests via compile_models=False.
        self.generator = maybe_compile(gen_raw, enabled=compile_models)
        self.adversary = maybe_compile(adv_raw, enabled=compile_models)

        if model_name == "dcgan":
            lr, betas = config.get("training.dcgan.lr"), tuple(config.get("training.dcgan.betas"))
            self.g_optimizer = torch.optim.Adam(gen_raw.parameters(), lr=lr, betas=betas)
            self.d_optimizer = torch.optim.Adam(adv_raw.parameters(), lr=lr, betas=betas)
            self.n_critic = 1
        elif model_name == "wgan_clip":
            lr = config.get("training.wgan_clip.lr")
            self.g_optimizer = torch.optim.RMSprop(gen_raw.parameters(), lr=lr)
            self.d_optimizer = torch.optim.RMSprop(adv_raw.parameters(), lr=lr)
            self.n_critic = config.get("training.wgan_clip.n_critic")
        else:
            lr, betas = config.get("training.wgan_gp.lr"), tuple(config.get("training.wgan_gp.betas"))
            self.g_optimizer = torch.optim.Adam(gen_raw.parameters(), lr=lr, betas=betas)
            self.d_optimizer = torch.optim.Adam(adv_raw.parameters(), lr=lr, betas=betas)
            self.n_critic = config.get("training.wgan_gp.n_critic")

        # Keep references to raw (un-compiled) modules so state_dict works reliably.
        self._gen_raw = gen_raw
        self._adv_raw = adv_raw

        self.amp = bool(config.get("performance.amp"))
        # Separate scalers for G and D so their update() calls don't interfere.
        self.g_scaler = torch.cuda.amp.GradScaler(enabled=self.amp and device.type == "cuda")
        self.d_scaler = torch.cuda.amp.GradScaler(enabled=self.amp and device.type == "cuda")

        self.fixed_noise = torch.randn(64, latent, 1, 1, device=device)
        # In-memory best FID: avoids reading the entire checkpoint from disk on
        # every save just to compare scores.
        self._best_fid: float = float("inf")

    def _noise(self, size: int) -> torch.Tensor:
        return torch.randn(size, self.config.get("models.latent_dim"), 1, 1, device=self.device)

    def _save_artifacts(self, step: int, fid: float | None) -> None:
        checkpoints = self.run_dir / "checkpoints"; samples = self.run_dir / "samples"
        checkpoints.mkdir(parents=True, exist_ok=True); samples.mkdir(parents=True, exist_ok=True)
        state = {
            "gen_step": step,
            "generator": self._gen_raw.state_dict(),
            "adversary": self._adv_raw.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_scaler": self.g_scaler.state_dict(),
            "d_scaler": self.d_scaler.state_dict(),
            "fid": fid,
        }
        torch.save(state, checkpoints / "latest.pt")
        torch.save(state, checkpoints / f"step_{step:05d}.pt")
        # Use the in-memory best FID to avoid loading the checkpoint from disk.
        if fid is not None and fid <= self._best_fid:
            self._best_fid = fid
            torch.save(state, checkpoints / "best_fid.pt")
        was_training = self.generator.training; self.generator.eval()
        with torch.inference_mode():
            save_image(self.generator(self.fixed_noise), samples / f"step_{step:05d}.png",
                       normalize=True, value_range=(-1, 1), nrow=8)
        if was_training: self.generator.train()

    def _load_checkpoint(self, path: Path) -> int:
        """Load a checkpoint and return the step to resume from."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._gen_raw.load_state_dict(state["generator"])
        self._adv_raw.load_state_dict(state["adversary"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        if "g_scaler" in state:
            self.g_scaler.load_state_dict(state["g_scaler"])
        if "d_scaler" in state:
            self.d_scaler.load_state_dict(state["d_scaler"])
        if state.get("fid") is not None:
            self._best_fid = float(state["fid"])
        resumed_step = int(state.get("gen_step", 0))
        print(f"[resume] Loaded checkpoint at step {resumed_step} from {path}")
        return resumed_step

    def train(self, train_loader, eval_loader=None, max_steps: int | None = None,
              resume: bool = True) -> dict[str, float | int | None]:
        """Run training.

        Args:
            train_loader: Training DataLoader.
            eval_loader: Optional evaluation DataLoader used for FID.
            max_steps: Override the config value for gen steps.
            resume: If True, automatically resume from 'checkpoints/latest.pt'
                    when it exists inside run_dir.  Set to False to force a
                    fresh run even when a checkpoint is present.
        """
        max_steps = max_steps or self.config.get("training.max_gen_steps")
        write_config_snapshot(self.config, self.run_dir / "config.yaml",
                              {"device": str(self.device), "model": self.model_name})
        logger = RunLogger(self.run_dir)
        evaluator = None
        fid_error = None
        if eval_loader is not None:
            try:
                evaluator = CachedFIDEvaluator(eval_loader, self.device,
                                               self.config.get("evaluation.fid_batch_size"))
            except RuntimeError as error:
                fid_error = str(error)

        # --- Crash-recovery: resume from latest checkpoint if present. ---
        start_step = 0
        latest_ckpt = self.run_dir / "checkpoints" / "latest.pt"
        if resume and latest_ckpt.exists():
            start_step = self._load_checkpoint(latest_ckpt)

        if start_step >= max_steps:
            print(f"[resume] Already completed {start_step}/{max_steps} steps - skipping.")
            return {"model": self.model_name, "gen_steps": max_steps,
                    "training_seconds": 0.0, "fid": self._best_fid if self._best_fid < float("inf") else None}

        iterator = iter(train_loader); critic_updates = 0; fid_seconds = 0.0
        last_fid = None; start = perf_counter()

        for step in range(start_step + 1, max_steps + 1):
            grad_stats: dict[str, float | None] = {
                "grad_mean": None, "grad_std": None,
                "grad_p05": None, "grad_p50": None, "grad_p95": None,
            }
            gp_value = None
            # Holds the fake from the last D update so G can reuse it
            # and skip a redundant generator forward pass.
            last_fake_for_g: torch.Tensor | None = None

            for _ in range(self.n_critic):
                try: real = next(iterator)
                except StopIteration: iterator = iter(train_loader); real = next(iterator)
                real = real.to(self.device, non_blocking=True)
                noise = self._noise(real.size(0))
                self.d_optimizer.zero_grad(set_to_none=True)
                with _autocast(self.device, self.amp):
                    fake = self.generator(noise)
                    real_scores, fake_scores = self.adversary(real), self.adversary(fake.detach())
                    if self.model_name == "dcgan":
                        critic_loss, _ = dcgan_losses(real_scores, fake_scores)
                    else:
                        critic_loss, _ = wgan_losses(real_scores, fake_scores)
                norms = None
                if self.model_name == "wgan_gp":
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        penalty, norms = gradient_penalty(
                            self.adversary, real.float(), fake.float(),
                            self.config.get("training.wgan_gp.lambda_gp"),
                        )
                    critic_loss = critic_loss + penalty; gp_value = penalty.item()
                self.d_scaler.scale(critic_loss).backward()
                self.d_scaler.step(self.d_optimizer)
                self.d_scaler.update()
                if self.model_name == "wgan_clip":
                    value = self.config.get("training.wgan_clip.clip_value")
                    for parameter in self._adv_raw.parameters():
                        parameter.data.clamp_(-value, value)
                critic_updates += 1
                # Keep the last fake so G can reuse it, saving one forward pass.
                last_fake_for_g = fake
                if critic_updates % self.config.get("evaluation.gradient_log_interval_critic_updates") == 0:
                    grad_stats = (
                        {"grad_mean": norms.mean().item(), "grad_std": norms.std(unbiased=False).item(),
                         "grad_p05": norms.quantile(.05).item(), "grad_p50": norms.quantile(.5).item(),
                         "grad_p95": norms.quantile(.95).item()}
                        if norms is not None
                        else critic_input_gradient_stats(self.adversary, real.float(), fake.float())
                    )

            self.g_optimizer.zero_grad(set_to_none=True)
            with _autocast(self.device, self.amp):
                # Reuse the fake from the last D step to skip one extra generator
                # forward pass (saves ~n_critic - 1 generator calls per G step when
                # n_critic > 1, and exactly 1 call when n_critic == 1).
                generated = last_fake_for_g if last_fake_for_g is not None else self.generator(self._noise(real.size(0)))
                fake_scores = self.adversary(generated)
                if self.model_name == "dcgan":
                    generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        fake_scores, torch.ones_like(fake_scores))
                else:
                    generator_loss = -fake_scores.mean()
            self.g_scaler.scale(generator_loss).backward()
            self.g_scaler.step(self.g_optimizer)
            self.g_scaler.update()

            if evaluator and step % self.config.get("evaluation.fid_interval") == 0:
                before = perf_counter()
                last_fid = evaluator.compute(self.generator, self.config.get("models.latent_dim"),
                                             self.config.get("dataset.eval_images"))
                fid_seconds += perf_counter() - before
            if step % self.config.get("evaluation.sample_interval") == 0 or step == max_steps:
                self._save_artifacts(step, last_fid)
            logger.log({
                "gen_step": step,
                "training_seconds": perf_counter() - start - fid_seconds,
                "generator_loss": generator_loss.item(),
                "critic_loss": critic_loss.item(),
                "real_score": real_scores.mean().item(),
                "fake_score": fake_scores.mean().item(),
                "gp_loss": gp_value, "fid": last_fid, "fid_error": fid_error,
                **grad_stats,
            })
        logger.close()
        return {"model": self.model_name, "gen_steps": max_steps,
                "training_seconds": perf_counter() - start - fid_seconds, "fid": last_fid}



def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weights_init(module) -> None:
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.zeros_(module.bias.data)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda") if device.type == "cuda" else nullcontext()


class GANTrainer:
    def __init__(self, config: RunConfig, model_name: str, run_dir: Path, device: torch.device):
        self.config, self.model_name, self.run_dir, self.device = config, model_name, run_dir, device
        set_seed(config.seed)
        channels, latent = config.get("models.base_channels"), config.get("models.latent_dim")
        self.generator = Generator(latent, channels).to(device)
        self.adversary = (Discriminator(channels) if model_name == "dcgan" else WGANCritic(channels)).to(device)
        self.generator.apply(_weights_init); self.adversary.apply(_weights_init)
        if model_name == "dcgan":
            lr, betas = config.get("training.dcgan.lr"), tuple(config.get("training.dcgan.betas"))
            self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
            self.d_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=lr, betas=betas)
            self.n_critic = 1
        elif model_name == "wgan_clip":
            lr = config.get("training.wgan_clip.lr")
            self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
            self.d_optimizer = torch.optim.RMSprop(self.adversary.parameters(), lr=lr)
            self.n_critic = config.get("training.wgan_clip.n_critic")
        else:
            lr, betas = config.get("training.wgan_gp.lr"), tuple(config.get("training.wgan_gp.betas"))
            self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
            self.d_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=lr, betas=betas)
            self.n_critic = config.get("training.wgan_gp.n_critic")
        self.amp = bool(config.get("performance.amp"))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and device.type == "cuda")
        self.fixed_noise = torch.randn(64, latent, 1, 1, device=device)

    def _noise(self, size: int) -> torch.Tensor:
        return torch.randn(size, self.config.get("models.latent_dim"), 1, 1, device=self.device)

    def _save_artifacts(self, step: int, fid: float | None) -> None:
        checkpoints = self.run_dir / "checkpoints"; samples = self.run_dir / "samples"
        checkpoints.mkdir(parents=True, exist_ok=True); samples.mkdir(parents=True, exist_ok=True)
        state = {"gen_step": step, "generator": self.generator.state_dict(), "adversary": self.adversary.state_dict(),
                 "g_optimizer": self.g_optimizer.state_dict(), "d_optimizer": self.d_optimizer.state_dict(), "fid": fid}
        torch.save(state, checkpoints / "latest.pt")
        torch.save(state, checkpoints / f"step_{step:05d}.pt")
        if fid is not None:
            best = checkpoints / "best_fid.pt"
            if not best.exists() or fid <= torch.load(best, map_location="cpu", weights_only=False).get("fid", float("inf")):
                torch.save(state, best)
        was_training = self.generator.training; self.generator.eval()
        with torch.inference_mode():
            save_image(self.generator(self.fixed_noise), samples / f"step_{step:05d}.png", normalize=True, value_range=(-1, 1), nrow=8)
        if was_training: self.generator.train()

    def train(self, train_loader, eval_loader=None, max_steps: int | None = None) -> dict[str, float | int | None]:
        max_steps = max_steps or self.config.get("training.max_gen_steps")
        write_config_snapshot(self.config, self.run_dir / "config.yaml", {"device": str(self.device), "model": self.model_name})
        logger = RunLogger(self.run_dir)
        evaluator = None
        fid_error = None
        if eval_loader is not None:
            try:
                evaluator = CachedFIDEvaluator(eval_loader, self.device, self.config.get("evaluation.fid_batch_size"))
            except RuntimeError as error:
                fid_error = str(error)
        iterator = iter(train_loader); critic_updates = 0; fid_seconds = 0.0; start = perf_counter()
        last_fid = None
        for step in range(1, max_steps + 1):
            grad_stats: dict[str, float | None] = {"grad_mean": None, "grad_std": None, "grad_p05": None, "grad_p50": None, "grad_p95": None}; gp_value = None
            for _ in range(self.n_critic):
                try: real = next(iterator)
                except StopIteration: iterator = iter(train_loader); real = next(iterator)
                real = real.to(self.device, non_blocking=True); noise = self._noise(real.size(0))
                self.d_optimizer.zero_grad(set_to_none=True)
                with _autocast(self.device, self.amp):
                    fake = self.generator(noise)
                    real_scores, fake_scores = self.adversary(real), self.adversary(fake.detach())
                    if self.model_name == "dcgan": critic_loss, _ = dcgan_losses(real_scores, fake_scores)
                    else: critic_loss, _ = wgan_losses(real_scores, fake_scores)
                norms = None
                if self.model_name == "wgan_gp":
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        penalty, norms = gradient_penalty(self.adversary, real.float(), fake.float(), self.config.get("training.wgan_gp.lambda_gp"))
                    critic_loss = critic_loss + penalty; gp_value = penalty.item()
                self.scaler.scale(critic_loss).backward(); self.scaler.step(self.d_optimizer); self.scaler.update()
                if self.model_name == "wgan_clip":
                    value = self.config.get("training.wgan_clip.clip_value")
                    for parameter in self.adversary.parameters(): parameter.data.clamp_(-value, value)
                critic_updates += 1
                if critic_updates % self.config.get("evaluation.gradient_log_interval_critic_updates") == 0:
                    grad_stats = ({"grad_mean": norms.mean().item(), "grad_std": norms.std(unbiased=False).item(),
                                   "grad_p05": norms.quantile(.05).item(), "grad_p50": norms.quantile(.5).item(), "grad_p95": norms.quantile(.95).item()}
                                  if norms is not None else critic_input_gradient_stats(self.adversary, real.float(), fake.float()))
            self.g_optimizer.zero_grad(set_to_none=True)
            with _autocast(self.device, self.amp):
                generated = self.generator(self._noise(real.size(0)))
                fake_scores = self.adversary(generated)
                if self.model_name == "dcgan":
                    generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
                else: generator_loss = -fake_scores.mean()
            self.scaler.scale(generator_loss).backward(); self.scaler.step(self.g_optimizer); self.scaler.update()
            if evaluator and step % self.config.get("evaluation.fid_interval") == 0:
                before = perf_counter(); last_fid = evaluator.compute(self.generator, self.config.get("models.latent_dim"), self.config.get("dataset.eval_images")); fid_seconds += perf_counter() - before
            if step % self.config.get("evaluation.sample_interval") == 0 or step == max_steps:
                self._save_artifacts(step, last_fid)
            logger.log({"gen_step": step, "training_seconds": perf_counter() - start - fid_seconds, "generator_loss": generator_loss.item(),
                        "critic_loss": critic_loss.item(), "real_score": real_scores.mean().item(), "fake_score": fake_scores.mean().item(),
                        "gp_loss": gp_value, "fid": last_fid, "fid_error": fid_error, **grad_stats})
        logger.close()
        return {"model": self.model_name, "gen_steps": max_steps, "training_seconds": perf_counter() - start - fid_seconds, "fid": last_fid}
