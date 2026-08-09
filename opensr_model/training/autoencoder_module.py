"""PyTorch Lightning training module for the checkpoint-compatible AutoencoderKL."""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch

from opensr_model.autoencoder.autoencoder import AutoencoderKL

from .losses import (
    PatchDiscriminator,
    adaptive_generator_weight,
    discriminator_hinge_loss,
    discriminator_vanilla_loss,
    generator_adversarial_loss,
)
from .metrics import (
    batch_clipped_fraction,
    compute_reconstruction_metrics,
    compute_rgb_nir_reconstruction_metrics,
)
from .optim import build_scheduler


class AutoencoderTrainingModule(pl.LightningModule):
    """Train the repository's exact autoencoder without modifying inference code.

    The loss follows the checkpoint-era latent-diffusion implementation: stochastic
    KL latents, a random three-of-four-band L1+LPIPS objective, and an adaptive
    three-channel PatchGAN. Native exports still contain exactly
    ``AutoencoderKL.state_dict()`` (or those keys merged into a supplied released
    full-model checkpoint by :class:`NativeCheckpointCallback`).
    """

    checkpoint_stage = "autoencoder"

    def __init__(
        self,
        model_config: Mapping[str, Any],
        training_config: Optional[Mapping[str, Any]] = None,
        *,
        pretrained_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        training_config = _plain_mapping(training_config or {})
        model_config = _plain_mapping(model_config)
        first_stage = _plain_mapping(
            model_config.get("first_stage_config", model_config)
        )
        if "embed_dim" not in first_stage:
            raise ValueError("first_stage_config.embed_dim is required")

        self.autoencoder = AutoencoderKL(
            first_stage, embed_dim=int(first_stage["embed_dim"])
        )
        self.model_config = model_config
        self.training_config = training_config
        self.loss_config = _plain_mapping(training_config.get("loss", {}))
        self.optimizer_config = _plain_mapping(training_config.get("optimizer", {}))
        self.scheduler_config = _plain_mapping(training_config.get("scheduler", {}))
        self.metric_config = _plain_mapping(training_config.get("metrics", {}))
        self.value_min = float(self.metric_config.get("value_min", 0.0))
        self.value_max = float(self.metric_config.get("value_max", 1.0))
        self.gradient_clip_val = float(training_config.get("gradient_clip_val", 0.0))
        self.manual_accumulate_grad_batches = int(
            training_config.get("accumulate_grad_batches", 1)
        )
        if self.manual_accumulate_grad_batches < 1:
            raise ValueError("training.accumulate_grad_batches must be positive")

        self.kl_weight = float(self.loss_config.get("kl_weight", 1e-4))
        self.perceptual_weight = float(self.loss_config.get("perceptual_weight", 1.0))
        self.discriminator_factor = float(
            self.loss_config.get("discriminator_factor", 1.0)
        )
        self.discriminator_weight = float(
            self.loss_config.get("discriminator_weight", 0.5)
        )
        self.adversarial_kind = str(self.loss_config.get("adversarial_kind", "hinge"))
        if self.adversarial_kind not in {"hinge", "vanilla"}:
            raise ValueError("loss.adversarial_kind must be hinge or vanilla")
        self.discriminator_start = int(self.loss_config.get("discriminator_start", 0))
        self.logvar = torch.nn.Parameter(
            torch.ones(()) * float(self.loss_config.get("logvar_init", 0.0))
        )

        self.perceptual_loss: Optional[torch.nn.Module] = None
        if self.perceptual_weight > 0:
            self.perceptual_loss = self._build_perceptual_loss(
                str(self.loss_config.get("perceptual_backbone", "vgg"))
            )

        self.discriminator: Optional[PatchDiscriminator] = None
        if self.discriminator_factor > 0:
            self.discriminator = PatchDiscriminator(
                in_channels=3,
                base_channels=int(self.loss_config.get("discriminator_channels", 64)),
                num_layers=int(self.loss_config.get("discriminator_layers", 3)),
            )
            self.automatic_optimization = False
        self.register_buffer("generator_updates", torch.zeros((), dtype=torch.long))
        self._manual_scheduler_metadata: list[dict[str, Any]] = []

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "training_config": training_config,
                "pretrained_checkpoint": pretrained_checkpoint,
            }
        )
        if pretrained_checkpoint:
            from .checkpoints import load_autoencoder_weights

            load_autoencoder_weights(
                self.autoencoder, pretrained_checkpoint, strict=True
            )

    def forward(
        self, image: torch.Tensor, *, sample_posterior: bool = True
    ) -> tuple[torch.Tensor, Any]:
        return self.autoencoder(image, sample_posterior=sample_posterior)

    def train(self, mode: bool = True) -> "AutoencoderTrainingModule":
        super().train(mode)
        # LPIPS is a frozen feature metric. Lightning recursively switches all
        # children to train mode, which would otherwise re-enable its dropout.
        if self.perceptual_loss is not None:
            self.perceptual_loss.eval()
        return self

    def native_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return standalone component keys with no Lightning wrapper prefix."""

        return OrderedDict(self.autoencoder.state_dict())

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        image, mask = self._batch_image_and_mask(batch)
        self._log_input_clipped_fraction(batch, "train", image)
        reconstruction, posterior, latent = self._reconstruct(image, sample=True)
        loss_reconstruction = self._mask_prediction(reconstruction, image, mask)
        loss_image, loss_reconstruction, selected_bands = self._random_three_band_pair(
            image, loss_reconstruction
        )
        components = self._generator_components(
            loss_reconstruction, loss_image, posterior, training=True
        )
        components["selected_band_mean"] = selected_bands.float().mean()
        generator_loss = components.pop("loss")

        if self.discriminator is None:
            self._log_components("train", components, generator_loss, image.shape[0])
            return generator_loss

        self._manual_training_step(
            image,
            mask,
            generator_loss,
            components,
            batch_idx,
        )
        return generator_loss.detach()

    def validation_step(
        self, batch: Mapping[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        image, mask = self._batch_image_and_mask(batch)
        self._log_input_clipped_fraction(batch, "val", image)
        # The released autoencoder was selected using stochastic validation
        # reconstructions, so do not replace the posterior sample with its mode.
        reconstruction, posterior, latent = self._reconstruct(image, sample=True)
        loss_reconstruction = self._mask_prediction(reconstruction, image, mask)
        loss_image, loss_reconstruction, selected_bands = self._random_three_band_pair(
            image, loss_reconstruction
        )
        components = self._generator_components(
            loss_reconstruction, loss_image, posterior, training=False
        )
        components["selected_band_mean"] = selected_bands.float().mean()
        loss = components.pop("loss")
        if self.discriminator is not None:
            discriminator_components = self._discriminator_components(
                loss_image, loss_reconstruction.detach()
            )
            components.update(discriminator_components)
        self._log_components("val", components, loss, image.shape[0])
        # Keep the legacy monitor name used by the released checkpoint.
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=image.shape[0],
        )
        metrics = compute_reconstruction_metrics(
            reconstruction,
            image,
            mask,
            value_min=self.value_min,
            value_max=self.value_max,
        )
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=image.shape[0],
            )
        channel_metrics = compute_rgb_nir_reconstruction_metrics(
            reconstruction,
            image,
            mask,
            value_min=self.value_min,
            value_max=self.value_max,
        )
        for name, value in channel_metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=image.shape[0],
            )
        self.log(
            "val/latent_mean",
            posterior.mean.mean(),
            sync_dist=True,
            batch_size=image.shape[0],
        )
        self.log(
            "val/latent_std",
            posterior.std.mean(),
            sync_dist=True,
            batch_size=image.shape[0],
        )

        sample_ids = batch.get(
            "sample_id", [f"validation_{batch_idx}_{i}" for i in range(image.shape[0])]
        )
        return {
            "images": OrderedDict(
                target=image.detach(),
                reconstruction=reconstruction.detach(),
                absolute_error=(reconstruction - image)
                .abs()
                .mean(dim=1, keepdim=True)
                .detach(),
            ),
            "sample_ids": list(sample_ids),
        }

    def configure_optimizers(self) -> Any:
        generator_optimizer = self._build_legacy_adam(
            self.autoencoder.parameters(), self.optimizer_config
        )
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1)
        if self.discriminator is not None:
            total_steps = max(
                1, math.ceil(total_steps / self.manual_accumulate_grad_batches)
            )
        generator_scheduler = self._build_legacy_scheduler(
            generator_optimizer, self.scheduler_config, total_steps=total_steps
        )
        if self.discriminator is None:
            if generator_scheduler is None:
                return generator_optimizer
            return {
                "optimizer": generator_optimizer,
                "lr_scheduler": generator_scheduler,
            }

        discriminator_config = _plain_mapping(
            self.training_config.get("discriminator_optimizer", self.optimizer_config)
        )
        discriminator_optimizer = self._build_legacy_adam(
            self.discriminator.parameters(), discriminator_config
        )
        schedulers = []
        self._manual_scheduler_metadata = []
        if generator_scheduler is not None:
            schedulers.append(generator_scheduler)
            self._manual_scheduler_metadata.append(
                {
                    "optimizer_index": 0,
                    "interval": str(generator_scheduler["interval"]),
                    "monitor": str(generator_scheduler.get("monitor", "val/loss")),
                }
            )
        discriminator_scheduler_config = _plain_mapping(
            self.training_config.get("discriminator_scheduler", self.scheduler_config)
        )
        discriminator_scheduler = self._build_legacy_scheduler(
            discriminator_optimizer,
            discriminator_scheduler_config,
            total_steps=total_steps,
        )
        if discriminator_scheduler is not None:
            schedulers.append(discriminator_scheduler)
            self._manual_scheduler_metadata.append(
                {
                    "optimizer_index": 1,
                    "interval": str(discriminator_scheduler["interval"]),
                    "monitor": str(discriminator_scheduler.get("monitor", "val/loss")),
                }
            )
        if schedulers:
            return [generator_optimizer, discriminator_optimizer], schedulers
        return [generator_optimizer, discriminator_optimizer]

    @staticmethod
    def _build_legacy_adam(
        parameters: Any, config: Mapping[str, Any]
    ) -> torch.optim.Adam:
        config = _plain_mapping(config)
        betas = tuple(config.get("betas", (0.5, 0.9)))
        if len(betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")
        return torch.optim.Adam(
            parameters,
            lr=float(config.get("learning_rate", config.get("lr", 1e-4))),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    @staticmethod
    def _build_legacy_scheduler(
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        *,
        total_steps: int,
    ) -> Optional[dict[str, Any]]:
        config = _plain_mapping(config)
        name = str(config.get("name", "plateau")).lower()
        if name != "plateau":
            return build_scheduler(optimizer, config, total_steps=total_steps)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 25)),
            threshold=float(config.get("threshold", 1e-4)),
            threshold_mode=str(config.get("threshold_mode", "rel")),
            min_lr=float(config.get("min_lr", 0.0)),
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": str(config.get("monitor", "train/total_loss")),
        }

    def on_train_epoch_end(self) -> None:
        if self.discriminator is None or not self._manual_scheduler_metadata:
            return
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler, metadata in zip(schedulers, self._manual_scheduler_metadata):
            if metadata["interval"] == "epoch" and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

    def on_validation_epoch_end(self) -> None:
        if (
            self.discriminator is None
            or not self._manual_scheduler_metadata
            or self.trainer.sanity_checking
        ):
            return
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler, metadata in zip(schedulers, self._manual_scheduler_metadata):
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                continue
            monitor = metadata["monitor"]
            metric = self.trainer.callback_metrics.get(monitor)
            if metric is None:
                raise RuntimeError(
                    f"Manual plateau scheduler monitor {monitor!r} was not logged"
                )
            scheduler.step(metric)

    def _reconstruct(
        self, image: torch.Tensor, *, sample: bool
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        posterior = self.autoencoder.encode(image)
        if sample:
            latent = posterior.mean + posterior.std * torch.randn_like(posterior.mean)
        else:
            latent = posterior.mode()
        reconstruction = self.autoencoder.decode(latent)
        return reconstruction, posterior, latent

    def _generator_components(
        self,
        reconstruction: torch.Tensor,
        image: torch.Tensor,
        posterior: Any,
        *,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        if reconstruction.shape != image.shape or reconstruction.shape[1] != 3:
            raise ValueError(
                "Legacy autoencoder image losses require matching three-band tensors"
            )
        pixel_error = (image.contiguous() - reconstruction.contiguous()).abs()
        perceptual_map = reconstruction.new_zeros((image.shape[0], 1, 1, 1))
        if self.perceptual_loss is not None:
            perceptual_map = self._perceptual(reconstruction, image)

        # LPIPS returns Bx1x1x1. The historical implementation deliberately
        # broadcast it over every pixel/channel before summing over the batch.
        reconstruction_map = pixel_error + self.perceptual_weight * perceptual_map
        nll_map = reconstruction_map / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_map) / float(image.shape[0])
        kl_loss = torch.sum(posterior.kl()) / float(image.shape[0])

        adversarial = reconstruction.new_zeros(())
        adaptive_weight = reconstruction.new_zeros(())
        gan_active = (
            self.discriminator is not None
            and int(self.generator_updates) >= self.discriminator_start
        )
        if gan_active:
            adversarial = generator_adversarial_loss(
                self.discriminator(reconstruction.contiguous()), self.adversarial_kind
            )
            if training:
                adaptive_weight = adaptive_generator_weight(
                    nll_loss,
                    adversarial,
                    self.autoencoder.decoder.conv_out.weight,
                    discriminator_weight=self.discriminator_weight,
                )
        disc_factor = reconstruction.new_tensor(
            self.discriminator_factor if gan_active else 0.0
        )
        loss = (
            nll_loss
            + self.kl_weight * kl_loss
            + adaptive_weight * disc_factor * adversarial
        )
        return {
            "loss": loss,
            "l1_loss": pixel_error.mean(),
            "reconstruction_loss": reconstruction_map.mean(),
            "nll_loss": nll_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_map.mean(),
            "logvar": self.logvar,
            "adaptive_discriminator_weight": adaptive_weight,
            "discriminator_factor": disc_factor,
            "generator_adversarial_loss": adversarial,
        }

    def _manual_training_step(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor],
        generator_loss: torch.Tensor,
        components: Mapping[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        generator_optimizer, discriminator_optimizer = self.optimizers()
        accumulate = self._accumulate_grad_batches()
        divisor = self._accumulation_divisor(batch_idx, accumulate)
        start_cycle = batch_idx % accumulate == 0
        should_step = (batch_idx + 1) % accumulate == 0 or self._is_last_training_batch(
            batch_idx
        )

        with generator_optimizer.toggle_model(sync_grad=should_step):
            if start_cycle:
                generator_optimizer.zero_grad()
            self.manual_backward(generator_loss / float(divisor))
            if should_step:
                if self.gradient_clip_val > 0:
                    self.clip_gradients(
                        generator_optimizer,
                        gradient_clip_val=self.gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                generator_optimizer.step()
                self.generator_updates.add_(1)
                self._step_manual_scheduler(0)

        discriminator_loss = image.new_zeros(())
        if int(self.generator_updates) >= self.discriminator_start:
            # Lightning 1.9 invoked the old ``training_step`` independently for
            # optimizer_idx 0 and 1. Reproduce that second stochastic posterior
            # sample and independent Python random three-band draw here.
            with torch.no_grad():
                discriminator_reconstruction, _, _ = self._reconstruct(
                    image, sample=True
                )
                discriminator_reconstruction = self._mask_prediction(
                    discriminator_reconstruction, image, mask
                )
                discriminator_image, discriminator_reconstruction, selected = (
                    self._random_three_band_pair(image, discriminator_reconstruction)
                )
            with discriminator_optimizer.toggle_model(sync_grad=should_step):
                if start_cycle:
                    discriminator_optimizer.zero_grad()
                discriminator_components = self._discriminator_components(
                    discriminator_image, discriminator_reconstruction
                )
                discriminator_components["discriminator_selected_band_mean"] = (
                    selected.float().mean()
                )
                discriminator_loss = discriminator_components["discriminator_loss"]
                self.manual_backward(discriminator_loss / float(divisor))
                if should_step:
                    if self.gradient_clip_val > 0:
                        self.clip_gradients(
                            discriminator_optimizer,
                            gradient_clip_val=self.gradient_clip_val,
                            gradient_clip_algorithm="norm",
                        )
                    discriminator_optimizer.step()
                    self._step_manual_scheduler(1)
                components = dict(components)
                components.update(discriminator_components)

        self._log_components("train", components, generator_loss, image.shape[0])

    def _discriminator_components(
        self, image: torch.Tensor, reconstruction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.discriminator is None:
            raise RuntimeError("Discriminator loss requested without a discriminator")
        real_logits = self.discriminator(image.contiguous().detach())
        fake_logits = self.discriminator(reconstruction.contiguous().detach())
        if self.adversarial_kind == "hinge":
            discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
        else:
            discriminator_loss = discriminator_vanilla_loss(real_logits, fake_logits)
        active = int(self.generator_updates) >= self.discriminator_start
        discriminator_loss = discriminator_loss * (
            self.discriminator_factor if active else 0.0
        )
        return {
            "discriminator_loss": discriminator_loss,
            "discriminator_real_logits": real_logits.mean(),
            "discriminator_fake_logits": fake_logits.mean(),
        }

    def _step_manual_scheduler(self, index: int) -> None:
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler, metadata in zip(schedulers, self._manual_scheduler_metadata):
            if metadata["optimizer_index"] == index and metadata["interval"] == "step":
                scheduler.step()

    def _perceptual(
        self,
        reconstruction: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if reconstruction.shape[1] != 3 or image.shape[1] != 3:
            raise ValueError("LPIPS requires the selected three-band tensors")
        # Preserve the historical executable behavior: the [0, 1] reflectance
        # tensors are passed directly to LPIPS, without an extra [-1, 1] mapping.
        return self.perceptual_loss(image.contiguous(), reconstruction.contiguous())

    @staticmethod
    def _random_three_band_pair(
        image: torch.Tensor, reconstruction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.shape != reconstruction.shape:
            raise ValueError("Image and reconstruction shapes must match")
        channels = image.shape[1]
        if channels < 3:
            raise ValueError("The legacy image objective requires at least three bands")
        if channels == 3:
            selected = torch.arange(3, device=image.device)
        else:
            selected = torch.tensor(
                random.sample(range(channels), 3),
                device=image.device,
                dtype=torch.long,
            )
        return image[:, selected], reconstruction[:, selected], selected

    @staticmethod
    def _mask_prediction(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return prediction
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[0] != prediction.shape[0]:
            raise ValueError("valid_mask must be Bx1xHxW (or BxHxW)")
        if mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("valid_mask must have the image spatial shape")
        mask = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)
        if mask.shape[1] == 1 and prediction.shape[1] != 1:
            mask = mask.expand(-1, prediction.shape[1], -1, -1)
        return prediction * mask + target.detach() * (1.0 - mask)

    @staticmethod
    def _build_perceptual_loss(backbone: str) -> torch.nn.Module:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS was enabled but is not installed. Install opensr-model[train]."
            ) from exc
        module = lpips.LPIPS(net=backbone)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    def _batch_image_and_mask(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if "image" not in batch:
            raise KeyError("Autoencoder batches require an 'image' HR tensor")
        image = batch["image"].float()
        mask = batch.get("valid_mask")
        if mask is not None:
            mask = mask.float()
        return image, mask

    def _log_input_clipped_fraction(
        self,
        batch: Mapping[str, Any],
        stage: str,
        reference: torch.Tensor,
    ) -> None:
        fraction = batch_clipped_fraction(batch, reference)
        if fraction is None:
            return
        self.log(
            f"{stage}/input_clipped_fraction",
            fraction,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=reference.shape[0],
        )

    def _log_components(
        self,
        stage: str,
        components: Mapping[str, torch.Tensor],
        loss: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.log(
            f"{stage}/total_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "train",
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in components.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=stage == "train",
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _accumulate_grad_batches(self) -> int:
        return self.manual_accumulate_grad_batches

    def _accumulation_divisor(self, batch_idx: int, accumulate: int) -> int:
        total = self.trainer.num_training_batches
        if not isinstance(total, int):
            return accumulate
        cycle_start = (batch_idx // accumulate) * accumulate
        return max(1, min(accumulate, total - cycle_start))

    def _is_last_training_batch(self, batch_idx: int) -> bool:
        total = self.trainer.num_training_batches
        return isinstance(total, int) and batch_idx + 1 >= total


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass
    return dict(value)


__all__ = ["AutoencoderTrainingModule"]
