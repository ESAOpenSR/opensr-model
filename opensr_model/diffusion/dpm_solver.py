"""Optional DPM-Solver++ 2M sampling for the native OpenSR denoiser."""

from numbers import Integral

import torch


class DPMSolverPlusPlusSampler:
    """A single inference trajectory; multistep history must not cross calls."""

    def __init__(self, model, steps: int):
        if (
            isinstance(steps, bool)
            or not isinstance(steps, Integral)
            or not 1 <= steps < model.num_timesteps
        ):
            raise ValueError(
                "DPM-Solver++ sampling_steps must be a positive integer smaller "
                "than the number of training timesteps"
            )
        if model.parameterization != "eps":
            raise ValueError("DPM-Solver++ inference requires parameterization='eps'")

        try:
            from diffusers import DPMSolverMultistepScheduler
        except ImportError as exc:
            raise ImportError(
                "DPM-Solver++ requires diffusers. "
                "Install it with `pip install 'opensr-model[dpm]'`."
            ) from exc

        self.model = model
        # Use the loaded buffers, not generic beta defaults or YAML endpoints.
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=model.num_timesteps,
            trained_betas=model.betas.detach().float().cpu().numpy(),
            prediction_type="epsilon",
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            thresholding=False,
            lower_order_final=True,
            timestep_spacing="linspace",
            steps_offset=0,
            use_karras_sigmas=False,
            final_sigmas_type="zero",
            rescale_betas_zero_snr=False,
        )
        self.scheduler.set_timesteps(int(steps))
        self.timesteps = self.scheduler.timesteps.tolist()

    @torch.no_grad()
    def step(self, latent: torch.Tensor, conditioning: torch.Tensor, t: int):
        timesteps = torch.full(
            (latent.shape[0],), t, device=latent.device, dtype=torch.long
        )
        model_input = self.scheduler.scale_model_input(latent, t)
        noise = self.model.apply_model(model_input, timesteps, conditioning)
        return self.scheduler.step(noise, t, latent).prev_sample
