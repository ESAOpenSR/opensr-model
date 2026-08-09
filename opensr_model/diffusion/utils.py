import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch import nn as nn
from tqdm import tqdm


def exists(val: Any) -> bool:
    """
    Returns whether the given value exists (i.e., is not None).

    Args:
        val (Any): The value to check.

    Returns:
        bool: Whether the value exists.
    """
    return val is not None


def default(val: Any, d: Callable) -> Any:
    """
    Returns the given value if it exists, otherwise returns the default value.

    Args:
        val (Any): The value to check.
        d (Callable): The default value or function to generate the default value.

    Returns:
        Any: The given value or the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def count_params(model: nn.Module, verbose: bool = False) -> int:
    """
    Returns the total number of parameters in the given model.

    Args:
        model (nn.Module): The model.
        verbose (bool): Whether to print the number of parameters.

    Returns:
        int: The total number of parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def disabled_train(self, mode: bool = True) -> nn.Module:
    """
    Overwrites the `train` method of the model to disable changing the mode.

    Args:
        mode (bool): Whether to enable or disable training mode.

    Returns:
        nn.Module: The model.
    """
    return self


def make_convolutional_sample(
    batch: Tensor,
    model: nn.Module,
    custom_steps: Optional[Union[int, Tuple[int, int]]] = None,
    eta: float = 1.0,
    quantize_x0: bool = False,
    custom_shape: Optional[Tuple[int, int]] = None,
    temperature: float = 1.0,
    noise_dropout: float = 0.0,
    corrector: Optional[nn.Module] = None,
    corrector_kwargs: Optional[dict] = None,
    x_T: Optional[Tensor] = None,
    ddim_use_x0_pred: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Generates a convolutional sample using the given model.

    Args:
        batch (Tensor): The input batch tensor.
        model (nn.Module): The model to use for sampling.
        custom_steps (Optional[Union[int, Tuple[int, int]]]): The custom number of steps.
        eta (float): The eta value.
        quantize_x0 (bool): Whether to quantize the initial sample.
        custom_shape (Optional[Tuple[int, int]]): The custom shape.
        temperature (float): The temperature value.
        noise_dropout (float): The noise dropout value.
        corrector (Optional[nn.Module]): The corrector module.
        corrector_kwargs (Optional[dict]): The corrector module keyword arguments.
        x_T (Optional[Tensor]): The target tensor.
        ddim_use_x0_pred (bool): Whether to use x0 prediction for DDim.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: The generated sample tensor and the
        target tensor (if provided).
    """
    # create an empty dictionary to store the log
    log = dict()

    # get the input data and conditioning from the model
    z, c, x, xrec, xc = model.get_input(
        batch,
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=not (
            hasattr(model, "split_input_params")
            and model.cond_stage_key == "coordinates_bbox"
        ),
        return_original_cond=True,
    )

    # if custom_shape is not None, generate random noise of the specified shape
    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    # store the input and reconstruction in the log
    log["input"] = x
    log["reconstruction"] = xrec

    # sample from the model using convsample_ddim
    with model.ema_scope("Plotting"):
        t0 = time.time()
        sample, intermediates = convsample_ddim(
            model=model,
            cond=c,
            steps=custom_steps,
            shape=z.shape,
            eta=eta,
            quantize_x0=quantize_x0,
            noise_dropout=noise_dropout,
            mask=None,
            x0=None,
            temperature=temperature,
            score_corrector=corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
        )
        t1 = time.time()

        # if ddim_use_x0_pred is True, use the predicted x0 from the intermediates
        if ddim_use_x0_pred:
            sample = intermediates["pred_x0"][-1]

    # decode the sample to get the generated image
    x_sample = model.decode_first_stage(sample)

    # try to decode the sample without quantization to get the unquantized image
    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    # store the generated image, time taken, and other information in the log
    log["sample"] = x_sample
    log["time"] = t1 - t0

    # return the log
    return log


def disabled_train(self: nn.Module, mode: bool = True) -> nn.Module:
    """
    Overwrites the `train` method of the model to disable changing the mode.

    Args:
        mode (bool): Whether to enable or disable training mode.

    Returns:
        nn.Module: The model.
    """
    return self


def convsample_ddim(
    model: nn.Module,
    cond: Tensor,
    steps: int,
    shape: Tuple[int, int],
    eta: float = 1.0,
    callback: Optional[callable] = None,
    noise_dropout: float = 0.0,
    normals_sequence: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    x0: Optional[Tensor] = None,
    quantize_x0: bool = False,
    temperature: float = 1.0,
    score_corrector: Optional[nn.Module] = None,
    corrector_kwargs: Optional[dict] = None,
    x_T: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Generates a convolutional sample using the given model and conditioning tensor.

    Args:
        model (nn.Module): The model to use for sampling.
        cond (Tensor): The conditioning tensor.
        steps (int): The number of steps.
        shape (Tuple[int, int]): The shape of the sample.
        eta (float): The eta value.
        callback (Optional[callable]): The callback function.
        normals_sequence (Optional[Tensor]): The normals sequence tensor.
        noise_dropout (float): The noise dropout value.
        mask (Optional[Tensor]): The mask tensor.
        x0 (Optional[Tensor]): The initial sample tensor.
        quantize_x0 (bool): Whether to quantize the initial sample.
        temperature (float): The temperature value.
        score_corrector (Optional[nn.Module]): The score corrector module.
        corrector_kwargs (Optional[dict]): The score corrector module keyword arguments.
        x_T (Optional[Tensor]): The target tensor.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: The generated sample tensor and the target tensor (if provided).
    """
    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(
        steps,
        batch_size=bs,
        shape=shape,
        conditioning=cond,
        callback=callback,
        normals_sequence=normals_sequence,
        quantize_x0=quantize_x0,
        eta=eta,
        mask=mask,
        x0=x0,
        temperature=temperature,
        verbose=False,
        score_corrector=score_corrector,
        noise_dropout=noise_dropout,
        corrector_kwargs=corrector_kwargs,
        x_T=x_T,
    )

    return samples, intermediates


def make_ddim_sampling_parameters(
    alphacums: np.ndarray, ddim_timesteps: np.ndarray, eta: float, verbose: bool = True
) -> tuple:
    """
    Computes the variance schedule for the ddim sampler, based on the given parameters.

    Args:
        alphacums (np.ndarray): Array of cumulative alpha values.
        ddim_timesteps (np.ndarray): Array of timesteps to use for computing alphas.
        eta (float): Scaling factor for computing sigmas.
        verbose (bool, optional): Whether to print out the selected alphas and sigmas. Defaults to True.

    Returns:
        tuple: A tuple containing three arrays: sigmas, alphas, and alphas_prev.
            sigmas (np.ndarray): Array of sigma values for each timestep.
            alphas (np.ndarray): Array of alpha values for each timestep.
            alphas_prev (np.ndarray): Array of alpha values for the previous timestep.
    """
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(
            f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
        )
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


def make_ddim_timesteps(
    ddim_discr_method: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Computes the timesteps to use for computing alphas in the ddim sampler.

    Args:
        ddim_discr_method (str): The method to use for discretizing the timesteps.
            Must be either 'uniform' or 'quad'.
        num_ddim_timesteps (int): The number of timesteps to use for computing alphas.
        num_ddpm_timesteps (int): The total number of timesteps in the DDPM model.
        verbose (bool, optional): Whether to print out the selected timesteps. Defaults to True.

    Returns:
        np.ndarray: An array of timesteps to use for computing alphas in the ddim sampler.
    """
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def noise_like(shape: tuple, device: str, repeat: bool = False) -> torch.Tensor:
    """
    Generates noise with the same shape as the given tensor.

    Args:
        shape (tuple): The shape of the tensor to generate noise for.
        device (str): The device to place the noise tensor on.
        repeat (bool, optional): Whether to repeat the same noise across the batch dimension. Defaults to False.

    Returns:
        torch.Tensor: A tensor of noise with the same shape as the input tensor.
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class DDIMSampler(object):
    def __init__(self, model: object, schedule: str = "linear", **kwargs: dict) -> None:
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = model.device

    def register_buffer(self, name: str, attr: torch.Tensor) -> None:
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.device):
                attr = attr.to(torch.device(self.device))
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        # make ddim timesteps. these are the timesteps at which we compute alphas
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )

        # get alphas_cumprod from the model
        alphas_cumprod = self.model.alphas_cumprod

        # check if alphas_cumprod is defined for each timestep
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"

        # define a function to convert tensor to torch tensor
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        # register buffers for betas, alphas_cumprod, and alphas_cumprod_prev
        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

        # calculate sigmas for original sampling steps
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    def sample(
        self,
        S: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        conditioning: Optional[torch.Tensor] = None,
        callback: Optional[callable] = None,
        img_callback: Optional[callable] = None,
        quantize_x0: bool = False,
        eta: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Optional[callable] = None,
        corrector_kwargs: Optional[dict] = None,
        verbose: bool = True,
        x_T: Optional[torch.Tensor] = None,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Samples from the model using DDIM sampling.

        Args:
            S (int): Number of DDIM steps.
            batch_size (int): Batch size.
            shape (Tuple[int, int, int]): Shape of the output tensor.
            conditioning (Optional[torch.Tensor], optional): Conditioning tensor. Defaults to None.
            callback (Optional[callable], optional): Callback function. Defaults to None.
            img_callback (Optional[callable], optional): Image callback function. Defaults to None.
            quantize_x0 (bool, optional): Whether to quantize the denoised image. Defaults to False.
            eta (float, optional): Learning rate for DDIM. Defaults to 0..
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.
            x0 (Optional[torch.Tensor], optional): Initial tensor. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 1..
            noise_dropout (float, optional): Noise dropout rate. Defaults to 0..
            score_corrector (Optional[callable], optional): Score corrector function. Defaults to None.
            corrector_kwargs (Optional[dict], optional): Keyword arguments for the score corrector function.
                Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            x_T (Optional[torch.Tensor], optional): Target tensor. Defaults to None.
            log_every_t (int, optional): Log every t steps. Defaults to 100.
            unconditional_guidance_scale (float, optional): Scale for unconditional guidance. Defaults to 1..
            unconditional_conditioning (Optional[torch.Tensor], optional): Unconditional conditioning tensor.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the generated samples and intermediate results.
        """
        # check if conditioning is None
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        # make schedule to compute alphas and sigmas
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        # parameters for sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        # sample from the model using ddim_sampling
        samples, intermediates = self.ddim_sampling(
            cond=conditioning,
            shape=size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )

        return samples, intermediates

    def ddim_sampling(
        self,
        cond: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        shape: Tuple[int, int, int],
        x_T: Optional[torch.Tensor] = None,
        ddim_use_original_steps: bool = False,
        callback: Optional[callable] = None,
        timesteps: Optional[List[int]] = None,
        quantize_denoised: bool = False,
        mask: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        img_callback: Optional[callable] = None,
        log_every_t: int = 100,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector: Optional[callable] = None,
        corrector_kwargs: Optional[Dict[str, Any]] = None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Samples from the model using DDIM sampling.

        Args:
            cond (Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]): Conditioning
                tensor. Defaults to None.
            shape (Tuple[int, int, int]): Shape of the output tensor.
            x_T (Optional[torch.Tensor], optional): Target tensor. Defaults to None.
            ddim_use_original_steps (bool, optional): Whether to use original DDIM steps. Defaults to False.
            callback (Optional[callable], optional): Callback function. Defaults to None.
            timesteps (Optional[List[int]], optional): List of timesteps. Defaults to None.
            quantize_denoised (bool, optional): Whether to quantize the denoised image. Defaults to False.
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.
            x0 (Optional[torch.Tensor], optional): Initial tensor. Defaults to None.
            img_callback (Optional[callable], optional): Image callback function. Defaults to None.
            log_every_t (int, optional): Log every t steps. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 1..
            noise_dropout (float, optional): Noise dropout rate. Defaults to 0..
            score_corrector (Optional[callable], optional): Score corrector function. Defaults to None.
            corrector_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the score corrector
                function. Defaults to None.
            unconditional_guidance_scale (float, optional): Scale for unconditional guidance. Defaults to 1.
            unconditional_conditioning (Optional[torch.Tensor], optional): Unconditional conditioning tensor.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Tuple containing the generated samples and intermediate results.
        """
        # Get the device and batch size
        device = self.model.betas.device
        b = shape[0]

        # Initialize the image tensor
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        # Get the timesteps
        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        # Initialize the intermediates dictionary
        intermediates = {"x_inter": [img], "pred_x0": [img]}

        # Set the time range and total steps
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        # Initialize the progress bar iterator
        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        # Loop over the timesteps
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Sample from the model using DDIM
            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                temperature=temperature,
                noise_dropout=noise_dropout,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            img, pred_x0 = outs


            # Append the intermediate results to the intermediates dictionary
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def p_sample_ddim(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: int,
        index: int,
        repeat_noise: bool = False,
        use_original_steps: bool = False,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the model using DDIM sampling.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.
            t (int): Current timestep.
            index (int): Index of the current timestep.
            repeat_noise (bool, optional): Whether to repeat noise. Defaults to False.
            use_original_steps (bool, optional): Whether to use original DDIM steps.
                Defaults to False.
            quantize_denoised (bool, optional): Whether to quantize the denoised image.
                Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1..
            noise_dropout (float, optional): Noise dropout rate. Defaults to 0..
            score_corrector (Optional[callable], optional): Score corrector function.
                Defaults to None.
            corrector_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments
                for the score corrector function. Defaults to None.
            unconditional_guidance_scale (float, optional): Scale for unconditional
                guidance. Defaults to 1..
            unconditional_conditioning (Optional[torch.Tensor], optional): Unconditional
                conditioning tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the generated samples and intermediate results.
        """
        t = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
                
        # get batch size and device
        b, *_, device = *x.shape, x.device

        # apply model with or without unconditional conditioning
        e_t = self.model.apply_model(x, t, c) 

        # get alphas, alphas_prev, sqrt_one_minus_alphas, and sigmas
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> np.ndarray:
    """
    Creates a beta schedule for the diffusion process.

    Args:
        schedule (str): Type of schedule to use. Can be "linear", "cosine", "sqrt_linear", or "sqrt".
        n_timestep (int): Number of timesteps.
        linear_start (float, optional): Starting value for linear schedule. Defaults to 1e-4.
        linear_end (float, optional): Ending value for linear schedule. Defaults to 2e-2.
        cosine_s (float, optional): Scaling factor for cosine schedule. Defaults to 8e-3.

    Returns:
        np.ndarray: Array of beta values.
    """
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Extracts values from a tensor into a new tensor based on indices.

    Args:
        a (torch.Tensor): Input tensor.
        t (torch.Tensor): Indices tensor.
        x_shape (Tuple[int, ...]): Shape of the output tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class LitEma(nn.Module):
    def __init__(
        self, model: nn.Module, decay: float = 0.9999, use_num_upates: bool = True
    ) -> None:
        """
        Initializes the LitEma class.

        Args:
            model (nn.Module): The model to apply EMA to.
            decay (float, optional): The decay rate for EMA. Must be between 0 and 1. Defaults to 0.9999.
            use_num_upates (bool, optional): Whether to use the number of updates to adjust decay. Defaults to True.
        """
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int)
            if use_num_upates
            else torch.tensor(-1, dtype=torch.int),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model: nn.Module) -> None:
        """
        Applies EMA to the model.

        Args:
            model (nn.Module): The model to apply EMA to.
        """
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(
                        one_minus_decay * (shadow_params[sname] - m_param[key])
                    )
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model: nn.Module) -> None:
        """
        Copies the EMA parameters to the model.

        Args:
            model (nn.Module): The model to copy the EMA parameters to.
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        """
        Saves the current parameters for restoring later.

        Args:
            parameters (Iterable[nn.Parameter]): The parameters to be temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        """
        Restores the parameters stored with the `store` method.

        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters (Iterable[nn.Parameter]): The parameters to be updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        # EMA scopes are temporary. Retaining these full parameter clones after
        # restore wastes another model's worth of accelerator memory and can
        # briefly double that overhead on the next call to store().
        self.collected_params.clear()
