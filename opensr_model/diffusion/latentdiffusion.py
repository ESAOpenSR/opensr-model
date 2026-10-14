from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from opensr_model.autoencoder.autoencoder import (
    AutoencoderKL,
    DiagonalGaussianDistribution,
)
from opensr_model.denoiser.unet import UNetModel
from opensr_model.diffusion.utils import (
    LitEma,
    count_params,
    default,
    disabled_train,
    exists,
    extract_into_tensor,
    make_beta_schedule,
    make_convolutional_sample,
)

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DiffusionWrapper(nn.Module):
    """
    A wrapper around a UNetModel that supports different types of conditioning.

    Args:
        diff_model_config (dict): A dictionary of configuration options for the UNetModel.
        conditioning_key (str, optional): The type of conditioning to use
            (None, 'concat', 'crossattn', 'hybrid', or 'adm'). Defaults to None.

    Raises:
        AssertionError: If the conditioning key is not one of the supported values.

    Example:
        >>> diff_model_config = {'in_channels': 3, 'out_channels': 3, 'num_filters': 32}
        >>> wrapper = DiffusionWrapper(diff_model_config, conditioning_key='concat')
        >>> x = torch.randn(1, 3, 256, 256)
        >>> t = torch.randn(1)
        >>> c_concat = [torch.randn(1, 32, 256, 256)]
        >>> y = wrapper(x, t, c_concat=c_concat)
    """

    def __init__(self, diff_model_config: dict, conditioning_key: Optional[str] = None):
        super().__init__()
        self.diffusion_model = UNetModel(**diff_model_config)
        self.conditioning_key = conditioning_key

        ckey_options = [None, "concat", "crossattn", "hybrid", "adm"]
        assert (
            self.conditioning_key in ckey_options
        ), f"Unsupported conditioning key: {self.conditioning_key}"

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[List[torch.Tensor]] = None,
        c_crossattn: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply the diffusion model to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The diffusion time.
            c_concat (List[torch.Tensor], optional): A list of tensors to concatenate with the input tensor.
                Used when conditioning_key is 'concat'. Defaults to None.
            c_crossattn (List[torch.Tensor], optional): A list of tensors to use for cross-attention.
                Used when conditioning_key is 'crossattn', 'hybrid', or 'adm'. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            NotImplementedError: If the conditioning key is not one of the supported values.
        """
        xc = torch.cat([x] + c_concat, dim=1)
        out = self.diffusion_model(xc, t)
        return out


class DDPM(nn.Module):
    """This class implements the classic DDPM (Diffusion Models) with Gaussian diffusion
    in image space.

    Args:
        unet_config (dict): A dictionary of configuration options for the UNetModel.
        timesteps (int): The number of diffusion timesteps to use.
        beta_schedule (str): The type of beta schedule to use (linear, cosine, or fixed).
        use_ema (bool): Whether to use exponential moving averages (EMAs) of the model weights during training.
        first_stage_key (str): The key to use for the first stage of the model (either "image" or "noise").
        linear_start (float): The starting value for the linear beta schedule.
        linear_end (float): The ending value for the linear beta schedule.
        cosine_s (float): The scaling factor for the cosine beta schedule.
        given_betas (list): A list of beta values to use for the fixed beta schedule.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.
        conditioning_key (str): The type of conditioning to use (None, 'concat', 'crossattn', 'hybrid', or 'adm').
        parameterization (str): The type of parameterization to use for the diffusion process (either "eps" or "x0").
        use_positional_encodings (bool): Whether to use positional encodings for the input.

    Methods:
        register_schedule: Registers the schedule for the betas and alphas.
        get_input: Gets the input from the DataLoader and rearranges it.
        decode_first_stage: Decodes the first stage of the model.
        ema_scope: Switches to EMA weights during training.

    Attributes:
        parameterization (str): The type of parameterization used for the diffusion process.
        cond_stage_model (None): The conditioning stage model (not used in this implementation).
        first_stage_key (str): The key used for the first stage of the model.
        use_positional_encodings (bool): Whether positional encodings are used for the input.
        model (DiffusionWrapper): The diffusion model.
        use_ema (bool): Whether EMAs of the model weights are used during training.
        model_ema (LitEma): The EMA of the model weights.
        v_posterior (float): The weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.

    Example:
        >>> unet_config = {
            'in_channels': 3,
            'model_channels': 160,
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'channel_mult': [1, 2, 2, 4],
            'num_head_channels': 32
        }
        >>> model = DDPM(
                unet_config, timesteps=1000, beta_schedule='linear',
                use_ema=True, first_stage_key='image'
            )
    """

    def __init__(
        self,
        unet_config: Dict[str, Any],
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        use_ema: bool = True,
        first_stage_key: str = "image",
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        given_betas: Optional[List[float]] = None,
        v_posterior: float = 0.0,
        conditioning_key: Optional[str] = None,
        parameterization: str = "eps",
        use_positional_encodings: bool = False,
    ) -> None:
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization

        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )

        self.cond_stage_model = None
        self.first_stage_key = first_stage_key
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)

        count_params(self.model, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas: Optional[List[float]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the schedule for the betas and alphas.

        Args:
            given_betas (list, optional): A list of beta values to use for the fixed beta schedule.
                Defaults to None.
            beta_schedule (str, optional): The type of beta schedule to use (linear, cosine, or fixed).
                Defaults to "linear".
            timesteps (int, optional): The number of diffusion timesteps to use. Defaults to 1000.
            linear_start (float, optional): The starting value for the linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): The ending value for the linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): The scaling factor for the cosine beta schedule. Defaults to 8e-3.
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_input(self, batch: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        """
        Gets the input from the DataLoader and rearranges it.

        Args:
            batch (Dict[str, torch.Tensor]): The batch of data from the DataLoader.
            k (str): The key for the input tensor in the batch.

        Returns:
            torch.Tensor: The input tensor, rearranged and converted to float.
        """

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.to(memory_format=torch.contiguous_format).float()

        return x

    @contextmanager
    def ema_scope(self, context: Optional[str] = None) -> Generator[None, None, None]:
        """
        A context manager that switches to EMA weights during training.

        Args:
            context (Optional[str]): A string to print when switching to and from EMA weights.

        Yields:
            None
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the first stage of the model.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The decoded output tensor.
        """

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded

            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)


class LatentDiffusion(DDPM):
    """
    LatentDiffusion is a class that extends the DDPM class and implements a diffusion
    model with a latent variable. The model consists of two stages: a first stage that
    encodes the input tensor into a latent tensor, and a second stage that decodes the
    latent tensor into the output tensor. The model also has a conditional stage that
    takes a conditioning tensor as input and produces a learned conditioning tensor
    that is used to condition the first and second stages of the model. The class
    provides methods for encoding and decoding tensors, computing the output tensor
    and loss, and sampling from the distribution at a given latent tensor and timestep.
    The class also provides methods for registering and applying schedules, and for
    getting and setting the scale factor and conditioning key.

    Methods:
        register_schedule(self, schedule: Schedule) -> None: Registers the given schedule
            with the model.
        make_cond_schedule(self, schedule: Schedule) -> Schedule: Returns a new schedule
            with the given schedule applied to the conditional stage of the model.
        encode_first_stage(self, x: torch.Tensor, t: int) -> torch.Tensor: Encodes the given
            input tensor with the first stage of the model for the given timestep.
        get_first_stage_encoding(self, x: torch.Tensor, t: int) -> torch.Tensor: Returns the
            encoding of the given input tensor with the first stage of the model for the
            given timestep.
        get_learned_conditioning(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the learned conditioning tensor for the given input
            tensor, timestep, and conditioning tensor.
        get_input(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
            Returns the input tensor for the given input tensor, timestep, and
            conditioning tensor.
        compute(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Computes the output tensor and loss for the given input tensor,
            timestep, and conditioning tensor.
        apply_model(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Applies
            the model to the given input tensor, timestep, and conditioning tensor.
        get_fold_unfold(self, ks: int, stride: int, vqf: int) -> Tuple[Callable, Callable]: Returns the fold
            and unfold functions for the given kernel size, stride, and vector quantization factor.
        forward(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor: Computes the
            output tensor for the given input tensor, timestep, and conditioning tensor.
        q_sample(self, z: torch.Tensor, t: int, eps: Optional[torch.Tensor] = None) -> torch.Tensor: Samples
            from the distribution at the given latent tensor and timestep.
    """

    def __init__(
        self,
        first_stage_config: Dict[str, Any],
        cond_stage_config: Union[str, Dict[str, Any]],
        num_timesteps_cond: Optional[int] = None,
        cond_stage_key: str = "image",
        cond_stage_trainable: bool = False,
        concat_mode: bool = True,
        cond_stage_forward: Optional[Callable] = None,
        conditioning_key: Optional[str] = None,
        scale_factor: float = 1.0,
        scale_by_std: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the LatentDiffusion model.

        Args:
            first_stage_config (Dict[str, Any]): The configuration for the first stage of the model.
            cond_stage_config (Union[str, Dict[str, Any]]): The configuration for the conditional stage of the model.
            num_timesteps_cond (Optional[int]): The number of timesteps for the conditional stage of the model.
            cond_stage_key (str): The key for the conditional stage of the model.
            cond_stage_trainable (bool): Whether the conditional stage of the model is trainable.
            concat_mode (bool): Whether to use concatenation or cross-attention for the conditioning.
            cond_stage_forward (Optional[Callable]): A function to apply to the output of the conditional stage of the model.
            conditioning_key (Optional[str]): The key for the conditioning.
            scale_factor (float): The scale factor for the input tensor.
            scale_by_std (bool): Whether to scale the input tensor by its standard deviation.
            *args (Any): Additional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]

        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward

        # Set Fusion parameters (SIMON)
        # TODO: We only have SISR parameters
        self.sr_type = "SISR"

        # Setup the AutoencoderKL model
        embed_dim = first_stage_config[
            "embed_dim"
        ]  # extract embedded dim fro first stage config
        self.first_stage_model = AutoencoderKL(first_stage_config, embed_dim=embed_dim)
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        # Setup the Unet model
        self.cond_stage_model = torch.nn.Identity()  # Unet
        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def register_schedule(
        self,
        given_betas: Optional[Union[float, torch.Tensor]] = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ) -> None:
        """
        Registers the given schedule with the model.

        Args:
            given_betas (Optional[Union[float, torch.Tensor]]): The betas for the schedule.
            beta_schedule (str): The type of beta schedule to use.
            timesteps (int): The number of timesteps for the schedule.
            linear_start (float): The start value for the linear schedule.
            linear_end (float): The end value for the linear schedule.
            cosine_s (float): The scale factor for the cosine schedule.
        """
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def make_cond_schedule(self) -> None:
        """
        Shortens the schedule for the conditional stage of the model.
        """
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given input tensor with the first stage of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded output tensor.
        """
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(
        self, encoder_posterior: Union[DiagonalGaussianDistribution, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns the encoding of the given input tensor with the first stage of the
        model for the given timestep.

        Args:
            encoder_posterior (Union[DiagonalGaussianDistribution, torch.Tensor]): The
                encoder posterior.

        Returns:
            torch.Tensor: The encoding of the input tensor.
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c: torch.Tensor) -> torch.Tensor:
        """
        Returns the learned conditioning tensor for the given input tensor.

        Args:
            c (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The learned conditioning tensor.
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
                # cond stage model is identity
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_input(
        self,
        batch: torch.Tensor,
        k: int,
        return_first_stage_outputs: bool = False,
        force_c_encode: bool = False,
        cond_key: Optional[str] = None,
        return_original_cond: bool = False,
        bs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns the input tensor for the given batch and timestep.

        Args:
            batch (torch.Tensor): The input batch tensor.
            k (int): The timestep.
            return_first_stage_outputs (bool): Whether to return the outputs of the first stage of the model.
            force_c_encode (bool): Whether to force encoding of the conditioning tensor.
            cond_key (Optional[str]): The key for the conditioning tensor.
            return_original_cond (bool): Whether to return the original conditioning tensor.
            bs (Optional[int]): The batch size.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: The input tensor, the outputs of the
            first stage of the model (if `return_first_stage_outputs` is `True`), and the encoded conditioning tensor
            (if `force_c_encode` is `True` and `cond_key` is not `None`).
        """

        # k = first_stage_key on this SR example
        x = super().get_input(batch, k)  # line 333

        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        # perform always for HR and for HR only of SISR
        if self.sr_type == "SISR" or k == "image":
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            # self.model.conditioning_key = "image" in SR example

            if cond_key is None:
                cond_key = self.cond_stage_key

            if cond_key != self.first_stage_key:
                if cond_key in ["caption", "coordinates_bbox"]:
                    xc = batch[cond_key]
                elif cond_key == "class_label":
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            # BUG if use_positional_encodings is True
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, "pos_x": pos_x, "pos_y": pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)

        """
        # overwrite LR original with encoded LR if wanted
        self.encode_conditioning = True
        if self.encode_conditioning==True and self.sr_type=="SISR":
            #print("Encoding conditioning!")
            # try to upsample->encode conditioning
            c = torch.nn.functional.interpolate(out[1], size=(512,512), mode='bilinear', align_corners=False)
            # encode c
            c = self.encode_first_stage(c).sample()
            out[1] = c
        """

        return out

    def compute(
        self, example: torch.Tensor, custom_steps: int = 200, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Performs inference on the given example tensor.

        Args:
            example (torch.Tensor): The example tensor.
            custom_steps (int): The number of steps to perform.
            temperature (float): The temperature to use.

        Returns:
            torch.Tensor: The output tensor.
        """
        guider = None
        ckwargs = None
        ddim_use_x0_pred = False
        temperature = temperature
        eta = 1.0
        custom_shape = None

        if hasattr(self, "split_input_params"):
            delattr(self, "split_input_params")

        logs = make_convolutional_sample(
            example,
            self,
            custom_steps=custom_steps,
            eta=eta,
            quantize_x0=False,
            custom_shape=custom_shape,
            temperature=temperature,
            noise_dropout=0.0,
            corrector=guider,
            corrector_kwargs=ckwargs,
            x_T=None,
            ddim_use_x0_pred=ddim_use_x0_pred,
        )

        return logs["sample"]

    def apply_model(
        self,
        x_noisy: torch.Tensor,
        t: int,
        cond: Optional[torch.Tensor] = None,
        return_ids: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies the model to the given noisy input tensor.

        Args:
            x_noisy (torch.Tensor): The noisy input tensor.
            t (int): The timestep.
            cond (Optional[torch.Tensor]): The conditioning tensor.
            return_ids (bool): Whether to return the IDs of the diffusion process.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor, and optionally the IDs of the
            diffusion process.
        """

        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_fold_unfold(
        self, x: torch.Tensor, kernel_size: int, stride: int, uf: int = 1, df: int = 1
    ) -> Tuple[nn.Conv2d, nn.ConvTranspose2d]:
        """
        Returns the fold and unfold convolutional layers for the given input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            kernel_size (int): The kernel size.
            stride (int): The stride.
            uf (int): The unfold factor.
            df (int): The fold factor.

        Returns:
            Tuple[nn.Conv2d, nn.ConvTranspose2d]: The fold and unfold convolutional layers.
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Computes the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The conditioning tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:  # This is FALSE in our case
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option # TRUE in our case
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Samples from the posterior distribution at the given timestep.

        Args:
            x_start (torch.Tensor): The starting tensor.
            t (int): The timestep.
            noise (Optional[torch.Tensor]): The noise tensor.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
