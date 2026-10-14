from typing import Tuple

import numpy as np
import torch
from opensr_model.autoencoder.utils import (
    Downsample,
    Normalize,
    ResnetBlock,
    Upsample,
    make_attn,
    nonlinearity,
)
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, int, int, int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignorekwargs: dict,
    ):
        """
        Encoder module responsible for downsampling and transforming an input image tensor.

        Args:
            ch (int): Base number of channels in the model.
            num_res_blocks (int): Number of residual blocks per resolution.
            attn_resolutions (tuple of int): Resolutions at which attention should be applied.
            in_channels (int): Number of channels in the input data.
            resolution (int): The resolution of the input data.
            z_channels (int): Number of channels for the latent variable 'z'.
            ch_mult (tuple of int, optional): Multipliers for the channels in different blocks. Defaults to (1, 2, 4, 8).
            dropout (float, optional): Dropout rate to use in ResNet blocks. Defaults to 0.0.
            resamp_with_conv (bool, optional): Whether to use convolution for downsampling. Defaults to True.
            double_z (bool, optional): If True, output channels will be doubled for 'z'. Defaults to True.
            use_linear_attn (bool, optional): If True, linear attention will be used. Overrides 'attn_type'. Defaults to False.
            attn_type (str, optional): Type of attention mechanism. Options are "vanilla" or "linear". Defaults to "vanilla".
            ignorekwargs (dict): Ignore extra keyword arguments.

        Examples:
            >>> encoder = Encoder(in_channels=3, z_channels=64, ch=32, resolution=64, num_res_blocks=2, attn_resolutions=(16, 8))
            >>> x = torch.randn(1, 3, 64, 64)
            >>> z = encoder(x)
        """
        super().__init__()

        # If linear attention is used, override the attention type.
        if use_linear_attn:
            attn_type = "linear"

        # Setting global attributes to create the encoder.
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Initial convolution for spectral reduction.
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # Downsampling with residual blocks and optionally attention
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Upsampling with residual blocks and optionally attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # Final convolution to get the latent variable 'z'
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
        x: Input tensor.

        Returns:
        Transformed tensor after passing through the Encoder.
        """

        # timestep embedding (if needed in the next Diffusion runs!)
        temb = None

        # Initial downsampling
        hs = [self.conv_in(x)]

        # Downsampling through the layers
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle processing with blocks and attention
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Final transformation to produce the output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, int, int, int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignorekwargs: dict,
    ):
        """
        A Decoder class that converts a given encoded data 'z' back to its original state.

        Args:
            ch (int): Number of channels in the input data.
            out_ch (int): Number of channels in the output data.
            num_res_blocks (int): Number of residual blocks in the network.
            attn_resolutions (Tuple[int, ...]): Resolutions at which attention mechanisms are applied.
            in_channels (int): Number of channels in the encoded data 'z'.
            resolution (int): The resolution of the output image.
            z_channels (int): Number of channels in the latent space representation.
            ch_mult (Tuple[int, int, int, int], optional): Multiplier for channels at different resolution
                levels. Defaults to (1, 2, 4, 8).
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
            resamp_with_conv (bool, optional): Whether to use convolutional layers for upsampling. Defaults to True.
            give_pre_end (bool, optional): If set to True, returns the output before the last layer. Useful for further
                processing. Defaults to False.
            tanh_out (bool, optional): If set to True, applies tanh activation function to the output. Defaults to False.
            use_linear_attn (bool, optional): If set to True, uses linear attention mechanism. Defaults to False.
            attn_type (str, optional): Type of attention mechanism used ("vanilla" or "linear"). Defaults to "vanilla".
            ignorekwargs (dict): Ignore extra keyword arguments.

        Examples:
            >>> decoder = Decoder(
                    ch=32, out_ch=3, z_channels=64, resolution=64,
                    in_channels=64, num_res_blocks=2,
                    attn_resolutions=(16, 8)
                )
            >>> z = torch.randn(1, 64, 8, 8)
            >>> x_reconstructed = decoder(z)
        """

        super().__init__()

        # If linear attention is required, set attention type as 'linear'
        if use_linear_attn:
            attn_type = "linear"

        # Initialize basic attributes for Decoding
        self.ch = ch
        self.temb_ch = 0  # Temporal embedding channel
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end  # Controls the final output
        self.tanh_out = tanh_out  # Apply tanh activation at the end

        # Compute input channel multiplier, initial block input channel and current resolution
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # Display z-shape details
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # Conversion layer: From z dimension to block input channels
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # Middle processing blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # Upsampling layers
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            # Apply ResNet blocks and attention at each resolution
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()
            up.block = block
            up.attn = attn

            # Upsampling operations
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2

            # Keep the order consistent with original resolutions
            self.up.insert(0, up)

        # Final normalization and conversion layers
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Decoder.

        Args:
            z (torch.Tensor): The latent variable 'z' to be decoded.

        Returns:
            torch.Tensor: Transformed tensor after passing through the Decoder.
        """

        self.last_z_shape = z.shape

        # Time-step embedding (not used, in the Decoder part)
        temb = None

        # Convert z to block input
        h = self.conv_in(z)

        # Middle processing blocks
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling steps
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Final output steps
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # Apply tanh activation if required
        if self.tanh_out:
            h = torch.tanh(h)

        return h


class DiagonalGaussianDistribution(object):
    """
    Represents a multi-dimensional diagonal Gaussian distribution.

    The distribution is parameterized by means and diagonal variances
    (or standard deviations) for each dimension. This means that the
    covariance matrix of this Gaussian distribution is diagonal
    (i.e., non-diagonal elements are zero).

    Attributes:
        parameters (torch.Tensor): A tensor containing concatenated means and log-variances.
        mean (torch.Tensor): The mean of the Gaussian distribution.
        logvar (torch.Tensor): The logarithm of variances of the Gaussian distribution.
        deterministic (bool): If true, the variance is set to zero, making the distribution
            deterministic.
        std (torch.Tensor): The standard deviation of the Gaussian distribution.
        var (torch.Tensor): The variance of the Gaussian distribution.

    Examples:
        >>> params = torch.randn((1, 10))  # Assuming 5 for mean and 5 for log variance
        >>> dist = DiagonalGaussianDistribution(params)
        >>> sample = dist.sample() # Sample from the distribution
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Initializes the DiagonalGaussianDistribution.

        Args:
            parameters (torch.Tensor): A tensor containing concatenated means and log-variances.
            deterministic (bool, optional): If set to true, this distribution becomes
                deterministic (i.e., has zero variance).
        """
        self.parameters = parameters
        self.deterministic = deterministic

        # Split the parameters into means and log-variances
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        # Limit the log variance values
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        # Calculate standard deviation & variance from log variance
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        # If deterministic, set variance and standard deviation to zero
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self) -> torch.Tensor:
        """
        Sample from the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled tensor.
        """

        # Sample from a standard Gaussian distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )

        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """
        Compute the KL divergence between this Gaussian distribution and another.

        Args:
            other (DiagonalGaussianDistribution, optional): The other Gaussian
                distribution. If None, computes the KL divergence with a standard
                Gaussian (mean 0, variance 1).

        Returns:
            torch.Tensor: KL divergence values.
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(device=self.parameters.device)
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: list = [1, 2, 3]) -> torch.Tensor:
        """
        Compute the negative log likelihood of a sample under this Gaussian distribution.

        Args:
            sample (torch.Tensor): The input sample tensor.
            dims (list, optional): The dimensions over which the sum is performed. Defaults
                to [1, 2, 3].

        Returns:
            torch.Tensor: Negative log likelihood values.
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(device=self.parameters.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """
        Get the mode of the Gaussian distribution (which is equal to its mean).

        Returns:
            torch.Tensor: The mode (mean) of the Gaussian distribution.
        """
        return self.mean


class AutoencoderKL(nn.Module):
    """
    Autoencoder with KL divergence regularization.

    This class implements an autoencoder model where the encoder outputs parameters of a
    Gaussian distribution, from which the latent representation can be sampled or its
    mode can be taken. The decoder then reconstructs the input from the latent
    representation.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        quant_conv (torch.nn.Conv2d): Convolutional layer used to process encoder outputs
            into Gaussian parameters.
        post_quant_conv (torch.nn.Conv2d): Convolutional layer used after sampling/mode
            from the Gaussian distribution.
        embed_dim (int): Embedding dimension of the latent space.

    Examples:

        >>> ddconfig = {
                "z_channels": 16,  "ch": 32,
                "out_ch": 3,  "ch_mult": (1, 2, 4, 8),
                "resolution": 64, "in_channels": 3,
                "double_z": True, "num_res_blocks": 2,
                "attn_resolutions": (16, 8)
        }
        >>> embed_dim = 8
        >>> ae_model = AutoencoderKL(ddconfig, embed_dim)
        >>> data = torch.randn((1, 3, 64, 64))
        >>> recon_data, posterior = ae_model(data)
    """

    def __init__(self, ddconfig: dict, embed_dim: int):
        """
        Initialize the AutoencoderKL.

        Args:
            ddconfig (dict): Configuration dictionary for the encoder and decoder.
            embed_dim (int): Embedding dimension of the latent space.
        """
        super().__init__()

        # Initialize the encoder and decoder with provided configurations
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # Check if the configuration expects double the z_channels
        assert ddconfig["double_z"], "ddconfig must have 'double_z' set to True."

        # Define convolutional layers to transform between the latent space and Gaussian parameters
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        Pass the input through the encoder and return the posterior Gaussian
            distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            DiagonalGaussianDistribution: Gaussian distribution parameters from the
                encoded input.
        """
        # Encoder's output
        h = self.encoder(x)

        # Convert encoder's output to Gaussian parameters
        moments = self.quant_conv(h)

        # Create a DiagonalGaussianDistribution using the moments
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation to reconstruct the input.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        # Process latent representation through a convolutional layer
        z = self.post_quant_conv(z)

        # Decoder's output
        dec = self.decoder(z)

        return dec

    def forward(self, input: torch.Tensor, sample_posterior: bool = True) -> tuple:
        """
        Forward pass of the autoencoder.

        Encodes the input, samples/modes from the resulting Gaussian distribution,
        and then decodes to get the reconstructed input.

        Args:
            input (torch.Tensor): Input tensor.
            sample_posterior (bool, optional): If True, sample from the posterior Gaussian
                distribution. If False, use its mode. Defaults to True.

        Returns:
            tuple: Reconstructed tensor and the posterior Gaussian distribution.
        """

        # Encode the input to get the Gaussian distribution parameters
        posterior = self.encode(input)

        # Sample from the Gaussian distribution or take its mode
        z = posterior.sample() if sample_posterior else posterior.mode()

        # Decode the sampled/mode latent representation
        dec = self.decode(z)

        return dec, posterior
