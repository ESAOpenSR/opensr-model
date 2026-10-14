from typing import List, Optional, Set, Tuple, Union

import torch
import torch as th
from einops import rearrange
from opensr_model.denoiser.utils import (
    BasicTransformerBlock,
    Downsample,
    Normalize,
    QKVAttention,
    QKVAttentionLegacy,
    TimestepBlock,
    Upsample,
    checkpoint,
    conv_nd,
    convert_module_to_f16,
    convert_module_to_f32,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from torch import nn


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        """
        A residual block with optional timestep conditioning.

        Args:
            channels (int): The number of input channels.
            emb_channels (int): The number of timestep embedding channels.
            dropout (float): The dropout probability.
            out_channels (int, optional): The number of output channels.
                Defaults to None (same as input channels).
            use_conv (bool, optional): Whether to use a convolutional skip connection.
                Defaults to False.
            use_scale_shift_norm (bool, optional): Whether to use scale-shift normalization.
                Defaults to False.
            dims (int, optional): The number of dimensions in the input tensor.
                Defaults to 2.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory.
                Defaults to False.
            up (bool, optional): Whether to use upsampling in the skip connection. Defaults to
                False.
            down (bool, optional): Whether to use downsampling in the skip connection. Defaults to
                False.

        Example:
            >>> resblock = ResBlock(channels=64, emb_channels=32, dropout=0.1)
            >>> x = torch.randn(1, 64, 32, 32)
            >>> emb = torch.randn(1, 32)
            >>> out = resblock(x, emb)
            >>> print(out.shape)
        """
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # input layers
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # skip connection
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # output layers
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # Skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x (torch.Tensor): An [N x C x ...] Tensor of features.
            emb (torch.Tensor): An [N x emb_channels] Tensor of timestep embeddings.

        Returns:
            torch.Tensor: An [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            # up/downsampling in skip connection
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # timestep embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # scale-shift normalization
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        # skip connection
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Args:
        channels (int): The number of input channels.
        num_heads (int, optional): The number of attention heads. Defaults to 1.
        num_head_channels (int, optional): The number of channels per attention head.
            If not specified, the input channels will be divided equally among the heads.
            Defaults to -1.
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory.
            Defaults to False.
        use_new_attention_order (bool, optional): Whether to split the qkv tensor before
            splitting the heads. If False, the heads will be split before the qkv tensor.
            Defaults to False.

    Example:
        >>> attention_block = AttentionBlock(channels=64, num_heads=4)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = attention_block(x)
    """

    def __init__(
        self,
        channels: int,
        num_heads: Optional[int] = 1,
        num_head_channels: Optional[int] = -1,
        use_checkpoint: Optional[bool] = False,
        use_new_attention_order: Optional[bool] = False,
    ) -> None:
        super().__init__()

        # Set the number of input channels and attention heads
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        # Set whether to use checkpointing and create normalization layer
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        # Create convolutional layer for qkv tensor and attention module
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        # Create convolutional layer for output projection
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the attention block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return checkpoint(self._forward, (x,), self.parameters(), False)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the attention block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        # Apply normalization and convolutional layer to qkv tensor
        qkv = self.qkv(self.norm(x))

        # Apply attention module and convolutional layer to output
        h = self.attention(qkv)
        h = self.proj_out(h)

        # Add input tensor to output and reshape
        return (x + h).reshape(b, c, *spatial)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image.

    Args:
        in_channels (int): The number of input channels.
        n_heads (int): The number of attention heads.
        d_head (int): The number of channels per attention head.
        depth (int, optional): The number of transformer blocks. Defaults to 1.
        dropout (float, optional): The dropout probability. Defaults to 0.
        context_dim (int, optional): The dimension of the context tensor.
            If not specified, cross-attention defaults to self-attention.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Set the number of input channels and attention heads
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        # Create convolutional layer for input projection
        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        # Create list of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        # Create convolutional layer for output projection
        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the spatial transformer block to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            context (torch.Tensor, optional): The context tensor. If not specified,
                cross-attention defaults to self-attention. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        # Apply input projection and reshape
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Reshape and apply output projection
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)

        # Add input tensor to output
        return x + x_in


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.

    Args:
        nn.Sequential: The sequential module.
        TimestepBlock: The timestep block module.

    Example:
        >>> model = TimestepEmbedSequential(
            ResBlock(channels=64, emb_channels=32, dropout=0.1),
            ResBlock(channels=64, emb_channels=32, dropout=0.1)
        )
        >>> x = torch.randn(1, 64, 32, 32)
        >>> emb = torch.randn(1, 32)
        >>> out = model(x, emb)
        >>> print(out.shape)

    """

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the sequential module to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            emb (torch.Tensor): The timestep embedding tensor.
            context (torch.Tensor, optional): The context tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Args:
        in_channels (int): The number of channels in the input tensor.
        model_channels (int): The base channel count for the model.
        out_channels (int): The number of channels in the output tensor.
        num_res_blocks (int): The number of residual blocks per downsample.
        attention_resolutions (Union[Set[int], List[int], Tuple[int]]): A collection
            of downsample rates at which attention will take place. For example, if
            this contains 4, then at 4x downsampling, attention will be used.
        dropout (float, optional): The dropout probability. Defaults to 0.
        channel_mult (Tuple[int], optional): The channel multiplier for each level
            of the UNet. Defaults to (1, 2, 4, 8).
        conv_resample (bool, optional): If True, use learned convolutions for upsampling
            and downsampling. Defaults to True.
        dims (int, optional): Determines if the signal is 1D, 2D, or 3D. Defaults to 2.
        num_classes (int, optional): If specified, then this model will be class-conditional
            with `num_classes` classes. Defaults to None.
        use_checkpoint (bool, optional): Use gradient checkpointing to reduce memory usage.
            Defaults to False.
        use_fp16 (bool, optional): Use half-precision floating point. Defaults to False.
        num_heads (int, optional): The number of attention heads in each attention layer.
            Defaults to -1.
        num_head_channels (int, optional): If specified, ignore num_heads and instead use
            a fixed channel width per attention head. Defaults to -1.
        num_heads_upsample (int, optional): Works with num_heads to set a different number
            of heads for upsampling. Deprecated. Defaults to -1.
        use_scale_shift_norm (bool, optional): Use a FiLM-like conditioning mechanism. Defaults
            to False.
        resblock_updown (bool, optional): Use residual blocks for up/downsampling. Defaults to False.
        use_new_attention_order (bool, optional): Use a different attention pattern for
            potentially increased efficiency. Defaults to False.
        use_spatial_transformer (bool, optional): Use a custom transformer support. Defaults to
            False.
        transformer_depth (int, optional): The depth of the custom transformer support. Defaults
            to 1.
        context_dim (int, optional): The dimension of the context tensor. Defaults to None.
        n_embed (int, optional): Custom support for prediction of discrete ids into codebook
            of first stage vq model. Defaults to None.
        legacy (bool, optional): Use legacy mode. Defaults to True.
        ignorekwargs (dict, optional): Ignore extra keyword arguments.
    Example:
        >>> cond_stage_config = {
                "image_size": 64,
                "in_channels": 8,
                "model_channels": 160,
                "out_channels": 4,
                "num_res_blocks": 2,
                "attention_resolutions": [16, 8],
                "channel_mult": [1, 2, 2, 4],
                "num_head_channels": 32
        }

        >>> model = UNetModel(**cond_stage_config)
        >>> x = torch.randn(2, 8, 128, 128)
        >>> emb = torch.randn(2)
        >>> out = model(x, emb)
        >>> print(out.shape)
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Union[Set[int], List[int], Tuple[int]],
        dropout: float = 0,
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        n_embed: Optional[int] = None,
        legacy: bool = True,
        **ignorekwargs: dict,
    ):
        super().__init__()

        # If num_heads_upsample is not set, set it to num_heads
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # If num_heads is not set, raise an error if num_head_channels is not set
        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        # If num_head_channels is not set, raise an error if num_heads is not set
        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        # Set the instance variables
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        # Set up the time embedding layers
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # If num_classes is not None, set up the label embedding layer
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Set up the input blocks
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        # parameters for the block attention
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Set up the attention blocks
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                # If the downsample rate is in the attention resolutions, add an attention block
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            # If the downsample rate is not the last one, add a downsample block
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Set up the middle block parameters
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        # If use_spatial_transformer is True, set up the spatial transformer
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Set up the output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # If the downsample rate is in the attention resolutions, add an attention block
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                # If the downsample rate is in the attention resolutions, add an attention block
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                # If the downsample rate is the last one, add an upsample block
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # Set up the output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # Set up the codebook id predictor layer
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch), conv_nd(dims, model_channels, n_embed, 1)
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x (torch.Tensor): An [N x C x ...] Tensor of inputs.
            timesteps (torch.Tensor, optional): A 1-D batch of timesteps.
                Defaults to None.
            context (torch.Tensor, optional): Conditioning plugged in via crossattn.
                Defaults to None.
            y (torch.Tensor, optional): An [N] Tensor of labels, if class-conditional.
                Defaults to None.

        Returns:
            torch.Tensor: An [N x C x ...] Tensor of outputs.
        """
        # print("aaa")
        # print(x.shape)
        # print(timesteps.shape)
        # print("aaa")
        # 1 + "a"

        # Check if y is specified only if the model is class-conditional
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Initialize a list to store the hidden states of the input blocks
        hs = []

        # Compute the timestep embeddings and time embeddings
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # Add label embeddings if the model is class-conditional
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # Convert the input tensor to the specified data type
        h = x.type(self.dtype)

        # Pass the input tensor through the input blocks and store the hidden states
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # Pass the output of the input blocks through the middle block
        h = self.middle_block(h, emb, context)

        # Pass the output of the middle block through the output blocks in reverse order
        for module in self.output_blocks:
            # Concatenate the output of the current output block with the corresponding
            # hidden state from the input blocks
            h = th.cat([h, hs.pop()], dim=1)

            # Pass the concatenated tensor through the current output block
            h = module(h, emb, context)

        # Convert the output tensor to the same data type as the input tensor
        h = h.type(x.dtype)

        # Return the output tensor or the codebook ID predictions if specified
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
