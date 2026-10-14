import torch
from einops import rearrange
from torch import nn


def Normalize(in_channels: int, num_groups: int = 32) -> torch.nn.GroupNorm:
    """
    Returns a GroupNorm layer that normalizes the input tensor along the channel dimension.

    Args:
        in_channels (int): Number of channels in the input tensor.
        num_groups (int): Number of groups to separate the channels into. Default is 32.

    Returns:
        torch.nn.GroupNorm: A GroupNorm layer that normalizes the input tensor along the
            channel dimension.

    Example:
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> norm_layer = Normalize(in_channels=64, num_groups=16)
        >>> output_tensor = norm_layer(input_tensor)
    """
    # Create a GroupNorm layer with the specified number of groups and input channels
    # Set eps to a small value to avoid division by zero
    # Set affine to True to learn scaling and shifting parameters
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a non-linear activation function to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.

    Example:
        >>> input_tensor = torch.randn(10, 20)
        >>> output_tensor = nonlinearity(input_tensor)
    """
    # Apply the sigmoid function to the input tensor
    sigmoid_x = torch.sigmoid(x)

    # Multiply the input tensor by the sigmoid of the input tensor
    output_tensor = x * sigmoid_x

    return output_tensor


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        """
        Initializes a Downsample module that reduces the spatial dimensions
        of the input tensor.

        Args:
            in_channels (int): Number of channels in the input tensor.
            with_conv (bool): Whether to use a convolutional layer for downsampling.

        Attributes:
            conv (torch.nn.Conv2d): Convolutional layer for downsampling. Only used
                if with_conv is True.

        Example:
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> downsample_module = Downsample(in_channels=64, with_conv=True)
            >>> output_tensor = downsample_module(input_tensor)
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Create a convolutional layer for downsampling
            # Use kernel size 3, stride 2, and no padding
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Downsample module to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, in_channels, height/2, width/2)
                if with_conv is False, or (batch_size, in_channels, (height+1)/2, (width+1)/2) if
                with_conv is True.
        """
        if self.with_conv:
            # Apply asymmetric padding to the input tensor
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)

            # Apply the convolutional layer to the padded input tensor
            x = self.conv(x)
        else:
            # Apply average pooling to the input tensor
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        # Return the output tensor
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        """
        Initializes an Upsample module that increases the spatial dimensions of
            the input tensor.

        Args:
            in_channels (int): Number of channels in the input tensor.
            with_conv (bool): Whether to use a convolutional layer for upsampling.

        Attributes:
            with_conv (bool): Whether to use a convolutional layer for upsampling.
            conv (torch.nn.Conv2d): Convolutional layer for upsampling. Only used
                if with_conv is True.

        Example:
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> upsample_module = Upsample(in_channels=64, with_conv=True)
            >>> output_tensor = upsample_module(input_tensor)
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # Create a convolutional layer for upsampling
            # Use kernel size 3, stride 1, and padding 1
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Upsample module to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels,
                height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, in_channels, height*2, width*2)
                if with_conv is False, or (batch_size, in_channels, height*2-1, width*2-1) if
                with_conv is True.
        """
        # Apply nearest interpolation to the input tensor to double its spatial dimensions
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            # Apply the convolutional layer to the upsampled input tensor
            x = self.conv(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        """
        Initializes a ResnetBlock module that consists of two convolutional layers with batch
            normalization and a residual connection.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int, optional): Number of channels in the output tensor. If None,
                defaults to in_channels.
            conv_shortcut (bool): Whether to use a convolutional layer for the residual connection.
                If False, uses a 1x1 convolution.
            dropout (float): Dropout probability.
            temb_channels (int): Number of channels in the conditioning tensor. If 0, no conditioning
                is used.

        Attributes:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output tensor.
            use_conv_shortcut (bool): Whether to use a convolutional layer for the residual connection.
            norm1 (utils.Normalize): Batch normalization layer for the first convolutional layer.
            conv1 (torch.nn.Conv2d): First convolutional layer.
            temb_proj (torch.nn.Linear): Linear projection layer for the conditioning tensor. Only used
                if temb_channels > 0.
            norm2 (utils.Normalize): Batch normalization layer for the second convolutional layer.
            dropout (torch.nn.Dropout): Dropout layer.
            conv2 (torch.nn.Conv2d): Second convolutional layer.
            conv_shortcut (torch.nn.Conv2d): Convolutional layer for the residual connection. Only
                used if use_conv_shortcut is True.
            nin_shortcut (torch.nn.Conv2d): 1x1 convolutional layer for the residual connection. Only
                used if use_conv_shortcut is False.
        """
        super().__init__()

        # Set the number of input and output channels
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # Batch normalization layer for the first convolutional layer
        self.norm1 = Normalize(in_channels)

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        # Linear projection layer for the conditioning tensor
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        # BN+Dropout+Conv layer for the last convolutional layer
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 3x3 conv for the residual connection
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                # 1x1 conv for the residual connection
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        """
        Applies the ResnetBlock module to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).
            temb (torch.Tensor): Conditioning tensor with shape (batch_size, temb_channels).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.

        Example:
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> resnet_block = ResnetBlock(in_channels=64, out_channels=128, dropout=0.5)
            >>> output_tensor = resnet_block(input_tensor, temb=None)
        """

        # BN+Sigmoid+Conv
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Linear projection layer for the conditioning tensor
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # BN+Sigmoid+Dropout+Conv
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 3x3 conv for the residual connection
                x = self.conv_shortcut(x)
            else:
                # 1x1 conv for the residual connection
                x = self.nin_shortcut(x)

        # Add the residual connection to the output tensor
        return x + h


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    """
    Creates an attention module of the specified type.

    Args:
        in_channels (int): Number of channels in the input tensor.
        attn_type (str): Type of attention module to create. Must be one of "vanilla",
            "linear", or "none". Defaults to "vanilla".

    Returns:
        nn.Module: Attention module.

    Raises:
        AssertionError: If attn_type is not one of "vanilla", "linear", or "none".

    Example:
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> attn_module = make_attn(in_channels=64, attn_type="vanilla")
        >>> output_tensor = attn_module(input_tensor)
    """
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        # Create a vanilla attention module
        return AttnBlock(in_channels)
    elif attn_type == "none":
        # Create an identity module
        return nn.Identity(in_channels)
    else:
        # Create a linear attention module
        return LinAttnBlock(in_channels)


class AttnBlock(nn.Module):
    """
    An attention module that computes attention weights for each spatial location in the input tensor.

    Args:
        in_channels (int): Number of channels in the input tensor.

    Attributes:
        in_channels (int): Number of channels in the input tensor.
        norm (Normalize): Normalization layer for the input tensor.
        q (torch.nn.Conv2d): Convolutional layer for computing the query tensor.
        k (torch.nn.Conv2d): Convolutional layer for computing the key tensor.
        v (torch.nn.Conv2d): Convolutional layer for computing the value tensor.
        proj_out (torch.nn.Conv2d): Convolutional layer for projecting the attended tensor.

    Example:
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> attn_module = AttnBlock(in_channels=64)
        >>> output_tensor = attn_module(input_tensor)
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        # Runtime-only option: no new parameters or checkpoint keys.
        self.query_chunk_size = None

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output tensor of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # reshape q to b,hw,c and transpose to b,c,hw
        k = k.reshape(b, c, h * w)  # reshape k to b,c,hw
        if self.query_chunk_size is not None:
            values = v.reshape(b, c, h * w).transpose(1, 2)
            chunks = []
            for start in range(0, h * w, self.query_chunk_size):
                weights = torch.bmm(q[:, start : start + self.query_chunk_size], k)
                weights = torch.softmax(weights * (c**-0.5), dim=-1)
                chunks.append(torch.bmm(weights, values))
            attended = torch.cat(chunks, dim=1).transpose(1, 2).reshape(b, c, h, w)
            return x + self.proj_out(attended)
        w_ = torch.bmm(
            q, k
        )  # compute attention weights w[b,i,j] = sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))  # scale the attention weights
        w_ = torch.nn.functional.softmax(
            w_, dim=2
        )  # apply softmax to get the attention probabilities

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # transpose w to b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(
            v, w_
        )  # compute the attended values h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)  # reshape h_ to b,c,h,w
        h_ = self.proj_out(h_)  # project the attended values to the output space

        return x + h_


class LinearAttention(nn.Module):
    """
    A linear attention module that computes attention weights for each spatial
    location in the input tensor.

    Args:
        dim (int): Number of channels in the input tensor.
        heads (int): Number of attention heads. Defaults to 4.
        dim_head (int): Number of channels per attention head. Defaults to 32.

    Example:
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> attn_module = LinearAttention(dim=64, heads=8, dim_head=16)
        >>> output_tensor = attn_module(input_tensor)
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the output tensor of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)
