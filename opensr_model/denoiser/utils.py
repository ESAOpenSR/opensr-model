import math
from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F


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


def conv_nd(dims: int, *args: Union[int, float], **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D convolution module.

    Args:
        dims (int): The number of dimensions for the convolution. Must be 1, 2, or 3.
        *args (Union[int, float]): Positional arguments to pass to the convolution module constructor.
        **kwargs: Keyword arguments to pass to the convolution module constructor.

    Returns:
        nn.Module: A convolution module with the specified number of dimensions.

    Raises:
        ValueError: If the number of dimensions is not 1, 2, or 3.

    Example:
        >>> conv = conv_nd(2, 16, 32, kernel_size=3)
        >>> x = torch.randn(1, 16, 32, 32)
        >>> out = conv(x)
        >>> out.shape
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args: Union[int, float], **kwargs) -> nn.Module:
    """
    Create a linear module.

    Args:
        *args (Union[int, float]): Positional arguments to pass
        to the linear module constructor.
        **kwargs: Keyword arguments to pass to the linear module constructor.

    Returns:
        nn.Module: A linear module.

    Example:
        >>> linear = linear(16, 32)
        >>> x = torch.randn(1, 16)
        >>> out = linear(x)
        >>> out.shape
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.

    Args:
        module (nn.Module): The module to zero out.

    Returns:
        nn.Module: The zeroed-out module.

    Example:
        >>> conv = conv_nd(2, 16, 32, kernel_size=3)
        >>> conv = zero_module(conv)
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor): A 1-D tensor of N indices, one per batch element. These may be fractional.
        dim (int): The dimension of the output.
        max_period (int): Controls the minimum frequency of the embeddings.
        repeat_only (bool): If True, repeat the timestep embeddings instead of computing new ones.

    Returns:
        torch.Tensor: An [N x dim] tensor of positional embeddings.

    Example:
        >>> timesteps = torch.arange(0, 10)
        >>> embeddings = timestep_embedding(timesteps, dim=16)
        >>> embeddings.shape
    """
    if not repeat_only:
        # Compute the frequencies of the sinusoidal embeddings
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)

        # Compute the arguments to the sinusoidal functions
        args = timesteps[:, None].float() * freqs[None]

        # Compute the sinusoidal embeddings
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # If the output dimension is odd, add a zero column to the embeddings
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        # Repeat the timestep embeddings instead of computing new ones
        embedding = repeat(timesteps, "b -> b d", d=dim)

    return embedding


def exists(val: Any) -> bool:
    """
    Check if a value exists (i.e., is not None).

    Args:
        val (Any): The value to check.

    Returns:
        bool: True if the value exists, False otherwise.
    """
    return val is not None


def default(val: Any, d: Any) -> Any:
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val (Any): The value to check.
        d (Any or Callable): The default value to return if `val` does not exist. If `d` is a callable, it will be called
            with no arguments to generate the default value.

    Returns:
        Any: The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if isinstance(d, Callable) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    Args:
        func (callable): The function to evaluate.
        inputs (Sequence): The argument sequence to pass to `func`.
        params (Sequence): A sequence of parameters `func` depends on but does not explicitly take as arguments.
        flag (bool): If False, disable gradient checkpointing.

    Returns:
        Any: The output of the function `func`.

    Example:
        >>> def my_func(x, y, z):
            return x * y + z
        >>> x = torch.randn(32, 64)
        >>> y = torch.randn(32, 64)
        >>> z = torch.randn(32, 64)
        >>> output = checkpoint(func=my_func, inputs=(x, y), params=(z,), flag=True)
        >>> output.shape
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


def normalization(channels: int) -> nn.Module:
    """
    Create a standard normalization layer using group normalization.

    Args:
        channels (int): The number of input channels.

    Returns:
        nn.Module: A group normalization layer with 32 groups and `channels` input channels.

    Example:
        >>> norm = normalization(channels=64)
        >>> x = torch.randn(32, 64, 128, 128)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([32, 64, 128, 128])
    """

    # Create a group normalization layer with 32 groups
    return GroupNorm32(32, channels)


def count_flops_attn(
    model: torch.nn.Module, _x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]
) -> None:
    """
    A counter for the `thop` package to count the operations in an attention operation.

    Args:
        model (torch.nn.Module): The PyTorch model to count the operations for.
        _x (Tuple[torch.Tensor]): The input tensors to the model (not used in this function).
        y (Tuple[torch.Tensor]): The output tensors from the model.

    Returns:
        None
    """
    # Get the batch size, number of channels, and spatial dimensions of the output tensor
    b, c, *spatial = y[0].shape

    # Compute the total number of spatial dimensions
    num_spatial = int(np.prod(spatial))

    # We perform two matrix multiplications with the same number of operations.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c

    # Add the number of operations to the model's total_ops attribute
    model.total_ops += torch.DoubleTensor([matmul_ops])


def convert_module_to_f16(x: nn.Module) -> nn.Module:
    """
    Convert a PyTorch module to use 16-bit floating point precision.

    Args:
        x (nn.Module): The PyTorch module to convert.

    Returns:
        nn.Module: The converted PyTorch module.
    """
    pass


def convert_module_to_f32(x: nn.Module) -> nn.Module:
    """
    Convert a PyTorch module to use 32-bit floating point precision.

    Args:
        x (nn.Module): The PyTorch module to convert.

    Returns:
        nn.Module: The converted PyTorch module.
    """
    pass


def avg_pool_nd(dims: int, *args: Union[int, tuple], **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D average pooling module.

    Args:
        dims (int): The number of dimensions of the pooling module (1, 2, or 3).
        *args (Union[int, tuple]): The positional arguments to pass to the pooling module.
        **kwargs: Additional keyword arguments to pass to the pooling module.

    Returns:
        nn.Module: A 1D, 2D, or 3D average pooling module.

    Raises:
        ValueError: If the number of dimensions is not 1, 2, or 3.

    Example:
        >>> pool = avg_pool_nd(2, kernel_size=3, stride=2)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> y = pool(x)
    """
    if dims == 1:
        # Create a 1D average pooling module
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        # Create a 2D average pooling module
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        # Create a 3D average pooling module
        return nn.AvgPool3d(*args, **kwargs)
    else:
        # Raise an error if the number of dimensions is not 1, 2, or 3
        raise ValueError(f"Unsupported number of dimensions: {dims}")


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
        >>> output_tensor.shape
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


class TimestepBlock(nn.Module):
    """
    Abstract base class for any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the module to `x` given `emb` timestep embeddings.

        Args:
            x (torch.Tensor): The input tensor to the module.
            emb (torch.Tensor): The timestep embeddings to apply to the
                input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output
                tensor and the updated timestep embeddings.
        """


class CrossAttention(nn.Module):
    """
    Cross-attention module that computes attention weights between a query
    tensor and a context tensor.

    Args:
        query_dim (int): The dimension of the query tensor.
        context_dim (int, optional): The dimension of the context tensor. If
            None, defaults to `query_dim`.
        heads (int, optional): The number of attention heads to use. Defaults
            to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults
            to 64.
        dropout (float, optional): The dropout probability to use. Defaults to 0.

    Inputs:
        - x (torch.Tensor): The query tensor of shape
            `(batch_size, query_seq_len, query_dim)`.
        - context (torch.Tensor, optional): The context tensor of shape
            `(batch_size, context_seq_len, context_dim)`. If None, defaults to `x`.
        - mask (torch.Tensor, optional): A boolean mask of shape
            `(batch_size, query_seq_len)` indicating which query elements should
            be masked out of the attention computation.

    Outputs:
        - torch.Tensor: The output tensor of shape `(batch_size, query_seq_len, query_dim)`.

    Example:
        >>> query = torch.randn(2, 10, 64)
        >>> context = torch.randn(2, 20, 64)
        >>> attn = CrossAttention(query_dim=64, context_dim=64, heads=8, dim_head=64, dropout=0.1)
        >>> out = attn(x=query, context=context)
        >>> out.shape
        torch.Size([2, 10, 64])
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        # default to self-attention
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape queries, keys, and values for multi-head attention
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # Aggregated attention weights
        sim = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class GEGLU(nn.Module):
    """
    Gated Exponential Linear Unit (GEGLU) activation function.

    Applies a linear projection to the input tensor, splits the result
    into two halves, and applies the GELU function to one half while
    leaving the other half unchanged. The output is the element-wise
    product of the two halves.

    Args:
        dim_in (int): The number of input features.
        dim_out (int): The number of output features.

    Example:
        >>> x = torch.randn(32, 64)
        >>> gelu = GEGLU(64, 128)
        >>> y = gelu(x)
        >>> y.shape
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """
        Apply the GEGLU activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Apply linear projection and split into two halves
        x, gate = self.proj(x).chunk(2, dim=-1)

        # Apply GELU to one half and leave the other half unchanged
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Feedforward neural network with an optional Gated Exponential Linear Unit (GEGLU) activation function.

    Applies a linear projection to the input tensor, applies the GELU or GEGLU activation function, and applies a
    linear projection to the output tensor.

    Args:
        dim (int): The number of input features.
        dim_out (int, optional): The number of output features. If not provided, defaults to `dim`.
        mult (float, optional): The multiplier for the inner dimension of the linear projections. Defaults to 4.
        glu (bool, optional): Whether to use the GEGLU activation function instead of GELU. Defaults to False.
        dropout (float, optional): The dropout probability. Defaults to 0.

    Example:
        >>> x = torch.randn(32, 64)
        >>> ff = FeedForward(64, 128, mult=2, glu=True, dropout=0.1)
        >>> y = ff(x)
        >>> y.shape
    """

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        # Define input projection
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        # Define network layers
        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feedforward neural network to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.net(x)


class CheckpointFunction(torch.autograd.Function):
    """
    A PyTorch autograd function that enables gradient checkpointing.

    Gradient checkpointing is a technique for reducing memory usage during backpropagation by recomputing intermediate
    activations on-the-fly instead of storing them in memory. This function implements the forward and backward passes
    for gradient checkpointing.

    Args:
        run_function (callable): The function to evaluate.
        length (int): The number of input tensors to `run_function`.
        *args: The input tensors and parameters to `run_function`.

    Returns:
        Any: The output of `run_function`.

    Example:
        >>> def my_func(x, y, z):
        ...     return x * y + z
        >>> x = torch.randn(32, 64)
        >>> y = torch.randn(32, 64)
        >>> z = torch.randn(32, 64)
        >>> output = CheckpointFunction.apply(my_func, 2, x, y, z)
        >>> output.shape
        torch.Size([32, 64])
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        Compute the forward pass of the gradient checkpointing function.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): The context object for the autograd function.
            run_function (callable): The function to evaluate.
            length (int): The number of input tensors to `run_function`.
            *args: The input tensors and parameters to `run_function`.

        Returns:
            Any: The output of `run_function`.
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """
        Compute the backward pass of the gradient checkpointing function.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): The context object for
                the autograd function.
            *output_grads: The gradients of the output tensors.

        Returns:
            tuple: The gradients of the input tensors and parameters.
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class BasicTransformerBlock(nn.Module):
    """
    A basic transformer block consisting of a self-attention layer, a feedforward layer, and another self-attention layer.

    Args:
        dim (int): The input and output dimension of the block.
        n_heads (int): The number of attention heads to use.
        d_head (int): The dimension of each attention head.
        dropout (float, optional): The dropout probability to use. Default: 0.0.
        context_dim (int, optional): The dimension of the context tensor for the second
            self-attention layer. If None, the second layer is a self-attention layer. Default: None.
        gated_ff (bool, optional): Whether to use a gated feedforward layer. Default: True.
        checkpoint (bool, optional): Whether to use gradient checkpointing to reduce memory
            usage. Default: True.

    Inputs:
        x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, dim)`.
        context (torch.Tensor, optional): The context tensor of shape
        `(batch_size, seq_len, context_dim)`. If None, the second self-attention
        layer is a self-attention layer. Default: None.

    Outputs:
        torch.Tensor: The output tensor of shape `(batch_size, seq_len, dim)`.

    Example:
        >>> block = BasicTransformerBlock(
            dim=512, n_heads=8, d_head=64,
            dropout=0.1, context_dim=256, gated_ff=True, checkpoint=True
        )
        >>> x = torch.randn(32, 128, 512)
        >>> context = torch.randn(32, 128, 256)
        >>> output = block(x, context)
        >>> output.shape
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """
        Initialize the basic transformer block.

        Args:
            dim (int): The input and output dimension of the block.
            n_heads (int): The number of attention heads to use.
            d_head (int): The dimension of each attention head.
            dropout (float, optional): The dropout probability to use. Default: 0.0.
            context_dim (int, optional): The dimension of the context tensor for the second self-attention layer. If None, the second layer is a self-attention layer. Default: None.
            gated_ff (bool, optional): Whether to use a gated feedforward layer. Default: True.
            checkpoint (bool, optional): Whether to use gradient checkpointing to reduce memory usage. Default: True.
        """
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the forward pass of the basic transformer block.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, dim)`.
            context (torch.Tensor, optional): The context tensor of shape
            `(batch_size, seq_len, context_dim)`. If None, the second self-attention
            layer is a self-attention layer. Default: None.

        Returns:
            torch.Tensor: The output tensor of shape `(batch_size, seq_len, dim)`.
        """
        return checkpoint(
            self._forward, (x, context), self.parameters(), False  # self.checkpoint
        )

    def _forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the forward pass of the basic transformer block.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, seq_len, dim)`.
            context (torch.Tensor, optional): The context tensor of shape
            `(batch_size, seq_len, context_dim)`. If None, the second self-attention
            layer is a self-attention layer. Default: None.

        Returns:
            torch.Tensor: The output tensor of shape `(batch_size, seq_len, dim)`.
        """
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class GroupNorm32(nn.GroupNorm):
    """
    A subclass of `nn.GroupNorm` that casts the input tensor to float32 before
    passing it to the parent class's `forward` method, and then casts the output
    back to the original data type of the input tensor.

    Args:
        num_groups (int): The number of groups to divide the channels into.
        num_channels (int): The number of channels in the input tensor.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool, optional): Whether to apply learnable affine transformations to
        the output. Default: True.

    Inputs:
        x (torch.Tensor): The input tensor of shape `(batch_size, num_channels, *)`.

    Outputs:
        torch.Tensor: The output tensor of the same shape as the input tensor.

    Example:
        >>> norm = GroupNorm32(num_groups=32, num_channels=64, eps=1e-5, affine=True)
        >>> x = torch.randn(32, 64, 128, 128)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([32, 64, 128, 128])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the group normalization layer.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size, num_channels, *)`.

        Returns:
            torch.Tensor: The output tensor of the same shape as the input tensor.
        """
        return super().forward(x.float()).type(x.dtype)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.

    Args:
        n_heads (int): The number of attention heads.

    Inputs:
        qkv (torch.Tensor): An [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.

    Outputs:
        torch.Tensor: An [N x (H * C) x T] tensor after attention.

    Example:
        >>> attn = QKVAttention(n_heads=8)
        >>> x = torch.randn(32, 24 * 8, 128)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([32, 192, 128])
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        Args:
            qkv (torch.Tensor): An [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.

        Returns:
            torch.Tensor: An [N x (H * C) x T] tensor after attention.
        """
        # Get the batch size, width, and length of the input tensor
        bs, width, length = qkv.shape

        # Ensure that the width is divisible by 3 * n_heads
        assert width % (3 * self.n_heads) == 0

        # Compute the number of channels per head
        ch = width // (3 * self.n_heads)

        # Split the input tensor into Q, K, and V tensors
        q, k, v = qkv.chunk(3, dim=1)

        # Compute the scaling factor for the dot product
        scale = 1 / math.sqrt(math.sqrt(ch))

        # Compute the dot product of Q and K, and apply the scaling factor
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )

        # Apply softmax to the dot product to get the attention weights
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the weighted sum of V using the attention weights
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )

        # Reshape the output tensor to the original shape
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output
    heads shaping.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Apply QKV attention.

        Args:
            qkv (torch.Tensor): An [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.

        Returns:
            torch.Tensor: An [N x (H * C) x T] tensor after attention.

        Example:
            >>> attn = QKVAttentionLegacy(n_heads=8)
            >>> x = torch.randn(32, 24 * 8, 128)
            >>> output = attn(x)
            >>> output.shape
            torch.Size([32, 192, 128])
        """
        # Get the batch size, width, and length of the input tensor
        bs, width, length = qkv.shape

        # Ensure that the width is divisible by 3 * n_heads
        assert width % (3 * self.n_heads) == 0

        # Compute the number of channels per head
        ch = width // (3 * self.n_heads)

        # Split the input tensor into Q, K, and V tensors
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        # Compute the scaling factor for the dot product
        scale = 1 / math.sqrt(math.sqrt(ch))

        # Compute the dot product of Q and K, and apply the scaling factor
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        # Apply softmax to the dot product to get the attention weights
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # Compute the weighted sum of V using the attention weights
        a = torch.einsum("bts,bcs->bct", weight, v)

        # Reshape the output tensor to the original shape
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(
        model: nn.Module, _x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]
    ) -> None:
        """
        A counter for the `thop` package to count the operations in an attention operation.

        Args:
            model (nn.Module): The PyTorch model to count the operations for.
            _x (Tuple[torch.Tensor]): The input tensors to the model (not used in this function).
            y (Tuple[torch.Tensor]): The output tensors from the model.

        Returns:
            None

        Example:
            >>> macs, params = thop.profile(
            ...     model,
            ...     inputs=(inputs, timestamps),
            ...     custom_ops={QKVAttention: QKVAttention.count_flops},
            ... )
        """
        count_flops_attn(model, _x, y)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Args:
        channels (int): The number of input and output channels.
        use_conv (bool): Whether to apply a convolution after upsampling.
        dims (int, optional): The number of dimensions of the input tensor (1, 2, or 3).
            Defaults to 2.
        out_channels (int, optional): The number of output channels.
            Defaults to None (same as input channels).
        padding (int, optional): The amount of padding to apply to the convolution.
            Defaults to 1.

    Example:
        >>> upsample = Upsample(channels=64, use_conv=True, dims=2, out_channels=128, padding=1)
        >>> x = torch.randn(32, 64, 128, 128)
        >>> output = upsample(x)
        >>> output.shape
        torch.Size([32, 128, 128, 128])
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            # Create a convolutional layer with the specified number of dimensions
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        """
        Apply upsampling to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to upsample.

        Returns:
            torch.Tensor: The upsampled tensor.
        """
        # Ensure that the input tensor has the correct number of channels
        assert x.shape[1] == self.channels
        # Upsample the input tensor using nearest-neighbor interpolation
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply the convolutional layer if necessary
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Args:
        channels (int): The number of input and output channels.
        use_conv (bool): Whether to apply a convolution after downsampling.
        dims (int, optional): The number of dimensions of the input tensor (1, 2, or 3). Defaults to 2.
        out_channels (int, optional): The number of output channels. Defaults to None
            (same as input channels).
        padding (int, optional): The amount of padding to apply to the convolution. Defaults to 1.

    Raises:
        AssertionError: If the input tensor does not have the correct number of channels.

    Example:
        >>> downsample = Downsample(64, use_conv=True, dims=2, out_channels=128, padding=1)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> y = downsample(x)
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            # Create a convolutional layer with the specified number of dimensions
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            # Create an average pooling layer with the specified number of dimensions
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """
        Apply downsampling to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to downsample.

        Returns:
            torch.Tensor: The downsampled tensor.

        Raises:
            AssertionError: If the input tensor does not have the correct number of channels.
        """

        # Ensure that the input tensor has the correct number of channels
        assert x.shape[1] == self.channels

        # Apply the convolutional or pooling layer
        return self.op(x)
