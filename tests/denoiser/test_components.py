"""Numerical and gradient contracts for the configurable denoiser blocks."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from opensr_model.autoencoder import utils as ae
from opensr_model.denoiser import utils as du
from opensr_model.denoiser.unet import ResBlock, SpatialTransformer, UNetModel


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("use_conv", [False, True])
def test_resampling_preserves_batch_channels_and_depth(dims, use_conv):
    x = torch.randn(2, 4, *([4] * dims), requires_grad=True)
    up = du.Upsample(4, use_conv, dims=dims)
    down = du.Downsample(4, use_conv, dims=dims)
    enlarged = up(x)
    spatial = (4, 8, 8) if dims == 3 else (8,) * dims
    assert enlarged.shape == (2, 4, *spatial)
    restored = down(enlarged)
    assert restored.shape == x.shape
    if not use_conv:
        torch.testing.assert_close(restored, x)
    restored.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(AssertionError):
        up(torch.zeros(1, 3, *([4] * dims)))


@pytest.mark.parametrize("factory", [du.conv_nd, du.avg_pool_nd])
def test_invalid_spatial_dimension_is_rejected(factory):
    with pytest.raises(ValueError, match="dimensions"):
        factory(4, 1)


def test_timestep_embeddings_handle_odd_dimensions_and_repetition():
    t = torch.tensor([0.0, 0.5, 3.0])
    encoded = du.timestep_embedding(t, 5)
    assert encoded.shape == (3, 5)
    torch.testing.assert_close(encoded[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(
        encoded[:, :2].square() + encoded[:, 2:4].square(), torch.ones(3, 2)
    )
    torch.testing.assert_close(
        du.timestep_embedding(t, 3, repeat_only=True), t[:, None].expand(-1, 3)
    )
    assert du.default(0, lambda: 9) == 0
    assert du.default(None, lambda: 9) == 9
    assert du.default(None, 7) == 7


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_recomputation_preserves_input_and_parameter_gradients(enabled):
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    weight = nn.Parameter(torch.tensor([3.0, 4.0]))
    unused = nn.Parameter(torch.ones(2))
    calls = []

    def operation(value):
        calls.append(torch.is_grad_enabled())
        return value.square() * weight

    result = du.checkpoint(operation, (x,), (weight, unused), enabled)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([6.0, -16.0]))
    torch.testing.assert_close(weight.grad, torch.tensor([1.0, 4.0]))
    assert unused.grad is None
    assert calls == ([False, True] if enabled else [True])


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("masked", [False, True])
def test_cross_attention_matches_pytorch_attention_and_ignores_masked_tokens(
    heads, masked
):
    torch.manual_seed(12)
    layer = du.CrossAttention(8, context_dim=6, heads=heads, dim_head=4)
    query = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 5, 6, requires_grad=True)
    mask = (
        torch.tensor([[True, True, False, True, False]]).expand(2, -1)
        if masked
        else None
    )
    result = layer(query, context, mask)
    q, k, v = [
        projection(tensor).reshape(2, -1, heads, 4).transpose(1, 2)
        for projection, tensor in (
            (layer.to_q, query),
            (layer.to_k, context),
            (layer.to_v, context),
        )
    ]
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None if mask is None else mask[:, None, None, :]
    )
    expected = layer.to_out(expected.transpose(1, 2).reshape(2, 3, heads * 4))
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    assert torch.isfinite(query.grad).all()
    if masked:
        assert torch.count_nonzero(context.grad[:, [2, 4]]) == 0
        changed = context.detach().clone()
        changed[:, [2, 4]] = 1e4
        torch.testing.assert_close(layer(query, changed, mask), result)


@pytest.mark.parametrize("gated", [False, True])
def test_transformer_self_attention_backpropagates(gated):
    layer = du.BasicTransformerBlock(8, 2, 4, gated_ff=gated)
    x = torch.randn(2, 5, 8, requires_grad=True)
    output = layer(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()
    )


@pytest.mark.parametrize("attention", [du.QKVAttention, du.QKVAttentionLegacy])
@pytest.mark.parametrize("heads", [1, 2])
def test_qkv_attention_matches_scaled_dot_product(attention, heads):
    q, k, v = (torch.randn(2, heads, 4, 6) for _ in range(3))
    packed = (
        torch.cat((q, k, v), dim=2).reshape(2, -1, 6)
        if attention is du.QKVAttentionLegacy
        else torch.cat([item.reshape(2, -1, 6) for item in (q, k, v)], dim=1)
    )
    expected = F.scaled_dot_product_attention(
        q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
    )
    result = attention(heads)(packed)
    torch.testing.assert_close(result, expected.transpose(-1, -2).reshape(2, -1, 6))
    counter = nn.Module()
    counter.total_ops = torch.zeros(1, dtype=torch.double)
    attention.count_flops(counter, (packed,), (result,))
    assert counter.total_ops.item() == 2 * 2 * 6**2 * (heads * 4)


@pytest.mark.parametrize("module", [ae, du])
@pytest.mark.parametrize("kind", ["AttnBlock", "LinAttnBlock"])
def test_image_attention_is_spatially_permutation_equivariant(module, kind):
    layer = getattr(module, kind)(32).eval()
    x = torch.randn(2, 32, 3, 4, requires_grad=True)
    result = layer(x)
    torch.testing.assert_close(layer(x.flip(-1)), result.flip(-1))
    result.mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("direction", ["same", "up", "down"])
@pytest.mark.parametrize("scale_shift", [False, True])
def test_resblock_zero_initialized_residual_preserves_skip(direction, scale_shift):
    layer = ResBlock(
        32,
        16,
        0,
        up=direction == "up",
        down=direction == "down",
        use_scale_shift_norm=scale_shift,
        use_checkpoint=True,
    )
    x = torch.randn(2, 32, 4, 4, requires_grad=True)
    output = layer(x, torch.randn(2, 16))
    expected = (
        F.interpolate(x, scale_factor=2)
        if direction == "up"
        else F.avg_pool2d(x, 2) if direction == "down" else x
    )
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


@pytest.mark.parametrize("conv_shortcut", [False, True])
def test_channel_changing_residual_blocks_support_conditioning(conv_shortcut):
    for layer, embedding in (
        (
            ae.ResnetBlock(
                in_channels=32,
                out_channels=64,
                dropout=0,
                conv_shortcut=conv_shortcut,
                temb_channels=8,
            ),
            torch.randn(2, 8),
        ),
        (
            ResBlock(32, 8, 0, out_channels=64, use_conv=conv_shortcut),
            torch.randn(2, 8),
        ),
    ):
        x = torch.randn(2, 32, 4, 4, requires_grad=True)
        output = layer(x, embedding)
        assert output.shape == (2, 64, 4, 4)
        output.square().mean().backward()
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "variant", ["spatial", "resample", "classes", "codebook", "heads"]
)
def test_unet_architecture_variants_forward_and_backward(variant):
    options = dict(
        in_channels=4,
        model_channels=32,
        out_channels=2,
        num_res_blocks=1,
        attention_resolutions=[1, 2],
        channel_mult=(1, 2),
        num_head_channels=16,
    )
    if variant == "spatial":
        options.update(use_spatial_transformer=True, context_dim=6, legacy=False)
    elif variant == "resample":
        options.update(
            resblock_updown=True,
            use_scale_shift_norm=True,
            use_new_attention_order=True,
        )
    elif variant == "classes":
        options["num_classes"] = 3
    elif variant == "codebook":
        options["n_embed"] = 7
    else:
        options.update(
            num_head_channels=-1, num_heads=2, legacy=False, conv_resample=False
        )
    model = UNetModel(**options)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    output = model(
        x,
        torch.tensor([0, 2]),
        context=torch.randn(2, 3, 6) if variant == "spatial" else None,
        y=torch.tensor([0, 2]) if variant == "classes" else None,
    )
    assert output.shape == (2, 7 if variant == "codebook" else 2, 8, 8)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    if variant == "classes":
        with pytest.raises(AssertionError, match="class-conditional"):
            model(x, torch.zeros(2))


def test_autoencoder_attention_factory_and_resampling():
    x = torch.randn(2, 32, 4, 4)
    for kind in ("none", "linear", "vanilla"):
        output = ae.make_attn(32, kind)(x)
        assert output.shape == x.shape and torch.isfinite(output).all()
    with pytest.raises(AssertionError, match="unknown"):
        ae.make_attn(32, "invalid")
    torch.testing.assert_close(ae.Downsample(32, False)(ae.Upsample(32, False)(x)), x)


def test_chunked_autoencoder_attention_matches_dense_gradients():
    dense = ae.AttnBlock(32)
    chunked = deepcopy(dense)
    chunked.query_chunk_size = 3
    x = torch.randn(2, 32, 3, 4, requires_grad=True)
    copy = x.detach().clone().requires_grad_(True)
    first, second = dense(x), chunked(copy)
    torch.testing.assert_close(first, second)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(x.grad, copy.grad)
