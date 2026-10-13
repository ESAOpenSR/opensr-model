"""Checkpoint compatibility helpers for OpenSR training.

The public inference API expects a *native* :class:`LatentDiffusion` state dict
under the top-level ``"state_dict"`` key.  Lightning wrappers and distributed
training commonly add one or more prefixes to those keys, while the released
OpenSR checkpoint contains the autoencoder below ``"first_stage_model."``.
This module keeps those serialization concerns outside the model
implementations.

All loads use PyTorch's restricted ``weights_only`` unpickler.  State-dict
prefixes are selected by comparing exact target keys and tensor shapes; keys
are never matched by order.  Saves use a temporary file in the destination
directory followed by :func:`os.replace`, so readers see either the previous
checkpoint or the complete new checkpoint.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import torch
from torch import Tensor, nn

CheckpointSource: TypeAlias = str | os.PathLike[str] | Mapping[str, Any]
TensorStateDict: TypeAlias = OrderedDict[str, Tensor]

_STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "autoencoder_state_dict",
    "weights",
    "model",
    "module",
)

# Prefixes introduced by common LightningModule layouts and distributed
# wrappers.  Additional prefixes are inferred from exact source/target suffix
# matches at runtime.
_KNOWN_PREFIXES = (
    "",
    "module.",
    "_forward_module.",
    "model.",
    "network.",
    "autoencoder.",
    "ae.",
    "first_stage_model.",
    "module.model.",
    "module.network.",
    "module.autoencoder.",
    "module.first_stage_model.",
    "model.module.",
    "model.autoencoder.",
    "model.first_stage_model.",
    "_forward_module.model.",
    "_forward_module.autoencoder.",
    "_forward_module.first_stage_model.",
    "module.model.autoencoder.",
    "module.model.first_stage_model.",
)

_AUTOENCODER_ROOTS = (
    "encoder.",
    "decoder.",
    "quant_conv.",
    "post_quant_conv.",
)

_DIFFUSION_BUFFER_KEY_ORDER = (
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
)
_DIFFUSION_BUFFER_KEYS = frozenset(_DIFFUSION_BUFFER_KEY_ORDER)
_DENOISER_PREFIX = "model."
_EMA_PREFIX = "model_ema."
_EMA_BOOKKEEPING_KEYS = frozenset({f"{_EMA_PREFIX}decay", f"{_EMA_PREFIX}num_updates"})


class CheckpointError(RuntimeError):
    """Base class for OpenSR checkpoint errors."""


class CheckpointFormatError(CheckpointError):
    """Raised when a payload does not contain a safe tensor state dict."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when checkpoint tensors cannot be mapped to a target model."""


@dataclass(frozen=True)
class ShapeMismatch:
    """A source tensor whose mapped key exists but has the wrong shape."""

    source_key: str
    target_key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class PrefixScore:
    """Compatibility score for one source-prefix/target-prefix transform."""

    source_prefix: str
    target_prefix: str
    matched_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    matched_numel: int

    @property
    def is_exact(self) -> bool:
        """Whether the transform is a strict key-and-shape match."""

        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


@dataclass(frozen=True)
class LoadReport:
    """Summary returned after loading weights into a model."""

    source: str
    source_prefix: str
    target_prefix: str
    matched_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    filtered_legacy_loss_keys: tuple[str, ...]

    @property
    def matched_count(self) -> int:
        """Number of target tensors loaded."""

        return len(self.matched_keys)

    @property
    def is_exact(self) -> bool:
        """Whether every target tensor was loaded with no extra mapped keys."""

        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


@dataclass(frozen=True)
class StateDictSpec:
    """Immutable expected state-dict keys and tensor shapes.

    Dtypes are deliberately excluded: PyTorch's strict loader supports loading
    lower-precision checkpoint tensors into a full-precision model. Keys and
    shapes, however, must match exactly for an inference-compatible export.
    """

    entries: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> "StateDictSpec":
        state = _copy_state_dict(state_dict, context="expected state dict")
        return cls(tuple((key, tuple(tensor.shape)) for key, tensor in state.items()))

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the expected shape for every key."""

        return dict(self.entries)

    def validate(self, state_dict: Mapping[str, Any], *, context: str) -> None:
        """Raise when a tensor mapping differs in any key or shape."""

        state = _copy_state_dict(state_dict, context=context)
        expected = self.shapes
        actual_keys = set(state)
        expected_keys = set(expected)
        missing = tuple(sorted(expected_keys - actual_keys))
        unexpected = tuple(sorted(actual_keys - expected_keys))
        mismatches = tuple(
            ShapeMismatch(
                source_key=key,
                target_key=key,
                source_shape=tuple(state[key].shape),
                target_shape=expected[key],
            )
            for key in sorted(actual_keys & expected_keys)
            if tuple(state[key].shape) != expected[key]
        )
        if missing or unexpected or mismatches:
            score = PrefixScore(
                source_prefix="",
                target_prefix="",
                matched_keys=tuple(
                    key
                    for key in sorted(actual_keys & expected_keys)
                    if tuple(state[key].shape) == expected[key]
                ),
                missing_keys=missing,
                unexpected_keys=unexpected,
                shape_mismatches=mismatches,
                matched_numel=0,
            )
            raise CheckpointCompatibilityError(
                f"{context} does not match the expected architecture. "
                + _compatibility_message(score)
            )


def _source_name(source: CheckpointSource) -> str:
    if isinstance(source, Mapping):
        return "<in-memory checkpoint>"
    return os.fspath(source)


def _looks_like_state_dict(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(item, (Tensor, nn.Parameter)) for item in value.values())
    )


def _copy_state_dict(
    state_dict: Mapping[str, Any], *, context: str = "state_dict"
) -> TensorStateDict:
    """Make a structural copy without cloning potentially gigabytes of storage."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointFormatError(f"{context} must be a non-empty mapping")

    copied: TensorStateDict = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise CheckpointFormatError(f"{context} contains a non-string key: {key!r}")
        if not isinstance(value, (Tensor, nn.Parameter)):
            raise CheckpointFormatError(
                f"{context}[{key!r}] is {type(value).__name__}, not a tensor"
            )
        copied[key] = value.detach()

    # torch.nn.Module.load_state_dict uses this private metadata for module
    # version migrations.  Preserve it when the input is a real state_dict.
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        copied._metadata = dict(metadata)  # type: ignore[attr-defined]
    return copied


def _restricted_torch_load(path: Path) -> Any:
    """Load a path without allowing arbitrary pickle globals."""

    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        # mmap avoids a second ~1.1 GB resident copy for the released model.
        return torch.load(path, mmap=True, **kwargs)
    except RuntimeError as exc:
        # Checkpoints written with the legacy tar serializer cannot be mmap'ed,
        # but can still be loaded with the restricted unpickler.
        if "mmap" not in str(exc).lower():
            raise CheckpointFormatError(
                f"Could not safely load checkpoint {path}: {exc}"
            ) from exc
    except Exception as exc:
        raise CheckpointFormatError(
            f"Could not safely load checkpoint {path}: {exc}"
        ) from exc

    try:
        return torch.load(path, **kwargs)
    except Exception as exc:
        raise CheckpointFormatError(
            f"Could not safely load legacy checkpoint {path}: {exc}"
        ) from exc


def load_checkpoint_payload(path: CheckpointSource) -> dict[str, Any]:
    """Safely load and normalize a native, legacy, or Lightning checkpoint.

    Parameters
    ----------
    path:
        Filesystem path or an already-loaded mapping.  File loads always use
        ``torch.load(..., weights_only=True, map_location="cpu")``.

    Returns
    -------
    dict
        A shallow payload copy with a validated tensor mapping at
        ``payload["state_dict"]``.  Raw state dicts and legacy containers such
        as ``"model_state_dict"`` are normalized to that layout.  Tensor
        storage is intentionally shared; the input mapping itself is never
        modified.

    Raises
    ------
    CheckpointFormatError
        If restricted loading fails or no tensor state dict can be found.
    """

    if isinstance(path, Mapping):
        loaded: Any = path
    else:
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise CheckpointFormatError(f"Checkpoint is not a file: {checkpoint_path}")
        loaded = _restricted_torch_load(checkpoint_path)

    if _looks_like_state_dict(loaded):
        return {"state_dict": _copy_state_dict(loaded, context="raw state_dict")}

    if not isinstance(loaded, Mapping):
        raise CheckpointFormatError(
            "Checkpoint payload must be a mapping or a raw tensor state dict; "
            f"got {type(loaded).__name__}"
        )

    payload = dict(loaded)
    for key in _STATE_DICT_KEYS:
        candidate = loaded.get(key)
        if candidate is None:
            continue
        if not _looks_like_state_dict(candidate):
            if key == "state_dict":
                raise CheckpointFormatError(
                    "Checkpoint 'state_dict' is not a non-empty tensor mapping"
                )
            continue
        payload["state_dict"] = _copy_state_dict(
            candidate, context=f"checkpoint[{key!r}]"
        )
        return payload

    raise CheckpointFormatError(
        "Checkpoint does not contain a tensor state dict under any supported "
        f"key: {', '.join(_STATE_DICT_KEYS)}"
    )


def _is_legacy_loss_key(key: str) -> bool:
    # The original Lightning modules attached LPIPS/discriminator parameters
    # below a module named exactly ``loss``.  Segment matching avoids dropping
    # unrelated names merely containing the substring, such as ``lossless``.
    return "loss" in key.split(".")


def filter_legacy_loss_keys(
    state_dict: Mapping[str, Any],
) -> tuple[TensorStateDict, tuple[str, ...]]:
    """Remove only legacy ``*.loss.*`` module tensors from a state dict."""

    state = _copy_state_dict(state_dict)
    filtered = tuple(sorted(key for key in state if _is_legacy_loss_key(key)))
    if not filtered:
        return state, ()

    filtered_set = set(filtered)
    kept: TensorStateDict = OrderedDict(
        (key, value) for key, value in state.items() if key not in filtered_set
    )
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        kept._metadata = dict(metadata)  # type: ignore[attr-defined]
    return kept, filtered


def _prefix_part_count(prefix: str) -> int:
    return len([part for part in prefix.split(".") if part])


def _prefix_candidates(
    source_keys: Sequence[str], target_keys: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    source_set = set(source_keys)
    target_set = set(target_keys)

    source_prefixes = {
        prefix
        for prefix in _KNOWN_PREFIXES
        if not prefix or any(key.startswith(prefix) for key in source_keys)
    }
    target_prefixes = {
        prefix
        for prefix in _KNOWN_PREFIXES
        if not prefix or any(key.startswith(prefix) for key in target_keys)
    }

    # Infer arbitrary outer wrapper names when stripping one makes at least one
    # complete target key, or adding one makes at least one complete source key.
    for key in source_keys:
        parts = key.split(".")
        for index in range(1, min(5, len(parts))):
            remainder = ".".join(parts[index:])
            if remainder in target_set:
                source_prefixes.add(".".join(parts[:index]) + ".")

    for key in target_keys:
        parts = key.split(".")
        for index in range(1, min(5, len(parts))):
            remainder = ".".join(parts[index:])
            if remainder in source_set:
                target_prefixes.add(".".join(parts[:index]) + ".")

    candidates = {
        (source_prefix, target_prefix)
        for source_prefix in source_prefixes
        for target_prefix in target_prefixes
    }
    candidates.add(("", ""))
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                _prefix_part_count(item[0]) + _prefix_part_count(item[1]),
                item,
            ),
        )
    )


def _evaluate_prefix(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    source_prefix: str,
    target_prefix: str,
) -> PrefixScore:
    matched: list[str] = []
    unexpected: list[str] = []
    mismatches: list[ShapeMismatch] = []
    matched_numel = 0

    for source_key, source_tensor in source.items():
        if source_prefix and not source_key.startswith(source_prefix):
            continue
        remainder = source_key[len(source_prefix) :]
        target_key = f"{target_prefix}{remainder}"
        target_tensor = target.get(target_key)
        if target_tensor is None:
            unexpected.append(target_key)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatches.append(
                ShapeMismatch(
                    source_key=source_key,
                    target_key=target_key,
                    source_shape=tuple(source_tensor.shape),
                    target_shape=tuple(target_tensor.shape),
                )
            )
            continue
        matched.append(target_key)
        matched_numel += source_tensor.numel()

    matched_set = set(matched)
    missing = sorted(set(target) - matched_set)
    return PrefixScore(
        source_prefix=source_prefix,
        target_prefix=target_prefix,
        matched_keys=tuple(sorted(matched)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(set(unexpected))),
        shape_mismatches=tuple(sorted(mismatches, key=lambda item: item.target_key)),
        matched_numel=matched_numel,
    )


def _score_rank(score: PrefixScore) -> tuple[int, ...]:
    return (
        int(score.is_exact),
        int(not score.missing_keys),
        len(score.matched_keys),
        score.matched_numel,
        -len(score.shape_mismatches),
        -len(score.unexpected_keys),
        -(
            _prefix_part_count(score.source_prefix)
            + _prefix_part_count(score.target_prefix)
        ),
        int(score.source_prefix == "" and score.target_prefix == ""),
    )


def score_state_dict_prefixes(
    source_state_dict: Mapping[str, Any], target_state_dict: Mapping[str, Any]
) -> tuple[PrefixScore, ...]:
    """Score plausible prefix transforms by exact target keys and shapes.

    The best score is returned first.  Dtype differences are deliberately not
    considered incompatible because loading fp16/bf16 weights into fp32 model
    parameters is a supported PyTorch operation.
    """

    source = _copy_state_dict(source_state_dict, context="source state_dict")
    target = _copy_state_dict(target_state_dict, context="target state_dict")
    scores = [
        _evaluate_prefix(source, target, source_prefix, target_prefix)
        for source_prefix, target_prefix in _prefix_candidates(
            tuple(source), tuple(target)
        )
    ]
    return tuple(sorted(scores, key=_score_rank, reverse=True))


def _compatibility_message(score: PrefixScore) -> str:
    def preview(values: Sequence[Any]) -> str:
        rendered = ", ".join(str(value) for value in values[:8])
        if len(values) > 8:
            rendered += f", ... (+{len(values) - 8})"
        return rendered or "none"

    return (
        "Best checkpoint prefix mapping is not strictly compatible: "
        f"strip={score.source_prefix!r}, add={score.target_prefix!r}, "
        f"matched={len(score.matched_keys)}, "
        f"missing=[{preview(score.missing_keys)}], "
        f"unexpected=[{preview(score.unexpected_keys)}], "
        f"shape_mismatches=[{preview(score.shape_mismatches)}]"
    )


def _mapping_signature(
    source: Mapping[str, Tensor], score: PrefixScore
) -> tuple[tuple[str, str], ...]:
    """Describe which source tensor supplies each matched target tensor."""

    matched = set(score.matched_keys)
    signature: list[tuple[str, str]] = []
    for source_key in source:
        if score.source_prefix and not source_key.startswith(score.source_prefix):
            continue
        remainder = source_key[len(score.source_prefix) :]
        target_key = f"{score.target_prefix}{remainder}"
        if target_key in matched:
            signature.append((source_key, target_key))
    return tuple(sorted(signature))


def _raise_if_ambiguous(
    source: Mapping[str, Tensor], scores: Sequence[PrefixScore]
) -> None:
    """Reject equally ranked transforms that select different source tensors."""

    if len(scores) < 2:
        return
    # Prefix depth and identity are deterministic presentation tie-breakers,
    # not compatibility evidence. Do not let them silently choose between two
    # equally complete copies such as ``teacher.*`` and ``module.student.*``.
    best_rank = _score_rank(scores[0])[:-2]
    contenders = tuple(
        score for score in scores if _score_rank(score)[:-2] == best_rank
    )
    signatures: dict[tuple[tuple[str, str], ...], list[PrefixScore]] = {}
    for score in contenders:
        signatures.setdefault(_mapping_signature(source, score), []).append(score)
    if len(signatures) <= 1:
        return

    descriptions = ", ".join(
        f"strip={score.source_prefix!r}/add={score.target_prefix!r}"
        for score in contenders[:8]
    )
    if len(contenders) > 8:
        descriptions += f", ... (+{len(contenders) - 8})"
    raise CheckpointCompatibilityError(
        "Checkpoint prefix mapping is ambiguous: equally compatible transforms "
        f"select different source tensors ({descriptions}). Remove duplicate "
        "model copies or provide an unambiguous component checkpoint."
    )


def remap_state_dict(
    source_state_dict: Mapping[str, Any],
    target_state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[TensorStateDict, PrefixScore]:
    """Remap the best-scoring source prefix to a target state dict.

    Only exact key-and-shape matches are returned.  ``strict=False`` permits a
    partial load, but still raises if no tensor can be matched; shape-mismatched
    tensors are never passed to :meth:`torch.nn.Module.load_state_dict`.
    """

    source = _copy_state_dict(source_state_dict, context="source state_dict")
    target = _copy_state_dict(target_state_dict, context="target state_dict")
    scores = score_state_dict_prefixes(source, target)
    if not scores or not scores[0].matched_keys:
        raise CheckpointCompatibilityError(
            "Checkpoint has no tensor with an exact target key and shape"
        )
    _raise_if_ambiguous(source, scores)
    best = scores[0]
    if strict and not best.is_exact:
        raise CheckpointCompatibilityError(_compatibility_message(best))

    matched_set = set(best.matched_keys)
    remapped: TensorStateDict = OrderedDict()
    for source_key, source_tensor in source.items():
        if best.source_prefix and not source_key.startswith(best.source_prefix):
            continue
        remainder = source_key[len(best.source_prefix) :]
        target_key = f"{best.target_prefix}{remainder}"
        if target_key in matched_set:
            remapped[target_key] = source_tensor
    return remapped, best


def _load_weights(
    model: nn.Module,
    source: CheckpointSource,
    *,
    strict: bool,
) -> LoadReport:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")

    payload = load_checkpoint_payload(source)
    state, filtered = filter_legacy_loss_keys(payload["state_dict"])
    target = model.state_dict()
    remapped, score = remap_state_dict(state, target, strict=strict)

    # remapped contains shape-compatible target keys only.  For partial loads,
    # PyTorch reports the same missing keys captured by PrefixScore.
    model.load_state_dict(remapped, strict=strict)
    return LoadReport(
        source=_source_name(source),
        source_prefix=score.source_prefix,
        target_prefix=score.target_prefix,
        matched_keys=score.matched_keys,
        missing_keys=score.missing_keys,
        unexpected_keys=score.unexpected_keys,
        shape_mismatches=score.shape_mismatches,
        filtered_legacy_loss_keys=filtered,
    )


def load_autoencoder_weights(
    model: nn.Module, path: CheckpointSource, strict: bool = True
) -> LoadReport:
    """Load an AutoencoderKL from a full OpenSR or component checkpoint.

    Released full checkpoints use ``first_stage_model.*`` keys; raw
    AutoencoderKL checkpoints use ``encoder.*``, ``decoder.*``,
    ``quant_conv.*``, and ``post_quant_conv.*``.  Prefix selection is based on
    the supplied model's exact state dict, so Lightning/DDP wrapper prefixes
    are handled without hard-coded key replacement.
    """

    return _load_weights(model, path, strict=strict)


def load_diffusion_weights(
    model: nn.Module, path: CheckpointSource, strict: bool = True
) -> LoadReport:
    """Load a full LatentDiffusion model from native or Lightning weights.

    Only tensors below a module segment named exactly ``loss`` are filtered for
    backwards compatibility with historical LPIPS/discriminator checkpoints.
    Every remaining key and shape must match when ``strict=True``.
    """

    return _load_weights(model, path, strict=strict)


def extract_autoencoder_state_dict(
    checkpoint: CheckpointSource,
    target_state_dict: Mapping[str, Any] | None = None,
    *,
    strict: bool = True,
) -> TensorStateDict:
    """Extract standalone AutoencoderKL tensors from any supported checkpoint.

    Supplying ``target_state_dict`` enables exact key/shape validation and is
    recommended.  Without it, extraction requires all four stable
    AutoencoderKL roots and returns only those roots with wrapper prefixes
    removed.
    """

    payload = load_checkpoint_payload(checkpoint)
    source, _ = filter_legacy_loss_keys(payload["state_dict"])
    if target_state_dict is not None:
        remapped, _ = remap_state_dict(source, target_state_dict, strict=strict)
        return remapped

    prefixes = {""}
    for key in source:
        for root in _AUTOENCODER_ROOTS:
            index = key.find(root)
            if index >= 0:
                prefixes.add(key[:index])

    candidates: list[tuple[int, int, str, TensorStateDict]] = []
    for prefix in prefixes:
        extracted: TensorStateDict = OrderedDict()
        roots_found: set[str] = set()
        for key, tensor in source.items():
            if prefix and not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            for root in _AUTOENCODER_ROOTS:
                if remainder.startswith(root):
                    extracted[remainder] = tensor
                    roots_found.add(root)
                    break
        if roots_found == set(_AUTOENCODER_ROOTS):
            candidates.append(
                (
                    len(extracted),
                    int("first_stage_model." in prefix),
                    prefix,
                    extracted,
                )
            )

    if not candidates:
        raise CheckpointCompatibilityError(
            "Could not find a complete AutoencoderKL component (encoder, "
            "decoder, quant_conv, and post_quant_conv)"
        )
    candidates.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
    return candidates[0][3]


def _safe_metadata_copy(value: Any, *, location: str = "metadata") -> Any:
    """Copy values accepted by PyTorch's restricted weights-only unpickler."""

    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (Tensor, nn.Parameter)):
        return value.detach()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointFormatError(
                    f"{location} contains a non-string key: {key!r}"
                )
            result[key] = _safe_metadata_copy(item, location=f"{location}.{key}")
        return result
    if isinstance(value, list):
        return [
            _safe_metadata_copy(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _safe_metadata_copy(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    raise CheckpointFormatError(
        f"{location} contains unsupported {type(value).__name__}; metadata must "
        "be weights-only-safe primitives, tensors, mappings, lists, or tuples"
    )


def _atomic_torch_save(
    payload: Mapping[str, Any], path: str | os.PathLike[str]
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            _best_effort_fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _best_effort_sync_directory(destination.parent)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _best_effort_fsync(file_descriptor: int) -> None:
    try:
        os.fsync(file_descriptor)
    except OSError:
        # Some network and virtual filesystems do not implement fsync. The
        # rename remains atomic there even when explicit durability is absent.
        pass


def _best_effort_sync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        _best_effort_fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_native_checkpoint(
    path: str | os.PathLike[str],
    state_dict: Mapping[str, Any],
    *,
    epoch: int,
    global_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a minimal checkpoint accepted by OpenSR inference.

    ``state_dict`` must already use native component or full-model keys.  This
    function intentionally does not serialize optimizers, callbacks, loggers,
    or wrapper modules.  A full LatentDiffusion state dict saved here is
    directly consumable by :meth:`SRLatentDiffusion.load_pretrained`.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an int")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be an int")
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")

    native_state = _copy_state_dict(state_dict)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "opensr_checkpoint_version": 1,
        "state_dict": native_state,
    }
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        payload["metadata"] = _safe_metadata_copy(metadata)
    return _atomic_torch_save(payload, path)


def materialize_diffusion_ema_weights(
    state_dict: Mapping[str, Any],
) -> TensorStateDict:
    """Copy EMA denoiser shadows into the ordinary inference weight keys.

    OpenSR's native checkpoint contract retains both model and model_ema
    tensors, but the public inference path executes model directly and does
    not enter an EMA context. This helper creates a structurally identical
    state dict in which every ordinary denoiser tensor is supplied by its EMA
    shadow. The EMA tensors themselves, schedule buffers, and first-stage
    tensors remain unchanged, so an export still satisfies the exact native
    key-and-shape contract and can also be used to initialize further training.

    LitEma names each shadow by removing dots from the corresponding model
    parameter name. The conversion verifies that this transformation is
    bijective and complete before replacing anything; a future architecture
    with a name collision or an untracked denoiser tensor is rejected instead
    of producing a partially EMA-materialized checkpoint.
    """

    state = _copy_state_dict(state_dict, context="native diffusion state_dict")
    model_keys = tuple(key for key in state if key.startswith(_DENOISER_PREFIX))
    shadow_keys = tuple(
        key
        for key in state
        if key.startswith(_EMA_PREFIX) and key not in _EMA_BOOKKEEPING_KEYS
    )
    missing_bookkeeping = tuple(sorted(_EMA_BOOKKEEPING_KEYS - set(state)))
    if not model_keys or not shadow_keys or missing_bookkeeping:
        details = (
            f"model_tensors={len(model_keys)}, ema_shadows={len(shadow_keys)}, "
            f"missing_ema_bookkeeping={list(missing_bookkeeping)}"
        )
        raise CheckpointCompatibilityError(
            "Cannot materialize EMA weights from an incomplete native diffusion "
            f"state dict ({details})"
        )

    shadow_by_flat_name = {key[len(_EMA_PREFIX) :]: key for key in shadow_keys}

    model_by_flat_name: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for key in model_keys:
        flat_name = key[len(_DENOISER_PREFIX) :].replace(".", "")
        previous = model_by_flat_name.get(flat_name)
        if previous is not None:
            collisions.setdefault(flat_name, [previous]).append(key)
        else:
            model_by_flat_name[flat_name] = key
    if collisions:
        preview = ", ".join(
            f"{name}: {keys}" for name, keys in tuple(collisions.items())[:4]
        )
        raise CheckpointCompatibilityError(
            "Cannot map EMA shadows because flattened denoiser names collide: "
            + preview
        )

    model_names = set(model_by_flat_name)
    shadow_names = set(shadow_by_flat_name)
    missing_shadows = tuple(sorted(model_names - shadow_names))
    orphan_shadows = tuple(sorted(shadow_names - model_names))
    if missing_shadows or orphan_shadows:
        raise CheckpointCompatibilityError(
            "EMA shadows do not exactly cover the ordinary denoiser state: "
            f"missing_shadows={list(missing_shadows[:8])}, "
            f"orphan_shadows={list(orphan_shadows[:8])}"
        )

    materialized: TensorStateDict = OrderedDict(state)
    for flat_name, model_key in model_by_flat_name.items():
        shadow_key = shadow_by_flat_name[flat_name]
        model_tensor = state[model_key]
        shadow_tensor = state[shadow_key]
        if tuple(model_tensor.shape) != tuple(shadow_tensor.shape):
            raise CheckpointCompatibilityError(
                f"EMA shadow {shadow_key!r} has shape {tuple(shadow_tensor.shape)}, "
                f"but {model_key!r} has shape {tuple(model_tensor.shape)}"
            )
        materialized[model_key] = shadow_tensor

    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        materialized._metadata = dict(metadata)  # type: ignore[attr-defined]
    return materialized


def _plain_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config_value(item) for item in value]
    return value


def build_native_diffusion_state_spec(
    model_config: Mapping[str, Any],
) -> StateDictSpec:
    """Build the exact native LatentDiffusion key/shape contract on ``meta``.

    Only the three state-owning components are instantiated. Schedule buffers
    are represented by shape-only meta tensors, avoiding both the full model's
    allocation and its NumPy schedule construction. The resulting spec matches
    the current ``LatentDiffusion.state_dict()`` layout, including EMA buffers.
    """

    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    try:
        first_stage_config = _plain_config_value(model_config["first_stage_config"])
        unet_config = _plain_config_value(model_config["cond_stage_config"])
    except KeyError as exc:
        raise ValueError(
            f"model_config is missing required section {exc.args[0]!r}"
        ) from exc
    denoiser_settings = _plain_config_value(model_config.get("denoiser_settings", {}))
    other = _plain_config_value(model_config.get("other", {}))
    if not isinstance(first_stage_config, Mapping) or not isinstance(
        unet_config, Mapping
    ):
        raise TypeError("first_stage_config and cond_stage_config must be mappings")
    if not isinstance(denoiser_settings, Mapping) or not isinstance(other, Mapping):
        raise TypeError("denoiser_settings and other must be mappings")
    if "embed_dim" not in first_stage_config:
        raise ValueError("first_stage_config.embed_dim is required")
    timesteps = int(denoiser_settings.get("timesteps", 1000))
    if timesteps <= 0:
        raise ValueError("denoiser_settings.timesteps must be positive")
    conditioning_key = "concat" if bool(other.get("concat_mode", True)) else "crossattn"

    # Local imports keep checkpoint inspection lightweight for callers that do
    # not need to construct an architecture contract.
    from opensr_model.autoencoder.autoencoder import AutoencoderKL
    from opensr_model.diffusion.latentdiffusion import DiffusionWrapper
    from opensr_model.diffusion.utils import LitEma

    with redirect_stdout(StringIO()), torch.device("meta"):
        denoiser = DiffusionWrapper(
            dict(unet_config), conditioning_key=conditioning_key
        )
        ema = LitEma(denoiser)
        autoencoder = AutoencoderKL(
            dict(first_stage_config), embed_dim=int(first_stage_config["embed_dim"])
        )

    expected: TensorStateDict = OrderedDict()
    schedule_placeholder = torch.empty(timesteps, device="meta", dtype=torch.float32)
    for key in _DIFFUSION_BUFFER_KEY_ORDER:
        expected[key] = schedule_placeholder
    for key, tensor in denoiser.state_dict().items():
        expected[f"model.{key}"] = tensor
    for key, tensor in ema.state_dict().items():
        expected[f"model_ema.{key}"] = tensor
    for key, tensor in autoencoder.state_dict().items():
        expected[f"first_stage_model.{key}"] = tensor
    return StateDictSpec.from_state_dict(expected)


def _coerce_state_dict_spec(
    expected: StateDictSpec | Mapping[str, Any],
) -> StateDictSpec:
    if isinstance(expected, StateDictSpec):
        return expected
    if isinstance(expected, Mapping):
        return StateDictSpec.from_state_dict(expected)
    raise TypeError(
        "expected_full_state_dict must be a StateDictSpec or tensor mapping"
    )


def _outer_prefixes_for_full_state(keys: Sequence[str]) -> set[str]:
    prefixes = {""}
    anchors = (
        "betas",
        "model.diffusion_model.",
        "first_stage_model.encoder.",
        "model_ema.",
    )
    for key in keys:
        for anchor in anchors:
            index = key.find(anchor)
            if index >= 0:
                prefixes.add(key[:index])
    prefixes.update(
        prefix
        for prefix in _KNOWN_PREFIXES
        if prefix and any(key.startswith(prefix) for key in keys)
    )
    return prefixes


def _normalize_native_full_state(state_dict: Mapping[str, Any]) -> TensorStateDict:
    state, filtered = filter_legacy_loss_keys(state_dict)
    candidates: list[tuple[int, int, str, TensorStateDict]] = []
    for prefix in _outer_prefixes_for_full_state(tuple(state)):
        normalized: TensorStateDict = OrderedDict()
        for key, tensor in state.items():
            if prefix and not key.startswith(prefix):
                continue
            normalized[key[len(prefix) :]] = tensor

        has_required_structure = (
            _DIFFUSION_BUFFER_KEYS.issubset(normalized)
            and any(key.startswith("model.diffusion_model.") for key in normalized)
            and any(key.startswith("model_ema.") for key in normalized)
            and any(key.startswith("first_stage_model.encoder.") for key in normalized)
            and any(key.startswith("first_stage_model.decoder.") for key in normalized)
            and "first_stage_model.quant_conv.weight" in normalized
            and "first_stage_model.post_quant_conv.weight" in normalized
        )
        if not has_required_structure:
            continue

        allowed = _DIFFUSION_BUFFER_KEYS
        unexpected = [
            key
            for key in normalized
            if key not in allowed
            and not key.startswith("model.diffusion_model.")
            and not key.startswith("model_ema.")
            and not key.startswith("first_stage_model.")
        ]
        if unexpected:
            continue
        candidates.append(
            (len(normalized), -_prefix_part_count(prefix), prefix, normalized)
        )

    if not candidates:
        suffix = f" Filtered legacy loss tensors: {len(filtered)}." if filtered else ""
        raise CheckpointCompatibilityError(
            "Base checkpoint is not a native full LatentDiffusion state dict "
            "with schedule, UNet, EMA, and first-stage tensors." + suffix
        )
    largest_candidate = max(item[0] for item in candidates)
    equally_complete = [item for item in candidates if item[0] == largest_candidate]
    if len(equally_complete) > 1:
        prefixes = ", ".join(repr(item[2]) for item in equally_complete[:8])
        if len(equally_complete) > 8:
            prefixes += f", ... (+{len(equally_complete) - 8})"
        raise CheckpointCompatibilityError(
            "Base checkpoint contains multiple equally complete native diffusion "
            f"states under prefixes {prefixes}; the full-model mapping is ambiguous"
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][3]


def merge_autoencoder_checkpoint(
    base_checkpoint: CheckpointSource,
    autoencoder_state_dict: CheckpointSource,
    output_path: str | os.PathLike[str],
    *,
    expected_full_state_dict: StateDictSpec | Mapping[str, Any],
    epoch: int,
    global_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Merge trained standalone AE weights into a full native checkpoint.

    The base payload and both input state mappings remain untouched.  The base
    provides every diffusion schedule, UNet, EMA, and first-stage key; exactly
    the ``first_stage_model.*`` tensors are replaced after strict key/shape
    validation.  The result is atomically written in the native format used by
    :meth:`SRLatentDiffusion.load_pretrained`.
    """

    expected_spec = _coerce_state_dict_spec(expected_full_state_dict)
    base_payload = load_checkpoint_payload(base_checkpoint)
    base_state = _normalize_native_full_state(base_payload["state_dict"])
    expected_spec.validate(base_state, context="Base checkpoint state_dict")

    prefix = "first_stage_model."
    target_autoencoder: TensorStateDict = OrderedDict(
        (key[len(prefix) :], tensor)
        for key, tensor in base_state.items()
        if key.startswith(prefix)
    )
    if not target_autoencoder:
        raise CheckpointCompatibilityError(
            "Base checkpoint has no first_stage_model tensors"
        )

    ae_payload = load_checkpoint_payload(autoencoder_state_dict)
    ae_source, _ = filter_legacy_loss_keys(ae_payload["state_dict"])
    trained_autoencoder, _ = remap_state_dict(
        ae_source, target_autoencoder, strict=True
    )

    # Structural copies avoid changing either source mapping.  Every tensor not
    # owned by the autoencoder remains exactly the base tensor object/value.
    merged: TensorStateDict = OrderedDict(base_state.items())
    for key, tensor in trained_autoencoder.items():
        merged[f"{prefix}{key}"] = tensor

    if tuple(merged) != tuple(base_state):
        raise CheckpointCompatibilityError(
            "Autoencoder merge unexpectedly changed the full state-dict key set"
        )
    expected_spec.validate(merged, context="Merged checkpoint state_dict")
    merged_metadata: dict[str, Any] = {
        "kind": "latent_diffusion",
        "autoencoder_merged": True,
    }
    if metadata is not None:
        merged_metadata.update(dict(metadata))
    return save_native_checkpoint(
        output_path,
        merged,
        epoch=epoch,
        global_step=global_step,
        metadata=merged_metadata,
    )


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointFormatError",
    "LoadReport",
    "PrefixScore",
    "ShapeMismatch",
    "StateDictSpec",
    "build_native_diffusion_state_spec",
    "extract_autoencoder_state_dict",
    "filter_legacy_loss_keys",
    "load_autoencoder_weights",
    "load_checkpoint_payload",
    "load_diffusion_weights",
    "materialize_diffusion_ema_weights",
    "merge_autoencoder_checkpoint",
    "remap_state_dict",
    "save_native_checkpoint",
    "score_state_dict_prefixes",
]
