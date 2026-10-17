# Changelog

## v1.2.1 — 2026-10-17

This maintenance release improves the notebooks, documentation, test coverage,
and training workflows introduced in v1.2.0.

### Notebooks and sampler examples

- Refresh the no-code, walkthrough, LDSR-S2/SEN2SR showcase, and sampling Colab
  notebooks, with documentation images organized under `resources/images/`.
- Add a reproducible DDIM/DPM-Solver++ comparison notebook, measured runtime
  and quality tables, and example figures. Measurements describe their stated
  evaluation subset and hardware, rather than a guaranteed speed or quality gain.
- Include the public notebooks, figures, measurements, and tests in the source
  distribution; keep model weights and local research artifacts separate.

### Training and fixes

- Organize training code into data, modules, runtime, and validation packages,
  while retaining aliases for previous module imports and the existing CLI.
- Package training presets under `configs/training_configs/`, with compatible
  config-name lookup, a SEN2NAIP fine-tuning preset, SEN2NEON validation manifest,
  and raw Sentinel-2 preview examples.
- Improve training data validation, deterministic source mixing, provenance,
  external validation, checkpoint selection, distributed metric aggregation,
  and logging. Add a `--prepare-only` data-audit option for transfer validation.
- Add optional query-chunked autoencoder attention without changing checkpoint
  parameter keys, and apply consistent code formatting throughout the repository.

### Tests and release checks

- Expand regression coverage for samplers, inference, spectral correction,
  training data, configuration, checkpoint contracts, and validation lifecycle.
- Run Black and the CPU test suite with coverage through pre-commit/pre-push
  hooks and the release workflow; PyPI publishing waits for those checks.
- Validate distribution metadata and release-tag/version agreement, preserve
  build artifacts, and attach the published wheel, source archive, and SHA-256
  checksums to the GitHub release.

### Upgrade and compatibility

Python **3.12 or newer** is required:

```bash
pip install --upgrade 'opensr-model==1.2.1'
# Include the optional training dependencies:
pip install --upgrade 'opensr-model[train]==1.2.1'
```

Inference defaults remain **3-step DPM-Solver++ with back-projection**. DDIM and
the other spectral-correction options remain available. V1 remains the canonical
checkpoint and default for model loading; this package release does not change
the model architecture or publish new weights. For scripts using training YAML
paths directly, use `opensr_model/configs/training_configs/train_*.yaml`.

See the [training guide](training.md), [public notebooks](../resources/notebooks/),
and [full comparison with v1.2.0](https://github.com/ESAOpenSR/opensr-model/compare/v1.2.0...v1.2.1).

## v1.2.0 — 2026-10-12

This release adds fast diffusion sampling, a complete two-stage training workflow,
and device-native spectral correction for four-band Sentinel-2 super-resolution.

### Sampling

- Add DPM-Solver++ 2M through Diffusers, using the loaded checkpoint's noise schedule.
- Make DPM-Solver++ with **3 sampling steps** the inference default. DDIM remains
  available, and sampling steps can be selected per call or in the model config.
- Support the new sampler in Lightning prediction, uncertainty sampling, and
  decoded intermediate outputs. Diffusers is now installed with the base package;
  the `dpm` extra remains a compatibility alias.

### Training

- Add independent PyTorch Lightning stages for the autoencoder and conditional
  diffusion model, with packaged YAML recipes and the `opensr-train`,
  `opensr-train-autoencoder`, and `opensr-train-diffusion` commands.
- Add paired TACO data loading, aligned augmentation, nodata masking, deterministic
  dataset splits, and support for SEN2NAIPv2 cross-sensor RGB-NIR pairs.
- Add reconstruction/perceptual/adversarial autoencoder losses, P2-weighted
  diffusion training with optional detail losses, EMA validation, and RGB/NIR
  metrics and image logging through TensorBoard or optional W&B.
- Save separate Lightning resume checkpoints and native inference exports, with
  inference metadata validation and support for pretrained initialization.

### Spectral matching and fixes

- Make PyTorch iterative **back-projection** the default spectral correction,
  matching the downsampled SR image to the observed LR input on the image's device.
- Keep per-band histogram matching and uncorrected output available through
  `spectral_correction="histogram"` and `spectral_correction="none"`.
- Add configurable blur, iteration limit, tolerance, relaxation, masks, and
  standalone convergence diagnostics. Sampler and correction are independent.
- Fix rectangular nodata-mask dimensions and expand inference, diffusion,
  spectral-correction, and training regression coverage.

### Upgrade and compatibility

Python **3.12 or newer** is required. Install the release with:

```bash
pip install --upgrade 'opensr-model==1.2.0'
# Include the optional training dependencies:
pip install --upgrade 'opensr-model[train]==1.2.0'
```

**Default inference output changes in this release.** To explicitly select the
previous sampler and correction:

```python
sr = model(image, sampler="ddim", sampling_steps=100, spectral_correction="histogram")
```

Existing explicit config values, checkpoint inference metadata, and call arguments
still take precedence where supported. Apply runtime overrides after loading a
checkpoint. The model architecture, RGB-NIR band order, 4× scale, and released
checkpoint remain unchanged; this package release does not include new weights.
Three-step sampling prioritizes speed; quality varies by scene. Back-projection
enforces the selected observation model and does not clamp the corrected output,
so values can fall outside `[0, 1]`.

See the [training guide](training.md), the detailed inference notes below, and
the [full comparison with v1.1.1](https://github.com/ESAOpenSR/opensr-model/compare/v1.1.1...v1.2.0).

## Inference changelog since the original release

Updated 12 October 2026. The baseline for this document is the **v1.0.0 release
(`034d10f`, 18 June 2025)** and the released
`opensr-ldsrs2_v1_0_0.ckpt` checkpoint. It records the subsequent inference,
packaging, deployment and evaluation changes through v1.2.0, including the
PyTorch spectral-correction integration and inference default changes.
Package releases do not imply new published model checkpoints.

Training instructions are in [training.md](training.md).

## Current behavior

The released model architecture, four RGB-NIR input/output bands, 4× spatial
upscaling, and native checkpoint tensor layout are unchanged by the new sampler
and correction options. No new weights are required.

**Inference defaults have changed:** `model(image)` now uses **DPM-Solver++
with 3 steps and PyTorch back-projection**, replacing DDIM with 100 steps and
per-band histogram matching. This applies to the bundled YAML and configs
without sampler/correction fields. Explicit config values, checkpoint inference
settings and call arguments still take precedence over defaults. To select
the previous inference settings, use:

```python
sr = model(image, sampler="ddim", sampling_steps=100, spectral_correction="histogram")
```

Two independent decisions can now be made:

| Decision | Options | Purpose |
|---|---|---|
| Diffusion sampler | `ddim`, `dpm_solver++` | Generate the latent SR prediction |
| Decoded-image correction | `histogram`, `back_projection`, `none` | Adjust the prediction using the observed LR image |

Changing either choice can change output pixels. These options are not claims
of bitwise reproduction of every older implementation or of the published paper.

## Timeline

| Milestone | Changes relevant to inference |
|---|---|
| v1.0.0, 18 June 2025 | Original stable LDSR-S2 release: DDIM inference, RGB-NIR 4× output, histogram correction and uncertainty sampling |
| v1.0.1–v1.0.8, June–October 2025 | Sampling argument/config naming alignment, dependency and documentation fixes, NaN/Inf handling, updated large-file integration |
| v1.1.0, 12 March 2026 (`75c4e31`) | Python 3.12 dependency modernization and packaged HPC/Slurm workflows |
| v1.1.1, 8 May 2026 (`10f4c01`) | Packaging/checkpoint safeguards, aligned examples and sampling APIs, expanded deployment support |
| Post-v1.1.1, August 2026 | Checkpoint inference metadata handling, schedule/config alignment, inference/metric regression coverage and ablation utilities; training changes documented separately |
| 12 October 2026 (`5b422ce`, `e32ff75`) | Optional DPM-Solver++ 2M integration and its automatic 3-step setting |
| v1.2.0, 12 October 2026 | Two-stage training; selectable PyTorch back-projection; rectangular nodata-mask fix; DPM-Solver++ (3 steps) and back-projection become the inference defaults; Diffusers becomes a base dependency |

## DPM-Solver++ 2M

### What changed

DPM-Solver++ 2M uses the trained checkpoint's noise schedule. It adds no
per-step noise, but initial noise and conditioning still depend on the random seed.

### Selection and defaults

```bash
pip install -e .
```

```python
sr_default = model(image)  # DPM-Solver++ (3 steps) + back-projection
sr_explicit = model(image, sampler="dpm_solver++", spectral_correction="back_projection")
sr_five = model(image, sampler="dpm_solver++", sampling_steps=5)
sr_ddim = model(image, sampler="ddim", sampling_steps=100)
```

The precedence for step count is an explicit `sampling_steps` argument, then
an integer in `denoiser_settings.sampling_steps`, then the automatic default:
**100 for DDIM, 3 for DPM-Solver++**. The bundled YAML sets `sampling_steps: null`
to allow that automatic choice. Old configs with explicit counts retain them.
Sampler selection uses the call argument, then `denoiser_settings.sampler`,
then DPM-Solver++ if the field is absent. Explicit DDIM selections remain valid.

DPM-Solver++ accepts a positive integer step count below the training timestep
count. Native DDIM additionally requires that the requested count divide the
training timestep count exactly; invalid schedules now fail explicitly.

Configured eta/temperature values apply to DDIM only. For DPM-Solver++, explicit
`sampling_eta` must be omitted or zero and `sampling_temperature` must be omitted
or one. Unsupported overrides raise an error rather than being silently ignored.
Diffusers is now a base installation dependency, so the default inference path
works with a standard installation. It is still imported only when this sampler
is selected. The `dpm` extra remains a compatibility alias for existing install
commands; missing installations produce an error with that installation instruction.

The selection also works through the Lightning wrapper, saved intermediate
iterations and uncertainty sampling. Existing training validation and the
deprecated attribution/explainer method retain their DDIM path.

The 3-step default prioritizes speed; quality and timing vary by scene and hardware.

## New spectral correction: iterative back-projection

### Why add it

Histogram matching independently aligns the value distribution of each band
with its LR reference. That can correct global appearance, but it does not
enforce pixelwise agreement with the observation. Independent nonlinear band
transformations also need not preserve spectral angles or band ratios.

An HR histogram need not equal an LR histogram: resolving a mixed pixel into
bright and dark components legitimately changes its distribution. For this
reason a histogram correction can modify even a perfectly correct HR reference.

Back-projection instead enforces an explicit observation model. If `y` is the
original LR image, `x` is the current decoded SR image and `D` is blur plus
downsampling, the target is `D(x) ≈ y`.

### Settings

Correction runs on the image's device. Float64 inputs retain their precision;
float16/bfloat16 inputs produce a float32 result.

| Option | Default | Meaning |
|---|---:|---|
| `scale` | 4 | HR/LR spatial ratio; model inference requires 4 |
| `blur_sigma` | 0.0 | Gaussian sigma in HR pixels; scalar or one value per band |
| `max_iterations` | 100 | Maximum updates; zero returns the initial masked image |
| `tolerance` | 1e-6 | Per-image masked LR RMSE threshold in reflectance units |
| `relaxation` | 1.0 | Fraction of the conservative step, in `(0, 1]` |

The default uses pure 4×4 block averaging. One full update corrects those block
means, apart from floating-point precision. An optional Gaussian blur uses a
four-sigma truncated kernel and periodic boundaries. Per-band values such as
`[0.8, 1.0, 1.2, 1.4]` are illustrative, **not a calibrated Sentinel-2 MTF**. Kernel and
boundary choices need evaluation for real imagery and tiled production use.

### Select it per call

```python
sr_hist = model(image, spectral_correction="histogram")
sr_bp = model(image, spectral_correction="back_projection")
sr_uncorrected = model(image, spectral_correction="none")

# Sampler and correction are independent.
sr_fast_bp = model(
    image,
    sampler="dpm_solver++",
    spectral_correction="back_projection",
)

# Optional per-band blur settings; adapt these to your imagery.
sr_blur = model(
    image,
    spectral_correction="back_projection",
    back_projection={
        "blur_sigma": [0.8, 1.0, 1.2, 1.4],
        "max_iterations": 400,
        "tolerance": 1e-7,
        "relaxation": 1.0,
    },
)
```

`back_projection` also accepts a `BackProjectionConfig` object. An explicit
options mapping replaces the configured mapping; omitted entries in that
mapping take the dataclass defaults. Passing back-projection options with a
different correction mode raises an error.

For YAML-based applications, including model construction in the HPC runner:

```yaml
spectral_correction: back_projection
back_projection:
  scale: 4
  blur_sigma: 0.0
  max_iterations: 100
  tolerance: 0.000001
  relaxation: 1.0
```

These are top-level model config settings, separate from `denoiser_settings`.
They are runtime postprocessing choices: the current checkpoint metadata
loader does not restore or export them as part of its supported inference
metadata fields. Set them in the model YAML or supply call arguments. A caller
that explicitly passes `histogram_matching=False` disables a YAML-selected
correction unless it also passes an explicit `spectral_correction` argument.

### Compatibility and ordering

| Call/config combination | Selected correction |
|---|---|
| Config without a `spectral_correction` field, no new arguments | Back-projection |
| Bundled current config, no new arguments | Back-projection |
| Config with `spectral_correction: histogram`, no new arguments | Histogram matching |
| `histogram_matching=False`, no explicit new selection | None |
| Explicit `spectral_correction="back_projection"` | Back-projection, even if the legacy bool is false |
| Explicit `spectral_correction="none"` | None, even if config selects correction |
| No explicit selection, legacy bool true | `config.spectral_correction`, or back-projection if absent |

Normal inference decodes and denormalizes first, applies the nodata mask,
removes padding and then performs the selected correction against the original
unpadded LR reference. `none` disables this final correction; the historical
decoder's own normalization/clamping behavior still applies.

With `save_iterations=True`, each decoded frame is corrected on the padded
grid. The returned frames retain the existing padded dimensions rather than
adopting the normal final-output crop. The Lightning `forward`/`predict_step`
wrappers pass through these options;
both uncertainty helpers support the same selection for every draw. The
deprecated explainer retains its existing DDIM/histogram interface.

The new path follows existing **per-band zero-as-nodata** semantics: zero LR
values are excluded from the observation residual, and corresponding HR pixels
stay zero throughout optimization. This is an application convention, not a
claim that measured zero reflectance is physically impossible. Gaussian blur
near such gaps uses this zero-filled support, which is an approximation at
nodata boundaries. A fully invalid sample produces zero output and is reported
as having no usable measurements in standalone diagnostics.

Rectangular nodata masks now use `(4 * height, 4 * width)` instead of a square
derived from width alone. `create_no_data_mask` still accepts its previous
integer size argument and additionally accepts `(height, width)`.

There is **no clamp after back-projection**. Clamping can break observation
consistency; unconstrained correction can produce negative values or values
above one. Small red+NIR denominators can also make NDVI unstable. Bounds or
other constraints would require a separately evaluated constrained solver.

### Standalone use and diagnostics

```python
from opensr_model.spectral_correction import BackProjectionConfig, back_projection

corrected, info = back_projection(
    decoded_sr, lr_reference,
    BackProjectionConfig(blur_sigma=0.0),
    return_info=True,
)
print(info["lr_rmse"], info["iterations"], info["converged"])
```

The standalone function accepts matching CHW or BCHW inputs, preserves that
rank, and requires both inputs on the same device. Optional boolean
`valid_mask` and `output_mask` control LR measurements and HR support. Masks
broadcast to BCHW; a per-image spatial mask should be B1HW. Without
`output_mask`, unmeasured HR pixels are not automatically forced to zero.
Diagnostics are tensors on the input device, including a residual history and
valid-measurement counts. No-observation samples are not labeled converged.
Reaching the iteration limit returns the current result; use diagnostics when
the convergence status matters.

## Other changes since v1.0.0

### Inference API and preprocessing

The public forward parameters were aligned from `eta`, `custom_steps` and
`temperature` to `sampling_eta`, `sampling_steps` and `sampling_temperature`,
with config fallback. Update old keyword calls accordingly. The Lightning
uncertainty helper retains a `custom_steps` compatibility alias. Examples use
the current names. The original forward signature defaulted eta to 1.0;
the bundled current DDIM config uses eta 0.95 and temperature 1.0. NaNs and
positive/negative infinities are converted to zero
at the public inference entry point. Conditioning upsampling now respects
separate height and width; small-input padding and output cropping remain in use.

Before this new correction integration, histogram matching had already been
moved to the final postprocessing stage and made explicit per sample and band,
using the unpadded LR reference for normal outputs. It still uses scikit-image
on CPU. Histogram equality is not a guarantee of pixelwise spectral fidelity.

### Checkpoint loading and configuration

Checkpoint downloads check for failures before saving the final file. Loading
validates weights and supported inference settings. Checkpoint metadata may
restore sampling, normalization and conditioning settings, so apply runtime
overrides after `load_pretrained`.

The bundled YAML now documents and uses the released checkpoint's actual
beta endpoints, **0.0015 → 0.0155**, and disabled input normalization. The
v1.0.0 YAML advertised 0.0001 → 0.01; strict native loading restores the
checkpoint's persisted schedule buffers. Correcting the YAML does not retrain
or rewrite the released checkpoint. DPM-Solver++ consumes those loaded buffers.

### Packaging, full-image processing and deployment

The package now declares Python >=3.12 and supported minimum versions of the
inference dependencies, includes matplotlib, Diffusers and packaged YAML resources,
and provides optional `hpc` and `test` extras. The former optional `dpm` extra is
retained as a compatibility alias now that its dependency is installed by default.
Package version reporting keeps
the original installed-distribution lookup and now exposes `__version__` in
addition to `version`. Training extras/entry points are described
in [training.md](training.md).

The large-file example was updated to `opensr_utils.large_file_processing`
with windowing, overlap, edge trimming, device routing and current argument
names. The installable `opensr-hpc` CLI stages Sentinel-2 cutouts and supports
single-patch, grid and shapefile-AOI submissions through Slurm. It includes
config validation, packaged task entrypoints, configurable batch size,
run/patch manifests, structured logs, metadata and compressed GeoTIFF outputs.
See [deployment/README.md](../deployment/README.md) for operational details.

### Validation

Run the inference regression tests with:

```bash
python -m pytest tests/test_inference.py tests/test_inference_utils.py \
  tests/test_spectral_correction.py tests/test_spectral_inference.py \
  tests/diffusion -q
```

Training tests require the `train` extra. Observation consistency does not
guarantee accurate subpixel spectra; results depend on the chosen blur/downsampling
model.
