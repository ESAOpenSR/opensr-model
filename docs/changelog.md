# Inference changelog since the original release

Updated 12 October 2026. The baseline for this document is the **v1.0.0 release
(`034d10f`, 18 June 2025)** and the released
`opensr-ldsrs2_v1_0_0.ckpt` checkpoint. It records the subsequent inference,
packaging, deployment and evaluation changes, including the current unreleased
PyTorch spectral-correction integration and inference default changes. The package version is still 1.1.1;
an entry here does not imply a new published package or checkpoint.

Fine-tuning, training objectives, data loading for training, optimizer settings,
training logs and training commands are documented in [training.md](training.md).
Those details are intentionally kept there.

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
| Current unreleased work, 12 October 2026 | Synthetic and real-HR spectral-correction experiments; selectable native PyTorch back-projection; rectangular nodata-mask fix; DPM-Solver++ (3 steps) and back-projection become the inference defaults; Diffusers becomes a base dependency; this changelog |

## DPM-Solver++ 2M

### What changed

[diffusion/dpm_solver.py](../opensr_model/diffusion/dpm_solver.py) adds an
adapter around Diffusers' `DPMSolverMultistepScheduler`. It calls the existing
OpenSR epsilon-predicting denoiser and uses the beta schedule loaded from the
checkpoint, rather than substituting a generic diffusion schedule.

The configured protocol is:

- `algorithm_type="dpmsolver++"`, solver order 2, multistep midpoint updates;
- epsilon prediction, linspace timestep spacing, zero steps offset;
- lower-order final handling and terminal sigma zero;
- no Karras sigmas, dynamic thresholding, or zero-SNR beta rescaling;
- fresh scheduler history on every inference call.

Each step makes one denoiser evaluation. Solver order two does not double that
count. This is the ODE variant, without per-step stochastic noise; initial
latent noise and sampled conditioning still depend on the random seed.

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

### Evidence behind the 3-step setting

The local sampler experiment contains 4,608 fresh predictions on 64
SEN2NAIPv2 images with three seeds, plus a separately labeled historical DDIM
series. Under that experiment's two RTX 3090 GPUs, FP32, batch size one,
128×128 LR inputs and histogram correction, the measured means were:

| Protocol | Seconds/image | Paired speedup versus DDIM-100 | RGB PSNR difference | NIR PSNR difference |
|---|---:|---:|---:|---:|
| DDIM-100, eta 0.95 | 3.875 | 1.000× | baseline | baseline |
| DPM++ 2M, 5 steps | 0.672 | 5.763× | +0.130 dB | +0.070 dB |
| DPM++ 2M, 3 steps | 0.607 | 6.389× | +0.112 dB | +0.047 dB |
| DPM++ 2M, 1 step | 0.559 | 6.936× | −2.000 dB | −0.800 dB |

These are recorded benchmark results, not a timing promise for another machine.
The 3-step default prioritizes speed. Cross-sensor reference metrics do not
establish equal perceptual quality on every scene, and the benchmark images
are not guaranteed to have been unseen during the original training.
See the [sampler report](../benchmarks/ddim_steps/output/sampler-comparison-20261012/report.md)
for matched controls, settings sensitivity and detailed methodology.

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

Each iteration computes:

```text
residual = y - D(x)
x = x + relaxation * scale² * D_transpose(residual)
```

`D_transpose` is the actual adjoint, including averaging weights, not an
arbitrary resize. For the implemented normalized blur and block average,
`||D|| ≤ 1/scale`; `0 < relaxation ≤ 1` therefore gives a conservative
Landweber step for reducing squared observation error. With noiseless,
consistent data this preserves components invisible to the observation
operator, including any incorrect fine detail in the initial prediction.

### Production implementation

[spectral_correction.py](../opensr_model/spectral_correction.py) contains:

- `BackProjectionConfig`: validated scale, blur, iteration and stopping options;
- `BlurDownsample`: batched Gaussian blur/block averaging and its adjoint;
- `back_projection`: correction with per-image stopping and optional diagnostics.

The implementation uses PyTorch tensor operations and depthwise separable
convolutions. It runs on the image's device, without NumPy conversion or CPU
image round trips. Small kernel tensors are constructed once per correction.
The stopping decision can synchronize a scalar with the host each iteration.
Float64 inputs retain float64; float16/bfloat16 are promoted to float32 and
**the corrected result is returned in float32**. Autocast is disabled inside
the correction. It is an inference operation under `no_grad`.

| Option | Default | Meaning |
|---|---:|---|
| `scale` | 4 | HR/LR spatial ratio; model inference requires 4 |
| `blur_sigma` | 0.0 | Gaussian sigma in HR pixels; scalar or one value per band |
| `max_iterations` | 100 | Maximum updates; zero returns the initial masked image |
| `tolerance` | 1e-6 | Per-image masked LR RMSE threshold in reflectance units |
| `relaxation` | 1.0 | Fraction of the conservative step, in `(0, 1]` |

The default uses pure 4×4 block averaging. One full update corrects those block
means, apart from floating-point precision. An optional Gaussian blur uses a
four-sigma truncated kernel and periodic boundaries, matching the controlled
benchmarks. For example `[0.8, 1.0, 1.2, 1.4]` reproduces their illustrative
RGB-NIR blur. These values are **not a calibrated Sentinel-2 MTF**. Kernel and
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

# Optional blur-aware settings, matching the experimental operator.
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
adopting the normal final-output crop. Intermediate references are retained in
physical units even if conditioning normalization mutates its working tensor.
The Lightning `forward`/`predict_step` wrappers pass through these options;
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

## Spectral-correction experiments

Two CPU benchmark versions were added under `benchmarks/spectral_matching`.
They compare uncorrected predictions, histogram matching, back-projection and
histogram matching followed by back-projection. The combined method is an
experimental comparison, not an additional production selection.

**Version 1** uses procedural four-band scenes with known HR truth: five seeds,
two observation operators, eight scenarios and four methods, giving **320
evaluations**. Scenarios include gain/offset, spatial bias, blur, displacement,
invisible checkerboard detail, perfect input, noisy observations and incorrect
blur. The observation operators are block averaging and band-dependent
Gaussian blur followed by averaging.

**Version 2** uses the first eight fixed evaluation HR scenes from the existing
SEN2NAIPv2 manifest. These are real NAIP-derived RGB-NIR images at 2.5 m that
the dataset has already harmonized to S2, not unmodified raw NAIP radiometry.
Checksums, central 256×256 crop windows, band order and scaling by `1e-4` are
recorded. Existing real S2 observations are not used: LR is generated from HR
to establish an exact reference. An additional bilinear-from-LR case brings
the total to **576 evaluations**. No diffusion checkpoint is run in either
correction experiment.

The headline below averages only gain/offset and spatial-bias cases equally
across scenes/seeds and both operators:

| Experiment / method | HR RMSE ↓ | SSIM ↑ | Spectral angle ° ↓ | NDVI RMSE ↓ |
|---|---:|---:|---:|---:|
| Procedural / histogram | 0.0279707 | 0.935605 | 5.10785 | 0.205153 |
| Procedural / back-projection | 0.00430345 | 0.996960 | 0.786681 | 0.0743939 |
| Real HR / histogram | 0.0104802 | 0.980629 | 2.41227 | 0.0559325 |
| Real HR / back-projection | 0.0013208 | 0.999028 | 0.265930 | 0.00771094 |

Back-projection reduced mean HR RMSE by 84.6% in the procedural radiometric
cases (20/20 paired wins) and 87.4% in the real-HR radiometric cases (32/32).
All 160 and 288 respective back-projection/combined runs reached their specified
LR RMSE threshold. These measurements originally used the independent
NumPy/SciPy reference implementation, with 400 maximum iterations and tolerance
`1e-7`; they are not timing measurements of the production PyTorch integration.

The complete results show limits as well as gains:

- Real-HR bilinear-from-LR RMSE changed from 0.0135491 uncorrected to 0.0122715
  with back-projection: consistency does not restore all missing detail.
- Real-HR invisible-checkerboard RMSE stayed 0.0165303 under back-projection;
  histogram matching achieved 0.0130477. Back-projection is not uniformly best.
- Fitting noise or the wrong operator can damage an otherwise perfect input.
- A correct noiseless operator leaves an already consistent HR image unchanged.
- Negative reflectances and unstable indices are measured rather than clipped
  away in the comparisons.

Metrics include HR RMSE, fixed-range PSNR, per-band SSIM, spectral angle,
NDVI RMSE, per-band bias/RMSE, LR observed/clean/assumed-operator consistency,
and out-of-range fractions. LR NDVI is calculated after degrading reflectance
bands, not by averaging HR NDVI. Visual panels include common display scales,
zoomed details and four-band error maps. CSVs, source hashes, inputs and
convergence histories support reproduction.

See [benchmark instructions](../benchmarks/spectral_matching/README.md),
[procedural results](../benchmarks/spectral_matching/results/synthetic/REPORT.md),
and [real-HR results](../benchmarks/spectral_matching/results/real_hr/REPORT.md).
The repository currently ignores `benchmarks/`, so these local artifacts are
not guaranteed to accompany an installed package or a fresh checkout.

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

Downloads use a timeout, HTTP status checks and a temporary file that is renamed
only after success. Failed downloads clean up the temporary file and give a
contextual exception. Loading validates the checkpoint mapping/state dict,
and uses `weights_only=True`. The original loss-tensor filtering and strict
state-dict loading are retained.

Supported inference metadata can restore normalization, conditioning and
denoiser/sampling settings after validation, including schedule length and
epsilon parameterization. Declared schedule endpoints are checked against
stored beta buffers. Changing the configured schedule length requires enough
metadata to rebuild matching buffers before strict loading. Runtime overrides
should be applied after `load_pretrained` where checkpoint metadata may reset
those supported settings.

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

### Validation and scope

Inference, autoencoder, noise-schedule and DDIM/DPM regression tests were added
after the original release. New spectral tests verify the adjoint, constants,
block means, a dense pseudoinverse solution, agreement with independent SciPy
iterations, per-sample stopping, precision/autocast behavior, masks, invalid
settings and input immutability. Integration tests cover both samplers,
rectangular images, padded intermediate frames, config/argument precedence,
Lightning and uncertainty propagation. Default-call regression tests verify
three DPM-Solver++ evaluations, LR block-average consistency and equality with
the explicitly selected DPM/back-projection combination, both with the bundled
config and with selection fields omitted. A CUDA comparison test is included
and is skipped when CUDA is unavailable.

Validation of this integration on CPU: **120 production inference/diffusion
tests passed**, with only the CUDA test skipped. The Diffusers 0.38.0
dependency was available for these checks. An additional 36 comparisons against
the saved first real-HR image (nine scenarios, two operators, float32/float64)
matched the independent NumPy/SciPy outputs: maximum absolute discrepancy was
`4.983e-7` in float32 and `1.111e-15` in float64. This comparison uses the
benchmark's 400-iteration / `1e-7` settings, not a new quality experiment.
See [PyTorch parity results](../benchmarks/spectral_matching/results/real_hr/torch_validation.json).

A separate end-to-end CPU smoke run loaded the released
`opensr-ldsrs2_v1_0_0.ckpt` and ran DPM-Solver++ at its 3-step default with
back-projection on a real-image-derived 64×64 LR input. It returned a finite
256×256 four-band image with block-averaged LR RMSE `9.25e-9`.
See [production smoke result](../benchmarks/spectral_matching/results/real_hr/production_smoke.json).
This confirms the actual checkpoint/inference path, beyond stub-based
integration tests; it is not a new ground-truth quality assessment.

Run the relevant production tests with:

```bash
python -m pytest tests/test_inference.py tests/test_inference_utils.py \
  tests/test_spectral_correction.py tests/test_spectral_inference.py \
  tests/diffusion -q
```

Install the package dependencies to exercise DPM tests instead of skipping them.
Full training tests have additional dependencies and are covered by the
training workflow. Neither the correction benchmarks nor their passing tests
prove that generated subpixel spectra equal unknown physical ground truth on
new S2 scenes; that requires independent validation and an appropriate sensor
degradation model.
