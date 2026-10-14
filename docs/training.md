# Training OpenSR

OpenSR has two independent PyTorch Lightning training stages.

Packaged training presets and validation manifests live in
`opensr_model/configs/training_configs/`; architecture configs remain in
`opensr_model/configs/`. The optional external-validation implementation is
grouped under `opensr_model/training/validation/` (data preparation, metrics,
images, and the trainer callback). The rest of the training package is arranged
by responsibility:

```text
opensr_model/training/
├── __init__.py       # public imports; compatibility for older scripts
├── cli.py            # training entrypoints
├── data/             # TACO catalogs, paired datasets, data module
├── modules/          # autoencoder/diffusion stages, losses, metrics, optimizers
├── runtime/          # configuration, checkpoints, logging, provenance
└── validation/       # SEN2NEON evaluation and raw-S2 previews
```

The CLI commands and public imports (`from opensr_model.training import
DiffusionTrainingModule, AutoencoderTrainingModule, OpenSRDataModule`) remain
unchanged. Older module import paths remain aliases for existing scripts.

- `opensr-train-autoencoder` trains the exact `AutoencoderKL` component.
- `opensr-train-diffusion` freezes that autoencoder and trains the existing conditional denoising UNet.
- `opensr-train autoencoder|diffusion` is the equivalent unified command.

Each run writes complete Lightning checkpoints for optimizer/loop resume and separate native checkpoints for the existing strict inference loader.

## Public repository and private experiment work

The public repository provides reusable training functionality, general usage
guidance, portable example configurations, and tests for those public features.
Research campaigns and ablations are private local work by default.

Keep campaign supervisors, experiment scripts, run-specific settings and
manifests, research-only tests, experiment records, results, imagery, logs,
checkpoints, and release-candidate packages in Git-ignored local locations.
Use `scripts/`, `benchmarks/`, and `training_runs/` for this work; local
diagnostic tests and notes are also excluded by `.gitignore`. Copy public
example configurations into an ignored run directory before adding private
paths, account details, or experiment settings.

Do not commit or force-add these local files, or add Git-ignore exceptions to
publish them. General functions developed during an experiment may remain in
`opensr_model`, with public tests and documentation that do not depend on local
scripts, private datasets, or campaign settings. Document the reusable
capability in public guidance; keep the experiment design and operational
instructions in the ignored local record. Publication of research artifacts
requires a separate explicit decision. Authorized experiment-tracking uploads
do not authorize committing those artifacts to the public repository.

## Installation

```bash
pip install -e '.[train]'
```

The `train` extra installs the legacy `tacoreader.v1` dependencies, LPIPS, TensorBoard, Pillow, TorchMetrics, and TorchVision. W&B is optional:

```bash
pip install -e '.[train,wandb]'
```

LPIPS with the VGG backbone may download pretrained weights the first time it is enabled. For an offline or dependency-light diagnostic run, disable `training.loss.perceptual_weight` for stage 1 or `training.sharpness.decoded_perceptual_weight` for stage 2.

## Data contract

Training accepts completed TACO `.tortilla` and `.taco` files. The example below
uses paired four-band RGB-NIR data from SEN2NAIPv2.

`data.taco_path` may be a directory, a file, or a list of either. Directories are
scanned non-recursively. Each run saves its file list in `resolved_config.yaml`;
use that config when resuming to keep the same dataset.

Every TACO sample becomes this mapping:

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | `4 × H × W` | RGB-NIR HR target |
| `LR_image` | `4 × H/4 × W/4` | aligned native-resolution RGB-B8 condition |
| `valid_mask` | `1 × H × W` | finite, non-nodata pixels used by losses |
| `sample_id` | string | stable identifier used in image filenames |
| `clipped_fraction` | scalar | selected valid source values clipped to the checkpoint range |

### SEN2NAIPv2 cross-sensor

Each sample in `sen2naipv2-crosssensor.taco` has a direct four-band `lr` asset at `130 × 130` and `10 m`, plus a four-band `hr` asset at `520 × 520` and `2.5 m`. Both assets use the checkpoint channel order red, green, blue, NIR (B04, B03, B02, B08). Values are `uint16` reflectance digital numbers with nodata `65535`; the loader scales them by `1e-4`, masks nodata, and clips valid outliers to the configured range.

The asset schema is detected automatically, so the legacy asset-ID settings do not need to change. Repeated acquisitions of one spatial patch (IDs differing only in the final `YYYYMMDD`) remain in the same deterministic split. A 512-pixel training crop produces aligned `512 × 512` HR and `128 × 128` LR tensors; validation retains the complete `520 × 520`/`130 × 130` pair.

```yaml
data:
  taco_path: /path/to/SEN2NAIP
  synthetic_hr_nir_dir: null
  reflectance_scale: 0.0001
  val_fraction: 0.1
  split_seed: 42
  factor: 4
  train_patch_size: 512
  value_range: [0.0, 1.0]
```

Training uses aligned random crops and paired flips/90-degree rotations.
Validation uses complete image pairs without random augmentation.

## Stage 1: autoencoder

Copy and edit `opensr_model/configs/training_configs/train_autoencoder.yaml`, then run:

```bash
opensr-train-autoencoder --config /path/train_autoencoder.yaml
```

To initialize from the current released full checkpoint:

```bash
opensr-train-autoencoder \
  --config /path/train_autoencoder.yaml \
  --pretrained /path/opensr-ldsrs2_v1_0_0.ckpt
```

The autoencoder recipe combines reconstruction, perceptual, KL and adversarial
losses. Loss weights and optimizer settings are configured in the YAML.

For gradient accumulation, set `training.accumulate_grad_batches` and keep
`trainer.accumulate_grad_batches: 1`. Custom multi-GPU Trainers must use
`ddp_find_unused_parameters_true`; the CLI selects this automatically. Use
`max_epochs` to set the training duration.

When initialized from a full checkpoint, native exports are full inference
checkpoints (`best-inference.ckpt` / `last-inference.ckpt`). Standalone
autoencoder runs export `best-autoencoder.ckpt` / `last-autoencoder.ckpt` for
stage 2.

## Stage 2: diffusion

Copy and edit `opensr_model/configs/training_configs/train_diffusion.yaml`. Start from either a full checkpoint or a standalone stage-1 checkpoint:

```yaml
model:
  pretrained_checkpoint: null
  autoencoder_checkpoint: /runs/ae/checkpoints/native/best-autoencoder.ckpt
  allow_random_first_stage: false
```

Then run:

```bash
opensr-train-diffusion --config /path/train_diffusion.yaml
```

The autoencoder remains frozen during diffusion training. The template uses
P2-weighted noise prediction with optional detail and perceptual losses,
configured under `training.sharpness`. Validation and native inference exports
use EMA weights by default.

Loading a full checkpoint also restores its diffusion noise schedule. Keep
`apply_normalization: false` for the released v1 checkpoint.

## Checkpoints: resume versus inference

The output layout is:

```text
training_runs/<stage>/
├── resolved_config.yaml
├── checkpoints/
│   ├── resume/       # model + optimizers + schedulers + loop/callback state
│   └── native/       # exact OpenSR component/full state_dict contract
├── images/
│   └── epoch_0000/
└── logs/
```

Use `resume/last.ckpt` only with `resume_checkpoint` or `--resume`. A resume does not need the original pretrained initializer to remain available. Use an existing released/native checkpoint with `model.pretrained_checkpoint` or `--pretrained`. The CLI rejects released weight-only or malformed files as training resumes because they do not contain Lightning optimizer and loop state. Data shuffle/augmentation RNG streams restart when a process is relaunched, so resume preserves model/optimizer/loop state but is not promised to reproduce a bitwise-identical sample stream.

Native diffusion exports load through `SRLatentDiffusion.load_pretrained`.
They support single-device and replicated DDP training; sharded FSDP/DeepSpeed
exports are not supported.

Both Lightning resume files and native diffusion exports also store
`opensr_inference_config`: the live beta endpoints/timestep count,
parameterization, normalization/conditioning contract, and sampling
steps/eta/temperature. New inference loads adopt those defaults and validate the
metadata against the persisted schedule tensors. Older checkpoints without this
metadata retain the caller's configuration, preserving backward compatibility.

## Metrics and complete-image logging

Scalar logging includes total/component losses, MAE, MSE, RMSE, PSNR, SSIM, spectral angle (radians), output range violations, valid and input-clipped fractions, latent statistics, timestep statistics, and learning rate. Reconstruction metrics retain their four-channel compatibility names and are also reported separately for RGB and NIR. Diffusion sampling likewise reports overall, RGB, and NIR LR consistency.

Validation saves RGB/NIR grids and individual RGB images under `images/`,
with a shared contrast stretch for comparison. Images are also sent to
TensorBoard or W&B when enabled.

## Practical notes

- Use `512 × 512` HR / `128 × 128` LR training crops. Validation can retain
  the complete image pair.
- Start with the template batch sizes and validation batch size 1; adjust for
  your GPU memory.
- `apply_normalization: false` is the supported current v1 checkpoint contract. Do not mix `[0, 1]` and `[-1, 1]` training tensors.
- Relative paths in copied configuration files are resolved against that file. Relative overrides used with the packaged default templates are resolved against the launch directory.
- Set the template data section to the SEN2NAIPv2 example above and adjust paths before running. Set `logging.type: tensorboard` if W&B is not installed.

CLI values can be overridden without editing YAML:

```bash
opensr-train-autoencoder --config train.yaml \
  --set data.taco_path=/path/to/SEN2NAIP \
  --set training.loss.perceptual_weight=0 \
  --set training.loss.discriminator_factor=0
```

## SEN2NAIPv2 cross-sensor fine-tuning and SEN2NEON validation

Use `opensr_model/configs/training_configs/train_sen2naip.yaml` for released-v1 fine-tuning on
observed Sentinel-2 RGBN inputs and the direct four-band NAIP targets. This
preset explicitly disables P2 weighting and all auxiliary/detail/perceptual
training losses: the objective is vanilla epsilon MSE. It freezes the
checkpoint autoencoder, preserves the checkpoint noise schedule and
normalization, and uses AdamW at `1e-5` with a constant learning rate.
The generic diffusion template retains its existing defaults.

Install `pip install -e '.[train,wandb]'`. Obtain the cross-sensor archive from
[tacofoundation/SEN2NAIPv2](https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2)
and, optionally, the corrected canonical release of
[isp-uv-es/SEN2NEON](https://huggingface.co/datasets/isp-uv-es/SEN2NEON).
Point the training source specifically at the cross-sensor directory/file;
the UNet and histogram-matched variants are not mixed into this preset.
The published variants have different sizes and input provenance:

| Variant | Published pairs | LR input |
| --- | ---: | --- |
| Cross-sensor (this preset) | 8,000 | Observed Sentinel-2, within one day of NAIP; cloud-filtered |
| UNet | 62,242 | Synthetic Sentinel-2-like imagery from degraded, harmonized NAIP |
| Histogram matching | 61,282 | Synthetic Sentinel-2-like imagery with histogram harmonization |

These are variant counts, not additive counts of independent real Sentinel-2
observations. Filtering and spatially grouped splitting reduce the usable
training count further; inspect the preparation manifest for the actual totals.
For the audited release the cross-sensor SHA-256 is
`c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5`.
Dataset licenses and attribution remain those of the respective publishers;
the packaged SEN2NEON manifest contains identifiers/checksums, not raster data.

```bash
export SEN2NAIP_ROOT=/path/to/SEN2NAIP/v2/crosssensor
export SEN2NEON_ROOT=/path/to/SEN2NEON  # optional; omit if unavailable
export OPENSR_CHECKPOINT=/path/to/opensr-ldsrs2_v1_0_0.ckpt

# Audit and freeze the data before allocating training GPUs.
opensr-train-diffusion --config opensr_model/configs/training_configs/train_sen2naip.yaml --prepare-only

# One GPU: batch 4, accumulation 2, effective batch 8.
opensr-train-diffusion --config opensr_model/configs/training_configs/train_sen2naip.yaml

# Alternatively, two GPUs: batch 4 on each, accumulation 1, effective batch 8.
opensr-train-diffusion --config opensr_model/configs/training_configs/train_sen2naip.yaml \
  --set trainer.devices=2 --set trainer.strategy=ddp \
  --set trainer.accumulate_grad_batches=1
```

Choose a different `run_dir` for each independent training run. Both commands
above illustrate alternatives, not concurrent jobs sharing one output directory.
The preset runs 5,000 optimizer updates; change `trainer.max_steps` as needed.
`logging.entity`, `logging.project`, `logging.group`, `logging.id`, `logging.tags`,
and `logging.mode` are configurable. The fine-tuning preset defaults to
`logging.mode=offline` for local W&B records without uploads; use
`logging.type=csv` without W&B. LPIPS-VGG validation
needs its pretrained weights cached or downloaded on first use; it is not a
training loss. Turning SEN2NEON off does not require LPIPS for this preset.

### Preparation and optional availability

Preparation reads full pairs before cropping and rejects any selected RGBN
pixel that is masked, nonfinite, zero, or nodata. This positive-pixel policy is
stricter than the general loader's reflectance mask and is opt-in through
`data.strict_full_validity`. It also excludes misaligned pairs; coordinate
roundoff up to 1 mm is accepted. All dates/NAIP-provider observations sharing
one SEN2NAIP spatial grid ID stay in the same deterministic 90/10 group split.
Training uses random aligned 512/128 crops; internal validation uses fixed
center crops of those sizes. Direct RGBN assets require no synthetic sidecar.

`data_manifest.json` records source hashes, exclusions, eligible IDs, and both
splits. Source-stat-keyed validity caches under the run directory's parent
`.data_cache` avoid repeating raster audits for unchanged archives. Remove
that cache to force a new pixel audit. The run-specific manifest and source
identities must remain available on resume. Files changing in place invalidate
the preparation cache; training data must remain immutable during a run.

`training.validation.sen2neon.enabled` accepts `auto`, `true`, or `false`.
The root comes from the configuration, then `SEN2NEON_ROOT`, then
`data/SEN2NEON`. In auto mode, an absent root disables SEN2NEON and uses the
existing internal `val/sr_ema_psnr` checkpoint monitor. A present but incomplete
or invalid dataset is an error; `true` also requires the root to exist. The
resolved availability and selection identity are frozen for the run, so resume
cannot silently switch validation datasets or selection criteria.

The packaged 192-tile manifest uses the corrected `s2_l2a_10m` observations
and canonical `neon_2.5m_linearized` references, with one-based bands
`[4, 3, 2, 8]`, reflectance scale `1e-4`, and clipped `[0,1]` tensors. Every
selected full 256/1024 pair must pass checksum, band/grid and positive-pixel
checks. It includes both eligible developed-area tiles and all 40 clean sites;
these IDs were selected independently of model scores. Preparation compares
training footprints with all 192 validation footprints using conservative
WGS84 bounding boxes and excludes entire intersecting training groups.
Released-model pretraining overlap is unknown.

### Validation and selection protocol

The initial EMA checkpoint is evaluated before updates. Thereafter, the same
48 tiles are evaluated every 1,000 **optimizer** updates, independent of gradient
accumulation or epoch boundaries. A final partial interval is also evaluated.
The subset includes developed tiles, then all sites, then fills remaining
slots by deterministic site/land-cover round robin. Change
`training.validation.every_n_train_steps` or
`training.validation.sen2neon.subset_size` in a new run to alter this budget.
Internal validation continues to log denoising loss separately.

External inference uses EMA, DDIM 100, eta 0.95, temperature 1, and a seed from
SHA-256(sample ID) plus the configured validation seed. It evaluates full
1024-pixel images, using exact query-chunked autoencoder attention rather than
spatial tiling. Validation restores training mode, attention settings, and RNG
state afterward. Metrics use float32, without histogram matching, spectral
back-projection, target registration, or per-output contrast normalization.
Stored reconstruction arrays are float16; do not recompute reference metrics
from these rounded display/artifact outputs.

Reported metrics include RGB LPIPS-VGG; RGB/NIR/RGBN MAE, RMSE, PSNR, SSIM,
bias and LR consistency; spectral angle; and NDVI MAE. The scoring mask excludes
the outer five HR pixels, with another 32-pixel boundary neighborhood excluded
for LPIPS. NDVI requires both red-plus-NIR denominators to exceed 0.02.
Aggregate scores are means over tiles, accompanied by site and land-cover
breakdowns. Unequal per-GPU shards are gathered once after local evaluation;
there are no padded duplicate samples or per-sample distributed collectives.

A candidate qualifies only if mean NIR MAE **and** mean spectral angle are
at most 1.10 times their respective starting-EMA values on the same 48 tiles.
Among qualifying candidates, the lowest RGB LPIPS wins. The starting model is
a valid initial candidate; if no improvement qualifies, it remains best.
Nonfinite selection metrics stop evaluation without replacing the best model.
The tolerance is `training.validation.sen2neon.spectral_tolerance=0.10`.
Each check records eligibility and reasons for rejection.

Outputs include:

- `checkpoints/native/baseline-inference.ckpt`: starting EMA model.
- `checkpoints/native/best-inference.ckpt`: best qualifying EMA model.
- `checkpoints/native/last-inference.ckpt`: latest EMA model, even if rejected.
- `checkpoints/resume/{best,last}.ckpt`: complete optimizer/loop/callback state.
- `validation/sen2neon_step*/`: metrics for all 48 tiles, summaries, selection
  decisions, and full RGBN reconstructions/previews for eight fixed examples.
- `validation/final_report.json`: assessment on 192 tiles of baseline, selected,
  and last models, with paired changes and 2,000 whole-site bootstrap resamples.

The final assessment retains all 192 reconstructions per checkpoint, reuses
available identical predictions, and checks
spectral limits again on 192 tiles. It does not silently change the checkpoint
chosen using 48 tiles. These are iterative validation results, not a blinded
final test. Set `training.validation.sen2neon.final_assessment=false` only when
intentionally omitting the larger assessment, for example in a smoke test.

### Raw Sentinel-2 previews and W&B artifacts

Copy `opensr_model/configs/training_configs/raw_s2_examples.json` and replace its example path.
Each entry specifies an ID, GeoTIFF path, four one-based RGBN band indices, a
multiplicative reflectance scale, and optionally `zoom_center_hr: [row, column]`.
Relative paths resolve against the manifest directory. Inputs must be fully
valid and on a common 10 m grid. Use `[4,3,2,8]` for a canonical 12-band stack,
or `[1,2,3,4]` for an already ordered four-band RGBN stack. No band order or
radiometric offsets are guessed from an ambiguous file.

```bash
opensr-train-diffusion --config opensr_model/configs/training_configs/train_sen2naip.yaml \
  --set training.validation.raw_s2.manifest=/path/to/raw_s2.json
```

At each external check, metrics cover all selection tiles. By default the
fine-tuning preset retains full four-band reconstructions for just eight fixed
SEN2NEON examples (`training.validation.sen2neon.preview_count=8`) and every raw
preview. A fixed eight-example table
shows SEN2NEON LR/SR/reference in RGB and NIR, with aligned 64- and 256-pixel
zooms. Raw previews show LR/SR only and have no reference-based quality score.
Shared display stretches come from the reference, or the observed LR when
unpaired, and never from the prediction. LR pixels are expanded with nearest
neighbor for aligned comparisons. Raw inputs are padded as necessary and SR
is cropped back to the original full footprint. All previews are also saved
as local PNGs; quantitative RGBN arrays are stored in NPZ artifacts.
Offline W&B keeps these records local. Set
`training.validation.sen2neon.save_all_reconstructions=true` to retain all 48
arrays at routine checks. This changes storage and artifact volume, not scoring:
computing full-image metrics still requires sampling every evaluated tile.
The 1,000-update cadence reduces that inference cost. The final 192-tile
assessment always saves all reconstructions, regenerating any that were not
retained during selection with the same deterministic sampling protocol.

Each run also retains resolved configuration, source snapshots and hashes,
dependency versions, dataset manifests, checkpoint identity, seeds, and Git
revision. Existing source files are checked on resume: use a new run directory
for changed executable code or data/validation protocols. DDP assumes all ranks
share the run directory, as on a multi-GPU host.

```bash
# Resume the original run without needing its original initializer file.
opensr-train-diffusion --config training_runs/sen2naip/resolved_config.yaml \
  --resume training_runs/sen2naip/checkpoints/resume/last.ckpt

# Evaluate the selected resume checkpoint with the same frozen protocol.
opensr-train-diffusion --config training_runs/sen2naip/resolved_config.yaml \
  --resume training_runs/sen2naip/checkpoints/resume/best.ckpt --validate-only
```

Native checkpoints load through `SRLatentDiffusion.load_pretrained` exactly as
released checkpoints do. Their ordinary denoiser tensors already contain EMA
weights. To compare inference numerically with this validation protocol, use
the exported DDIM settings and disable optional decoded spectral correction
(`spectral_correction="none"`); validation scores uncorrected predictions.
Resume preserves model, optimizer, baseline, and selection state, but data
worker shuffle/augmentation streams are not guaranteed bitwise-identical after
relaunch.

### Named training sources and validation storage

`data.sources` adds named, spatially isolated TACO training pools. Each source
specifies `taco_path` and a JSON list at `train_ids`. The original source is
named `observed`; its validation split stays unchanged.
`data.sampling_proportions` specifies the proportion for each named source,
including `observed`. Proportions must sum to one and produce integer
per-GPU batch quotas. `data.sampling_seed` controls training draws
independently of `data.split_seed`. Omitting sources preserves the existing
single-source loader.

Under `training.validation.sen2neon`, `guardrail_baseline` can pin selection
to an explicit spectral reference instead of the branch's starting metrics.
`storage_dtype` controls reconstruction-array precision, while
`save_all_reconstructions` controls quantitative retention. `preview_count`
independently limits the visible scene cohort. Retaining more quantitative
arrays does not require expanding the imagery dashboard.
