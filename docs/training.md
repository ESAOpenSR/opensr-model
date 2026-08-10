# Training OpenSR

OpenSR now has two independent PyTorch Lightning training stages. The model architectures and native state-dict layout are unchanged, and the inference API remains backward compatible.

- `opensr-train-autoencoder` trains the exact `AutoencoderKL` component.
- `opensr-train-diffusion` freezes that autoencoder and trains the existing conditional denoising UNet.
- `opensr-train autoencoder|diffusion` is the equivalent unified command.

Each run writes complete Lightning checkpoints for optimizer/loop resume and separate native checkpoints for the existing strict inference loader.

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

Training accepts completed TACO `.tortilla` and `.taco` files. The worldwide
corpus additionally requires the generated per-sample HR-NIR NPZ sidecars
described below. CSV manifests, loose raster inputs, generic array/tensor
datasets, and on-the-fly degradations are deliberately not part of this
pipeline. Both supported TACO schemas use the `tacoreader.v1` adapter for their
container layout.

`data.taco_path` may be a directory, a completed `.tortilla`/`.taco` file, or a list containing either. A directory is scanned non-recursively for visible files with those exact suffixes; hidden upload temporaries are excluded. The part list is snapshotted when the data module is constructed and its explicit filenames are written to `resolved_config.yaml`. Restart the training process after another upload completes if that new part should participate in a new run; use the run's resolved config when resuming to retain the exact earlier snapshot. The same discovery logic handles a directory containing one file or dozens of parts and never changes the active dataset halfway through a process.

Every TACO sample becomes this mapping:

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | `4 × H × W` | RGB HR target plus direct or generated HR NIR |
| `LR_image` | `4 × H/4 × W/4` | aligned native-resolution RGB-B8 condition |
| `valid_mask` | `1 × H × W` | finite, non-nodata pixels used by losses |
| `sample_id` | string | stable identifier used in image filenames |
| `clipped_fraction` | scalar | selected valid source values clipped to the checkpoint range |

### SEN2NAIPv2 cross-sensor

The extracted `/work/data/SEN2NAIP/sen2naipv2-crosssensor.taco` file contains 8,000 samples. Each sample has a direct four-band `lr` asset at `130 × 130` and `10 m`, plus a four-band `hr` asset at `520 × 520` and `2.5 m`. Both assets use the checkpoint channel order red, green, blue, NIR (B04, B03, B02, B08). Values are `uint16` reflectance digital numbers with nodata `65535`; the loader scales them by `1e-4`, masks nodata, and clips valid outliers to the configured range. Unlike the worldwide schema below, HR NIR is read directly and is not interpolated from LR.

The asset schema is detected automatically, so the legacy asset-ID settings do not need to change. Repeated acquisitions of one spatial patch (IDs differing only in the final `YYYYMMDD`) remain in the same deterministic split. A 512-pixel training crop produces aligned `512 × 512` HR and `128 × 128` LR tensors; validation retains the complete `520 × 520`/`130 × 130` pair.

```yaml
data:
  taco_path: /work/data/SEN2NAIP
  reflectance_scale: 0.0001
  val_fraction: 0.1
  split_seed: 42
  factor: 4
  train_patch_size: 512
  value_range: [0.0, 1.0]
```

### Worldwide four-asset corpus

The uploaded samples contain `lrharm` (128×128 RGB), `hrharm` (512×512 RGB), and `lr` (128×128, 12-band Sentinel-2) assets. Bands 1, 2, and 3 of the harmonized assets are already RGB; B8 is band 8 of the raw LR asset. All band indices in YAML are one-based.

The native `uint16` reflectance digital numbers are converted to `float32` with `x / 10000`, then valid values are clipped to the `[0, 1]` radiometric contract expected by the current checkpoints. Clipping is measured per sample and logged as `train/input_clipped_fraction` and `val/input_clipped_fraction`; it is rare in the current upload but is not silently confused with nodata. Raster masks are intersected across the selected channels; the dataset's nodata value `65535`, NaN, and infinity never contribute to losses or metrics. Zero is valid reflectance and is not masked. This explicit normalization is configured with `reflectance_scale: 0.0001`; it is not inferred from sample statistics.

The `hrharm` asset contains RGB only. Its fourth training channel now comes from
`synth_nirs/<sample_id>.npz`, generated from HR RGB by the SharpNIR model. Every
sidecar must contain exactly `nir: float16[1,512,512]` with finite values already
normalized to `[0,1]`; it is not scaled by `1e-4` again. Native LR B8 remains the
fourth conditioning channel at 128×128.

For each sample, the loader bilinearly enlarges native LR B8 to 512×512 with
`align_corners=False`, then uses `skimage.exposure.match_histograms` to match the
synthetic NIR's valid-pixel histogram to that enlarged reference. Only pixels in
the joint HR/LR validity mask participate, so nodata does not affect either
empirical distribution. Histogram matching changes radiometry monotonically
while retaining the synthetic image's spatial detail ordering.

Training refuses to fall back to interpolated LR B8. Before fitting, it requires
`synth_nirs/inference_manifest.json` to have status `complete`, match the exact
paths and sizes of the snapshotted TACO parts, account for every catalog sample,
and declare the expected output normalization. It also checks that every TACO
sample ID has a visible NPZ sidecar. Each file's key, dtype, shape, finiteness,
and range are then checked lazily as it is read. This prevents a partially
generated or stale sidecar collection from silently entering a run.

Synthetic HR NIR supplies spatially detailed pseudo-supervision, but it is still
model-generated rather than independently observed 2.5 m NIR. NIR validation
metrics therefore measure agreement with SharpNIR targets, not physical sensor
ground truth.

Use this data section for the current upload:

```yaml
data:
  taco_path: /work/data/SISR_worldwide
  lr_asset_id: lrharm
  hr_asset_id: hrharm
  raw_lr_asset_id: lr
  rgb_bands: [1, 2, 3]
  nir_band: 8
  hr_nir_strategy: synthetic_sidecar
  synthetic_hr_nir_dir: /work/data/SISR_worldwide/synth_nirs
  reflectance_scale: 0.0001
  val_fraction: 0.1
  split_seed: 42
  factor: 4
  train_patch_size: 512
  value_range: [0.0, 1.0]
```

The train/validation assignment is a stable hash of each spatial base ID and `split_seed`, so it does not depend on file order. Provider suffixes (`_bing`, `_google`, and `_esri`) are removed before hashing, which keeps every view of one location in the same split and prevents spatial leakage. Training uses aligned random crops and paired flips/90-degree rotations. Validation always reads the complete 512×512 HR sample and corresponding 128×128 LR sample, without random augmentation.

## Stage 1: autoencoder

Copy and edit `opensr_model/configs/train_autoencoder.yaml`, then run:

```bash
opensr-train-autoencoder --config /path/train_autoencoder.yaml
```

To initialize from the current released full checkpoint:

```bash
opensr-train-autoencoder \
  --config /path/train_autoencoder.yaml \
  --pretrained /path/opensr-ldsrs2_v1_0_0.ckpt
```

There is one shipped autoencoder recipe, reconstructed from checkpoint-era
`latent-diffusion` commit `e7ea72a` rather than inferred from the paper:

- stochastic posterior samples in both training and validation;
- one uniformly sampled three-of-four-band view per optimizer forward for summed
  L1+LPIPS image loss;
- the summed posterior KL term at weight `1e-4`;
- the original three-channel, BatchNorm `NLayerDiscriminator` with hinge loss;
- adaptive GAN weight from the decoder's last-layer gradient-norm ratio, scaled by
  `0.5`, active from the first update;
- Adam at `1e-4` with betas `(0.5, 0.9)`, plus the original plateau schedule.

The historical scalar `logvar` remains fixed at zero because the old optimizer
did not include it. Its unusual summed loss scaling and independently resampled
generator/discriminator forwards are preserved deliberately. The only
data-era change is nodata safety: invalid predictions are replaced by detached
targets before L1, LPIPS, and PatchGAN evaluation. Reflectance normalization is
performed once by the TACO loader.

The GAN uses Lightning manual optimization. The shipped dual-RTX-3090 setting
keeps `trainer.accumulate_grad_batches: 1` and performs accumulation inside the
module with `training.accumulate_grad_batches: 2`. At batch 1 per GPU on two
GPUs this preserves the required 512-to-128 spatial contract and gives effective
global batch 4. On multi-process runs the CLI selects
`ddp_find_unused_parameters_true`; custom Trainers must do the same. Use epoch
limits because Lightning increments `global_step` once for each generator and
discriminator optimizer step.
Reconstructions remain unclamped for every loss; clamping is limited to metrics
and plots.

With a released full checkpoint in `model.pretrained_checkpoint`, the default `checkpoint.native.autoencoder_base_checkpoint: auto` detects the full contract and merges the 204 trained first-stage tensors into the unchanged schedule, UNet, and EMA tensors. The resulting `best-inference.ckpt` and `last-inference.ckpt` load strictly through `SRLatentDiffusion.load_pretrained`. Standalone AE initialization is detected and remains a standalone `best-autoencoder.ckpt`/`last-autoencoder.ckpt` export for stage 2.

## Stage 2: diffusion

Start from either a full checkpoint or a standalone stage-1 checkpoint:

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

The first-stage model is forced to eval mode and all of its parameters remain frozen. HR is encoded to the target latent. LR is bilinearly enlarged by four with `align_corners=False`, then encoded without the target latent scale factor, matching current inference. Other parameterizations and `apply_normalization: true` are rejected because the current public inference wrapper would interpret them differently. EMA updates happen only after real optimizer steps, including with gradient accumulation.

The single diffusion recipe is deliberately biased toward perceptual quality:

- [P2 perception-prioritized weighting](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html) is applied to the epsilon-MSE objective;
- a low-timestep latent Laplacian loss rewards recoverable high-frequency detail;
- every eighth batch may decode one low-noise RGB prediction for L1, Laplacian,
  and LPIPS supervision;
- validation samples with EMA weights using 100 DDIM steps and `eta=0.95`;
- native exports materialize EMA shadows into the ordinary UNet keys consumed by
  the unchanged public inference path.

The unweighted epsilon MSE and every auxiliary component are logged separately.
The decoded losses are bounded by timestep, cadence, batch count, and RGB-only
selection because differentiating through the frozen full-resolution decoder is
expensive. These settings intentionally trade some pixel fidelity for sharper,
more plausible texture; hallucinated detail is therefore a model-selection risk,
not an unexpected failure mode.

Full released checkpoint loading includes all diffusion schedule buffers. This matters because the released file contains an actual beta range of `0.0015..0.0155`; importing only UNet weights would make the training schedule depend on a separate YAML file instead of the checkpoint itself.

Validation uses fixed local posterior, timestep, and noise generators and logs both
P2-weighted and ordinary noise-prediction loss plus reconstructed-`x0` latent MSE
over the full validation set. A fixed small subset receives reproducible seeded
stochastic DDIM sampling, decoded SR metrics, spectral-angle distance, and
downsample-to-LR consistency.

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

Diffusion native checkpoints contain the bare 830-key `LatentDiffusion.state_dict()` under top-level `state_dict`, with no Lightning wrapper prefix. They retain all schedule, autoencoder, ordinary UNet, and EMA keys. By default only the values in the ordinary UNet keys are replaced by their EMA shadows, because those are the keys public inference executes; set `checkpoint.native.weight_source: raw` only for an intentional diagnostic export. Loading remains strict and prefix remapping is selected by exact target keys and tensor shapes, never by key order. Native export supports ordinary single-device and replicated DDP training; sharded FSDP/DeepSpeed strategies are intentionally rejected because they require a collective full-state gather.

Both Lightning resume files and native diffusion exports also store
`opensr_inference_config`: the live beta endpoints/timestep count,
parameterization, normalization/conditioning contract, and sampling
steps/eta/temperature. New inference loads adopt those defaults and validate the
metadata against the persisted schedule tensors. Older checkpoints without this
metadata retain the caller's configuration, preserving backward compatibility.

## Metrics and complete-image logging

Scalar logging includes total/component losses, MAE, MSE, RMSE, PSNR, SSIM, spectral angle (radians), output range violations, valid and input-clipped fractions, latent statistics, timestep statistics, and learning rate. Reconstruction metrics retain their four-channel compatibility names and are also reported separately for RGB and NIR. Worldwide NIR values measure fidelity to synthetic SharpNIR pseudo-targets, not independently observed HR NIR. Diffusion sampling likewise reports overall, RGB, and NIR LR consistency.

Every configured validation epoch saves the same number of complete validation tensors—without detail crops—as RGB grids, NIR grids, and individual RGB PNGs. Autoencoder panels contain target, reconstruction, and absolute error. Diffusion panels contain native LR, enlarged LR, conditioning reconstruction, target, AE reconstruction, sampled SR, and absolute error. PNGs are always local; TensorBoard and W&B figures are added when those loggers are selected.

## Practical notes

- The spatial contract is fixed: training and validation use `512 × 512` HR and
  `128 × 128` LR tensors. Do not lower `data.train_patch_size` to solve an OOM.
- The shipped settings were soak-tested for 100 real TACO batches with PyTorch
  2.11 and two 24 GiB RTX 3090s in `16-mixed` precision. Autoencoder batch 1 per
  GPU with module accumulation 2 peaked at 20.15 GiB reserved per GPU and took
  about 0.74 s per microbatch. Diffusion batch 4 per GPU without accumulation
  peaked at 14.46 GiB and took about 0.55 s per batch; forcing decoded
  L1/Laplacian/LPIPS on every batch still remained around 14.35 GiB.
- Full-resolution autoencoder batch 2 per GPU is not viable with the compatible
  vanilla attention implementation: it OOMed after allocating 22.6 GiB. Keep
  validation batch size at 1 as shipped.
- Re-run the same real-model/TACO probe with
  `python benchmarks/benchmark_training_memory.py autoencoder|diffusion ...`;
  use `--help` for the checkpoint, device, batch, precision, and optional
  worst-case decoded-loss controls.
- TACO spatial base IDs drive the deterministic, provider-grouped hash split; validation reads the full sample and DDIM uses a fixed local generator.
- `apply_normalization: false` is the supported current v1 checkpoint contract. Do not mix `[0, 1]` and `[-1, 1]` training tensors.
- Relative paths in copied configuration files are resolved against that file. Relative overrides used with the packaged default templates are resolved against the launch directory.
- The shipped templates point to `/work/data/SISR_worldwide`. Override `data.taco_path` on any machine where the TACO parts live elsewhere.

CLI values can be overridden without editing YAML:

```bash
opensr-train-autoencoder --config train.yaml \
  --set data.taco_path=/data/SISR_worldwide \
  --set training.loss.perceptual_weight=0 \
  --set training.loss.discriminator_factor=0
```
