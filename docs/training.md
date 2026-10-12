# Training OpenSR

OpenSR has two independent PyTorch Lightning training stages.

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

Copy and edit `opensr_model/configs/train_diffusion.yaml`. Start from either a full checkpoint or a standalone stage-1 checkpoint:

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
