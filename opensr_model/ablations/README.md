# Temperature × eta RGB ablation

This utility renders a controlled close-up grid showing how DDIM `eta` and
sampling `temperature` affect OpenSR predictions. Every grid cell starts from
the same random seed, uses the same RGB stretch, and shows the same center
detail crop. At `eta=0`, DDIM is deterministic and temperature has no effect.

```bash
python -m opensr_model.ablations.temperature_eta \
  --input /path/to/lr.tif \
  --checkpoint opensr-ldsrs2_v1_0_0.ckpt
```

Useful options are `--etas 0,0.5,0.95`, `--temperatures 0,0.5,1,1.5`,
`--steps 100`, `--seed 42`, and `--output-dir PATH`.
