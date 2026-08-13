# Temperature × eta RGB ablation

This utility renders a controlled close-up grid showing how DDIM `eta` and
sampling `temperature` affect OpenSR predictions. Every grid cell starts from
the same random seed, uses the same RGB stretch, and shows the same center
detail crop. At `eta=0`, DDIM is deterministic and temperature has no effect.

```bash
/work/envs/opensr/bin/python -m opensr_model.ablations.temperature_eta \
  --input runs/opensr_test_20260312T123948411350Z/patches/patch_000001/inputs/lr.tif \
  --checkpoint opensr-ldsrs2_v1_0_0.ckpt
```

Useful options are `--etas 0,0.5,0.95`, `--temperatures 0,0.5,1,1.5`,
`--steps 100`, `--seed 42`, and `--output-dir PATH`.
