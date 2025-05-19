

# Latent Diffusion Super-Resolution - Sentinel 2 (LDSR-S2)
This repository contains the code of the paper [Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion](https://ieeexplore.ieee.org/abstract/document/10887321).  

**PLEASE NOTE**:
- This model is currently research-grade code, more user-friendly adaptations are planned for the future
- The 20m SR is only experimental and produces artifacts. An approach similar to [SEN2SR](https://github.com/ESAOpenSR/SEN2SR/tree/main) is in the works - stay tuned for that.

## Citation
If you use this model in your work, please cite  
```tex
@ARTICLE{10887321,
  author={Donike, Simon and Aybar, Cesar and Gómez-Chova, Luis and Kalaitzis, Freddie},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion}, 
  year={2025},
  volume={18},
  number={},
  pages={6940-6952},
  doi={10.1109/JSTARS.2025.3542220}}
```

## Install and Usage
```bash
pip install opensr-model
```
```python
# Load the model --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# set the type of model, 4x10m or 6x20m
model_type = "10m"
assert model_type in ["10m","20m"], "model_type must be either 10m or 20m"

if model_type == "10m": # if 10m, create according model and load ckpt
    model = opensr_model.SRLatentDiffusion(bands=model_type,device=device) # 10m
    model.load_pretrained("opensr_10m_v4_v2.ckpt") # 10m

if model_type == "20m": # if 20m, create according model and load ckpt
    model = opensr_model.SRLatentDiffusion(bands=model_type,device=device) # 20m
    model.load_pretrained("opensr_20m_v1.ckpt") # 20m

# set model to eval mode
model = model.eval()

# test functionality of selected model --------------------------------------------
if model_type == "10m":
    X = torch.rand(1,4,128,128)
if model_type == "20m":
    X = torch.rand(1,6,128,128)
sr = model(X)
```

## Weights and Checkpoints
The model should load automatically with the model.load_pretrained command. Alternatively, the checkpoints can be found on [HuggingFace](https://huggingface.co/simon-donike/RS-SR-LTDF/tree/main)

## Description
This package contains the latent-diffusion model to super-resolute 10 and 20m bands of Sentinel-2. This repository contains the bare model. It can be embedded in the "opensr-utils" package in order to be applied to Sentinel-2 Imagery. 
## Results Preview
Some example Sr scenes can be found as [super-resoluted tiffs](https://drive.google.com/drive/folders/1OBgYS6c8Kpe_JuGzWOQwOK6UYwhm-3Vh?usp=drive_link) on Doogle Drive. Scenes available:
- Buenos Aires, Argentina  
- Blue Mountains, Australia  
- Louisville, USA  
- Kutahya, Türkyie  
- Catalunya, Spain  

## Examples
Example on S2NAIP dataset
![example](resources/example.png)

Example on S2 image
![example2](resources/example2.png)




## Status
This is a work in progress and published explicitly as a research preview. This repository will leave the experimental stage with the publication of v1.0.0. 
[![PyPI Downloads](https://static.pepy.tech/badge/opensr-model)](https://pepy.tech/projects/opensr-model)
