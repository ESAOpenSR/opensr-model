<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

# Latent Diffusion Super-Resolution - Sentinel 2 (LDSR-S2)
This repository contains the code of the paper [Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion](https://ieeexplore.ieee.org/abstract/document/10887321).  

  
<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/ldsr-s2_schema.png?raw=true" width="750"/>

**PLEASE NOTE**:
- This model is currently research-grade code, more user-friendly adaptations are planned for the future.
- This repository will leave the experimental stage with v1.0.0.


## Google Colab Demo - Interactive Notebook  
Run this model in Google Colab! The notebook now supports the unrestricted selection of your own coordinates and automatic download and SR of S2 imagery.  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1onza61SP5IyiUQM85PtyVLz6JovvQ0TN?usp=sharing)

## Citation
If you use this model in your work, please cite  
```tex
@ARTICLE{ldsrs2,
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

Minimal Example  
```python
import opensr_model # import pachage
model = opensr_model.SRLatentDiffusion(config, device=device) # create model
model.load_pretrained(config.ckpt_version) # load checkpoint
lr = torch.rand(1,4,128,128) # create data
sr = model.forward(lr, custom_steps=100) # run SR
```  
  
Run the 'demo.py' file to gain an understanding how the package works. It will SR and example tensor and save the according uncertainty map.
Output of demo.py file:
![example](resources/sr_example.png)  
![example](resources/uncertainty_map.png)

## Weights and Checkpoints
The model should load automatically with the model.load_pretrained command. Alternatively, the checkpoints can be found on [HuggingFace](https://huggingface.co/simon-donike/RS-SR-LTDF/tree/main)

## Description
This package contains the latent-diffusion model to super-resolute 10 and 20m bands of Sentinel-2. This repository contains the bare model. It can be embedded in the "opensr-utils" package in order to be applied to Sentinel-2 Imagery. 

## S2 Examples
Example on real S2 image
![example2](resources/example2.png)

Examples on S2NAIP training dataset
![example](resources/example.png)


## Status
This is a work in progress and published explicitly as a research preview. This repository will leave the experimental stage with the publication of v1.0.0.   
  
## Results Preview
Some example Sr scenes can be found as [super-resoluted tiffs](https://drive.google.com/drive/folders/1OBgYS6c8Kpe_JuGzWOQwOK6UYwhm-3Vh?usp=drive_link) on Doogle Drive. Scenes available:
- Buenos Aires, Argentina  
- Blue Mountains, Australia  
- Louisville, USA  
- Kutahya, Türkyie  
- Catalunya, Spain  
      
[![PyPI Downloads](https://static.pepy.tech/badge/opensr-model)](https://pepy.tech/projects/opensr-model)
