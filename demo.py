import time
import requests
import numpy as np
import torch
import opensr_model
import safetensors.torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt


# Load the model --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create Model and load CKPT
model = opensr_model.SRLatentDiffusion(device=device) # 10m
model.load_pretrained("opensr_10m_v4_v6.ckpt") # 10m

# set model to eval mode
model = model.eval()

# test functionality of selected model --------------------------------------------
X = torch.rand(1,4,128,128)
sr = model(X)
assert sr.shape == (1,X.shape[1],512,512), "Model does not produce expected output shape!"
    
# test a pred --------------------------------------------------------------
sr = model(X,custom_steps=500)
sr = sr.cpu()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(rearrange(X[0,:3,:,:].cpu()*1.5, 'c h w -> h w c').numpy())
ax[0].set_title("LR")
ax[1].imshow(rearrange(sr[0,:3,:,:].cpu()*1.5, 'c h w -> h w c').numpy())
ax[1].set_title("SR")
plt.savefig("example_128.png")
plt.close()


