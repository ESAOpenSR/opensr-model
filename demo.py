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
assert sr.shape == (1,X.shape[1],512,512), "Model does not produce expected output shape!"

# Download RGBNIR test image --------------------------------------------------------------
file = "https://huggingface.co/datasets/jfloresf/demo/resolve/main/lr_000008.safetensors"
response = requests.get(file)
with open("demo.safetensors", "wb") as f:
    f.write(response.content)
    
X = safetensors.torch.load_file("demo.safetensors")["lr_data"]
X = X.to(device)*1

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


