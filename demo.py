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
    model.load_pretrained("opensr_10m_v4_v6.ckpt") # 10m

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

# Tun the Explainer -----------------------------------------------------------
mask = X[0, 0]* 0
mask[20:40, 80:100] = 1
mask.requires_grad = True
X.requires_grad = True

output = model.explainer(
    X=X, mask=mask,
    temperature=1.0,
    eta=1.0,
    custom_steps=100,
    steps_to_consider_for_attributions=list(range(100)),
    attribution_method="mean_grad",
    enable_checkpoint = True,
    verbose=False
)


for index in range(100):
    step = index
    hr_image = output[index]["latent"]
    grads = output[index]["attribution"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # first plot the low resolution image with the mask as a red square with opacity 0.5
    ax[0].imshow(X[0, 0:3].permute(1, 2, 0).detach().cpu().numpy()*3)
    ax[0].imshow(mask.detach().cpu().numpy(), alpha=0.2, cmap="Reds")
    ax[0].set_title("Low resolution image with mask")
    ax[0].axis("off")
    ax[1].imshow(hr_image[0, 0:3].permute(1, 2, 0).detach().cpu().numpy()*3)
    ax[1].set_title(f"High resolution image - {step}")
    ax[1].axis("off")
    ax[2].imshow(grads.detach().cpu().numpy())
    ax[2].set_title(f"Attribution map - {step}")
    ax[2].axis("off")
    plt.savefig("gif/demo_%03d.png" % index)
    plt.close()
