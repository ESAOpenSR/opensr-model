import torch
import opensr_model
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


# Load the model --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create Model and load CKPT
config = OmegaConf.load("opensr_model/configs/config_10m.yaml")
model = opensr_model.SRLatentDiffusion(config, device=device) # 10m
model.load_pretrained(config.ckpt_version) # 10m

# make sure  model is in eval mode
assert model.training == False, "Model has to be in eval mode."

# test functionality of selected model --------------------------------------------
X = torch.rand(1,4,128,128)
sr = model.forward(X.to(device), custom_steps=100)
    
# test a pred --------------------------------------------------------------
sr = sr.cpu()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(rearrange(X[0,:3,:,:].cpu()*1.5, 'c h w -> h w c').numpy())
ax[0].set_title("LR")
ax[1].imshow(rearrange(sr[0,:3,:,:].cpu()*1.5, 'c h w -> h w c').numpy())
ax[1].set_title("SR")
plt.savefig("example.png")
plt.close()


