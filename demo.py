import torch
import opensr_model
from omegaconf import OmegaConf

# -------------------------------------------------------------------------------------
# 0.0 Create Model
device = "cuda" if torch.cuda.is_available() else "cpu" #try to run on GPU
config = OmegaConf.load("opensr_model/configs/config_10m.yaml") # load config
model = opensr_model.SRLatentDiffusion(config, device=device) # create model
model.load_pretrained(config.ckpt_version) # load checkpoint
assert model.training == False, "Model has to be in eval mode."

# -------------------------------------------------------------------------------------
# 0.1 - Get Data Example
from opensr_model.utils import download_from_HF
lr = download_from_HF(file_name="example_lr.pt")
lr = (lr/10_000).to(torch.float32).to(device)

# -------------------------------------------------------------------------------------
# 1. Run Super-Resolution
sr = model.forward(lr, custom_steps=100)

# -------------------------------------------------------------------------------------
# 2. Run Encertainty Map Generation
uncertainty_map = model.uncertainty_map(lr,n_variations=25,custom_steps=100) # create uncertainty map

# -------------------------------------------------------------------------------------
# 3 Plot Examples
from opensr_model.utils import plot_example,plot_uncertainty
plot_example(lr,sr,out_file="example.png")
plot_uncertainty(uncertainty_map,out_file="uncertainty_map.png",normalize=True)

