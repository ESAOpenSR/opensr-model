# This script is an example how opensr-utils can be used in unison with opensr-model.

# Import and Instanciate SR Model
import opensr_model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = opensr_model.SRLatentDiffusionLightning(bands="10m",device=device) 
model.load_pretrained("opensr_10m_v4_v6.ckpt")

# perform SR with opensr-utils
from opensr_utils.main import windowed_SR_and_saving
path = "/data3/inf_data/S2A_MSIL2A_20241026T105131_N0511_R051_T30SYJ_20241026T150453.SAFE/"
sr_obj = windowed_SR_and_saving(folder_path=path, window_size=(128, 128), factor=4, keep_lr_stack=True)
sr_obj.start_super_resolution(band_selection="10m",model=model,forward_call="forward",custom_steps=100,overlap=20, eliminate_border_px=0) # start

