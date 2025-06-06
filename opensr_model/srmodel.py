import pathlib
from typing import Union
import requests
import torch
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from skimage.exposure import match_histograms
import torch.utils.checkpoint as checkpoint
from opensr_model.diffusion.utils import DDIMSampler
from tqdm import tqdm
import numpy as np
from typing import Literal
import shutil
import random
import numpy as np
from einops import rearrange
from opensr_model.utils import suppress_stdout
from opensr_model.utils import assert_tensor_validity
from opensr_model.utils import revert_padding
from opensr_model.utils import create_no_data_mask
from opensr_model.utils import apply_no_data_mask



class SRLatentDiffusion(torch.nn.Module):
    def __init__(self,config, device: Union[str, torch.device] = "cpu"):
        super().__init__()

        # Set up the model
        self.config = config        
        self.model = LatentDiffusion(
            config.first_stage_config,
            config.cond_stage_config,
            timesteps=config.denoiser_settings.timesteps,
            unet_config=config.cond_stage_config,
            linear_start=config.denoiser_settings.linear_start,
            linear_end=config.denoiser_settings.linear_end,
            concat_mode=config.other.concat_mode,
            cond_stage_trainable=config.other.cond_stage_trainable,
            first_stage_key=config.other.first_stage_key,
            cond_stage_key=config.other.cond_stage_key,
        )


        # Set up the model for inference
        self.set_normalization() # decide wether to use norm
        self.device = device # set self device
        self.model.device = device # set model device as selected
        self.model = self.model.to(device) # move model to device
        self.model.eval() # set model state
        self = self.eval() # set main model state
        self._X = None # placeholder for LR image
        self.encode_conditioning = config.encode_conditioning # encode LR images before dif?

    def set_normalization(self):
        if self.config.apply_normalization==True:
            from opensr_model.utils import linear_transform_4b
            self.linear_transform = linear_transform_4b
        else:
            from opensr_model.utils import linear_transform_placeholder
            self.linear_transform = linear_transform_placeholder
            print("Normalization disabled.")
        
    def _tensor_encode(self,X: torch.Tensor):
        # set copy to model
        self._X = X.clone()
        # normalize image
        X_enc = self.linear_transform(X, stage="norm")
        # encode LR images
        if self.encode_conditioning==True :
            # try to upsample->encode conditioning
            X_int = torch.nn.functional.interpolate(X, size=(X.shape[-1]*4,X.shape[-1]*4), mode='bilinear', align_corners=False)
            # encode conditioning
            X_enc = self.model.first_stage_model.encode(X_int).sample()
        # move to same device as the model
        X_enc = X_enc.to(self.device)
        return X_enc

    def _tensor_decode(self, X_enc: torch.Tensor, spe_cor: bool = True):       
        # Decode
        X_dec = self.model.decode_first_stage(X_enc)
        X_dec = self.linear_transform(X_dec, stage="denorm")
        # Apply spectral correction
        if spe_cor:
            for i in range(X_dec.shape[1]):
                X_dec[:, i] = self.hq_histogram_matching(X_dec[:, i], self._X[:, i])
        # If the value is negative, set it to 0
        X_dec[X_dec < 0] = 0    
        return X_dec
    
    def _prepare_model(
        self,
        X: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        verbose: bool = False 
    ):
        # Create the DDIM sampler
        ddim = DDIMSampler(self.model)
        
        # make schedule to compute alphas and sigmas
        ddim.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=verbose)
        
        # Create the HR latent image
        latent = torch.randn(X.shape, device=X.device)
                
        # Create the vector with the timesteps
        timesteps = ddim.ddim_timesteps
        time_range = np.flip(timesteps)
        
        return ddim, latent, time_range

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        eta: float = 1.0,
        custom_steps: int = 100,
        temperature: float = 1.0,
        histogram_matching: bool = True,
        save_iterations: bool = False,
        verbose: bool = False
    ):
        """Obtain the super resolution of the given image.

        Args:
            X (torch.Tensor): If a Sentinel-2 L2A image with reflectance values
                in the range [0, 1] and shape CxWxH, the super resolution of the image
                is returned. If a batch of images with shape BxCxWxH is given, a batch
                of super resolved images is returned.
            custom_steps (int, optional): Number of steps to run the denoiser. Defaults
                to 100.
            temperature (float, optional): Temperature to use in the denoiser.
                Defaults to 1.0. The higher the temperature, the more stochastic
                the denoiser is (random noise gets multiplied by this).
            spectral_correction (bool, optional): Apply spectral correction to the SR
                image, using the LR image as reference. Defaults to True.

        Returns:
            torch.Tensor: The super resolved image or batch of images with a shape of
                Cx(Wx4)x(Hx4) or BxCx(Wx4)x(Hx4).
        """
        
        # Assert shape, size, dimensionality
        X,padding = assert_tensor_validity(X)
        
        # create no_data_mask
        no_data_mask = create_no_data_mask(X, target_size= X.shape[-1]*4)

        # Normalize the image
        X = X.clone()
        Xnorm = self._tensor_encode(X)
        
        # ddim, latent and time_range
        ddim, latent, time_range = self._prepare_model(
            X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=verbose
        )
        iterator = tqdm(time_range, desc="DDIM Sampler", total=custom_steps,disable=True)

        # Iterate over the timesteps
        if save_iterations:
            save_iters = []
            
        for i, step in enumerate(iterator):
            outs = ddim.p_sample_ddim(
                x=latent,
                c=Xnorm,
                t=step,
                index=custom_steps - i - 1,
                use_original_steps=False,
                temperature=temperature
            )
            latent, _ = outs
            
            if save_iterations:
                save_iters.append(
                    self._tensor_decode(latent, spe_cor=histogram_matching)
                )
        
        if save_iterations:
            return save_iters
        
        sr = self._tensor_decode(latent, spe_cor=histogram_matching) # decode the latent image
        sr = apply_no_data_mask(sr, no_data_mask) # apply no data mask as in LR image
        sr = revert_padding(sr,padding) # remove padding from the SR image if there was any
        return sr


    def hq_histogram_matching(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies histogram matching to align the color distribution of image1 to image2.

        This function adjusts the pixel intensity distribution of `image1` (typically the
        low-resolution or degraded image) to match that of `image2` (typically the 
        high-resolution or reference image). The operation is done per channel and 
        assumes both images are in (C, H, W) format.

        Args:
            image1 (torch.Tensor): The source image whose histogram will be modified (C, H, W).
            image2 (torch.Tensor): The reference image whose histogram will be matched (C, H, W).

        Returns:
            torch.Tensor: A new tensor with the same shape as `image1`, but with pixel 
                        intensities adjusted to match the histogram of `image2`.

        Raises:
            ValueError: If input tensors are not 2D or 3D.
        """

        # Go to numpy
        np_image1 = image1.detach().cpu().numpy()
        np_image2 = image2.detach().cpu().numpy()

        if np_image1.ndim == 3:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
        elif np_image1.ndim == 2:
            np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
        else:
            raise ValueError("The input image must have 2 or 3 dimensions.")

        # Go back to torch
        image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

        return image1_hat

    def load_pretrained(self, weights_file: str):
        """
        Loads pretrained model weights from a local file or downloads them from Hugging Face if not present.

        If the specified `weights_file` does not exist locally, it is automatically downloaded from the 
        Hugging Face model hub under `simon-donike/RS-SR-LTDF`. A progress bar is shown during download.

        After loading, the method removes any perceptual loss-related weights from the state dict and 
        loads the remaining weights into the model.

        Args:
            weights_file (str): Path to the local weights file. If the file is not found, it will be downloaded 
                                using this name from the Hugging Face repository.

        Raises:
            RuntimeError: If the weights cannot be loaded or parsed correctly.

        Example:
            self.load_pretrained("model_weights.ckpt")
        """

        # download pretrained model
        # create download link based on input 
        hf_model = str("https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/"+str(weights_file))
        
        # Total size in bytes.
        if not pathlib.Path(weights_file).exists():
            print("Downloading pretrained weights from: ", hf_model)
            response = requests.get(hf_model, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Open the file to write as binary - write bytes to a file
            with open(weights_file, "wb") as f:
                # Setup the progress bar
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=weights_file) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        f.write(data)

        weights = torch.load(weights_file, map_location=self.device)["state_dict"]

        # Remote perceptual tensors from weights
        for key in list(weights.keys()):
            if "loss" in key:
                del weights[key]

        self.model.load_state_dict(weights, strict=True)
        print("Loaded pretrained weights from: ", weights_file)
        
        
    def uncertainty_map(self, x, n_variations=15, custom_steps=100):
        """
        Estimates uncertainty maps for each sample in the input batch using repeated stochastic forward passes.

        For each input sample, the method generates multiple super-resolved outputs by varying the random seed.
        It then computes the per-pixel standard deviation across these outputs as a proxy for uncertainty.
        The returned uncertainty map represents the average width of the confidence interval per pixel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is batch size.
            n_variations (int): Number of stochastic forward passes per input sample.
            custom_steps (int): Custom inference steps passed to the forward method.

        Returns:
            torch.Tensor: Uncertainty maps of shape (B, 1, H, W), where each value indicates pixel-wise uncertainty.
        """
        assert n_variations>3, "n_variations must be greater than 3 to compute uncertainty."
        
        
        batch_size = x.shape[0]
        rand_seed_list = random.sample(range(1, 9999), n_variations)

        all_variations = []
        for b in range(batch_size):
            variations = []
            x_b = x[b].unsqueeze(0)  # shape (1, 4, 512, 512)
            for seed in rand_seed_list:
                with suppress_stdout():
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    #pytorch_lightning.utilities.seed.seed_everything(seed=seed, workers=True)

                sr = self.forward(x_b, custom_steps=custom_steps)  # shape (1, C, H, W)
                variations.append(sr.detach().cpu())

            variations = torch.stack(variations)  # (n_variations, 1, C, H, W)
            srs_mean = variations.mean(dim=0)
            srs_stdev = variations.std(dim=0)
            interval_size = (srs_stdev * 2).mean(dim=1)  # mean over channels

            all_variations.append(interval_size)  # each is (1, H, W)

        result = torch.stack(all_variations)  # (B, 1, H, W)
        return result
    
    

    def _attribution_methods(
        self,
        X: torch.Tensor,
        grads: torch.Tensor,
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"            
                        ],
                    ):
        """
        DEPRECIATED; SUBJECT TO REMOVAL
        """
        if attribution_method == "grad_x_input":
            return torch.norm(grads * X, dim=(0, 1))
        elif attribution_method == "max_grad":
            return grads.abs().max(dim=0).max(dim=0)
        elif attribution_method == "mean_grad":
            return grads.abs().mean(dim=0).mean(dim=0)
        elif attribution_method == "min_grad":
            return grads.abs().min(dim=0).min(dim=0)
        else:
            raise ValueError(
                "The attribution method must be one of: grad_x_input, max_grad, mean_grad, min_grad"
            )
    
    def explainer(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        eta: float = 1.0,
        temperature: float = 1.0,
        custom_steps: int = 100,
        steps_to_consider_for_attributions: list = list(range(100)),
        attribution_method: Literal[
            "grad_x_input", "max_grad", "mean_grad", "min_grad"
        ] = "grad_x_input",      
        verbose: bool = False,
        enable_checkpoint = True,
        histogram_matching=True        
            ):  
        """
        DEPRECIATED; SUBJECT TO REMOVAL
        """
        # Normalize and encode the LR image
        X = X.clone()
        Xnorm = self._tensor_encode(X)
        
        # ddim, latent and time_range
        ddim, latent, time_range = self._prepare_model(
            X=Xnorm, eta=eta, custom_steps=custom_steps, verbose=verbose
        )
                    
        # Iterate over the timesteps
        container = []
        iterator = tqdm(time_range, desc="DDIM Sampler", total=custom_steps,disable=True)
        for i, step in enumerate(iterator):
            
            # Activate or deactivate gradient tracking
            if i in steps_to_consider_for_attributions:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            
            # Compute the latent image
            if enable_checkpoint:
                outs = checkpoint.checkpoint(
                    ddim.p_sample_ddim,
                    latent,
                    Xnorm,
                    step,
                    custom_steps - i - 1,
                    temperature,
                    use_reentrant=False,
                )
            else:                
                outs = ddim.p_sample_ddim(
                    x=latent,
                    c=Xnorm,
                    t=step,
                    index=custom_steps - i - 1,
                    temperature=temperature
                )
            latent, _ = outs
            
            
            if i not in steps_to_consider_for_attributions:
                continue
            
            # Apply the mask
            output_graph = (latent*mask).mean()
            
            # Compute the gradients
            grads = torch.autograd.grad(output_graph, Xnorm, retain_graph=True)[0]
            
            # Compute the attribution and save it
            with torch.no_grad():
                to_save = {
                    "latent": self._tensor_decode(latent, spe_cor=histogram_matching),
                    "attribution": self._attribution_methods(
                        Xnorm, grads, attribution_method
                    )
                }
            container.append(to_save)
        
        return container



# -----------------------------------------------------------------------------
# Logic to create PyTorch Lightning Model from dif model
# Logic to handle outputs from PL model and save them
# -----------------------------------------------------------------------------

import torch
import pytorch_lightning
from pytorch_lightning import LightningModule

class SRLatentDiffusionLightning(LightningModule):
    """
    This Pytorch Lightning Class wraps around the torch model to
    aid in distrubuted GPU processing and optimized dataloaders
    provided by PL. ToDo: implement demo showcase
    """
    def __init__(self,config, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.model = SRLatentDiffusion(config,device=device)
        self.model = self.model.eval()

    @torch.no_grad()
    def forward(self, x,**kwargs):
        #print("Dont call 'forward' on the PL model, instead use 'predict'")
        return self.model(x)
    
    def load_pretrained(self, weights_file: str):
        self.model.load_pretrained(weights_file)
        print("PL Model: Model loaded from ", weights_file)

    @torch.no_grad()
    def predict_step(self, x, idx: int = 0,**kwargs):
        # perform SR
        assert self.model.training == False, "Model in Training mode. Abort." # make sure we're in eval
        p = self.model.forward(x,custom_steps=self.custom_steps,temperature=self.temperature)
        return(p)
    
    @torch.no_grad()
    def uncertainty_map(self, x,n_variations=15,custom_steps=100):
        uncertainty_map = self.model.uncertainty_map(x,n_variations,custom_steps)
        return(uncertainty_map)

