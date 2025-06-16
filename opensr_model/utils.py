import torch
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter


def linear_transform_placeholder(t_input, stage="norm"):
    return( t_input )

def linear_transform_4b(t_input,stage="norm"):
    assert stage in ["norm","denorm"]
    # get the shape of the tensor
    shape = t_input.shape

    # if 5 d tensor, norm/denorm individually
    if len(shape)==5:
        stack = []
        for batch in t_input:
            stack2 = []
            for i in range(0, t_input.size(1), 4):
                slice_tensor = batch[i:i+4, :, :, :]
                slice_denorm = linear_transform_4b(slice_tensor,stage=stage)
                stack2.append(slice_denorm)
            stack2 = torch.stack(stack2)
            stack2 = stack2.reshape(shape[1], shape[2], shape[3], shape[4])
            stack.append(stack2)
        stack = torch.stack(stack)
        return(stack)

    # here only if len(shape) == 4
    squeeze_needed = False
    if len( shape ) == 3:
        squeeze_needed = True
        t_input = t_input.unsqueeze(0)
        shape = t_input.shape

    assert len(shape)==4 or len(shape)==5,"Input tensor must have 4 dimensions (B,C,H,W) - or 5D for MISR"
    transpose_needed = False
    if shape[-1]>shape[1]:
        transpose_needed = True
        t_input = rearrange(t_input,"b c h w -> b w h c")
    
    # define constants
    rgb_c = 3.
    nir_c = 5.

    # iterate over batches
    return_ls = []
    for t in t_input:
        if stage == "norm":
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (10.0 / rgb_c) # R
            t[:,:,1] = t[:,:,1] * (10.0 / rgb_c) # G
            t[:,:,2] = t[:,:,2] * (10.0 / rgb_c) # B
            t[:,:,3] = t[:,:,3] * (10.0 / nir_c) # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0,1)
            # bring to -1..+1
            t = (t*2)-1
        if stage == "denorm":
            # bring to 0..1
            t = (t+1)/2
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (rgb_c / 10.0) # R
            t[:,:,1] = t[:,:,1] * (rgb_c / 10.0) # G
            t[:,:,2] = t[:,:,2] * (rgb_c / 10.0) # B
            t[:,:,3] = t[:,:,3] * (nir_c / 10.0) # NIR
            # clamp to get rif of outlier pixels
            t = t.clamp(0,1)
        
        # append result to list
        return_ls.append(t)

    # after loop, stack image
    t_output = torch.stack(return_ls)
    #print("stacked",t_output.shape)

    if transpose_needed==True:
        t_output = rearrange(t_output,"b w h c -> b c h w")
    if squeeze_needed:
        t_output = t_output.squeeze(0)

    return(t_output)


def linear_transform_6b(t_input,stage="norm"):
    # iterate over batches
    assert stage in ["norm","denorm"]
    bands_c = 5.
    return_ls = []
    clamp = False
    for t in t_input:
        if stage == "norm":
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (10.0 / bands_c) 
            t[:,:,1] = t[:,:,1] * (10.0 / bands_c) 
            t[:,:,2] = t[:,:,2] * (10.0 / bands_c) 
            t[:,:,3] = t[:,:,3] * (10.0 / bands_c) 
            t[:,:,4] = t[:,:,4] * (10.0 / bands_c) 
            t[:,:,5] = t[:,:,5] * (10.0 / bands_c) 
            # clamp to get rif of outlier pixels
            if clamp:
                t = t.clamp(0,1)
            # bring to -1..+1
            t = (t*2)-1
        if stage == "denorm":
            # bring to 0..1
            t = (t+1)/2
            # divide according to conventions
            t[:,:,0] = t[:,:,0] * (bands_c / 10.0) 
            t[:,:,1] = t[:,:,1] * (bands_c / 10.0) 
            t[:,:,2] = t[:,:,2] * (bands_c / 10.0) 
            t[:,:,3] = t[:,:,3] * (bands_c / 10.0) 
            t[:,:,4] = t[:,:,4] * (bands_c / 10.0)
            t[:,:,5] = t[:,:,5] * (bands_c / 10.0)
            # clamp to get rif of outlier pixels
            if clamp:
                t = t.clamp(0,1)
        
        # append result to list
        return_ls.append(t)

    # after loop, stack image
    t_output = torch.stack(return_ls)

    return t_output

def assert_tensor_validity(tensor):

    # ASSERT BATCH DIMENSION
    # if unbatched, add batch dimension
    if len(tensor.shape)==3:
        tensor = tensor.unsqueeze(0)

    # ASSERT BxCxHxW ORDER
    # Check the size of the input tensor
    if tensor.shape[-1]<10:
        tensor = rearrange(tensor,"b w h c -> b c h w")


    height, width = tensor.shape[-2],tensor.shape[-1]    
    # Calculate how much padding is needed for height and width
    if height < 128 or width < 128:
        pad_height = max(0, 128 - height)  # Amount to pad on height
        pad_width = max(0, 128 - width)    # Amount to pad on width

        # Padding for height and width needs to be added to both sides of the dimension
        # The pad has the format (left, right, top, bottom)
        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        padding = padding
        
        # Apply symmetric padding
        tensor = torch.nn.functional.pad(tensor, padding, mode='reflect')

    else: # save padding with 0s
        padding = (0,0,0,0)
        padding = padding

    return tensor,padding



def revert_padding(tensor,padding):
    left, right, top, bottom = padding
    # account for 4x upsampling Factor
    left, right, top, bottom = left*4, right*4, top*4, bottom*4
    # Calculate the indices to slice from the padded tensor
    start_height = top
    end_height = tensor.size(-2) - bottom
    start_width = left
    end_width = tensor.size(-1) - right

    # Slice the tensor to remove padding
    unpadded_tensor = tensor[:,:, start_height:end_height, start_width:end_width]
    return unpadded_tensor



def create_no_data_mask(X,target_size = 512):
    """
    Create a mask for no-data values in the input tensor.
    No-data values are defined as those equal to 0.
    
    Args:
        X (torch.Tensor): Input tensor of shape (B, C, H, W).
        
    Returns:
        torch.Tensor: Mask of shape (B, C, H, W) where no-data values are marked as True.
    """
    # Create a mask where no-data values are True
    interpolated_X = torch.nn.functional.interpolate(X, size=(target_size,target_size), mode='nearest')
    mask = (interpolated_X == 0).float()
    return mask

def apply_no_data_mask(X, mask):
    """
    Apply a no-data mask to the input tensor.
    
    Args:
        X (torch.Tensor): Input tensor of shape (B, C, H, W).
        mask (torch.Tensor): Mask of shape (B, C, H, W) where no-data values are marked as True.
        
    Returns:
        torch.Tensor: Tensor with no-data values set to 0.
    """
    # Apply the mask to set no-data values to 0
    X_masked = X * (1 - mask)
    return X_masked


import matplotlib.pyplot as plt
from einops import rearrange

def plot_example(lr, sr, out_file="example.png"):
    # Scale and clamp
    sr = sr.cpu() * 3.5
    sr = sr.clamp(0.001, 0.9999)
    lr = lr.cpu() * 3.5
    lr = lr.clamp(0.001, 0.9999)

    # Convert to HWC NumPy
    lr_img = rearrange(lr[0, :3, :, :], 'c h w -> h w c').numpy()
    sr_img = rearrange(sr[0, :3, :, :], 'c h w -> h w c').numpy()

    # Center crop 32x32
    def center_crop(img, size=32):
        h, w, _ = img.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return img[top:top+size, left:left+size, :]

    lr_crop = center_crop(lr_img)
    sr_crop = center_crop(sr_img, size=32*4)

    # Plot 2 rows × 2 columns
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(lr_img)
    ax[0, 0].set_title("LR")
    ax[0, 1].imshow(sr_img)
    ax[0, 1].set_title("SR")
    ax[1, 0].imshow(lr_crop)
    ax[1, 0].set_title("LR Center Crop 32x32")
    ax[1, 1].imshow(sr_crop)
    ax[1, 1].set_title("SR Center Crop 32x32")

    for a in ax.flatten():
        a.axis('off')

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    

def plot_uncertainty(uncertainty_map,out_file="uncertainty_map.png",normalize=True):
    uncertainty_map = uncertainty_map.cpu()
    img = rearrange(uncertainty_map[0, :, :, :], 'c h w -> h w c').numpy()
    
    # Convert to grayscale if single-channel
    if img.shape[2] == 1:
        img = img[:, :, 0]

    if normalize:
        # Stretch to [0, 1]
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)
    
    # apply gaussian to smoothen visualization
    img = gaussian_filter(img, sigma=1.0)

    # plot image
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(img, cmap='RdYlGn_r')
    ax.set_title("Uncertainty Map")
    label = "Uncertainty (Normalized)" if normalize else "Uncertainty (Absolute)"
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def download_from_HF(file_name="example_lr.pt"):
    """
    Downloads an example low-resolution tensor from the Hugging Face Hub.
    This function uses the `hf_hub_download` function to fetch a sample tensor
    for testing purposes.
    
    Returns:
        torch.Tensor: The downloaded low-resolution tensor.
    """
    from huggingface_hub import hf_hub_download
    import torch
    

    # Download the file from your HF model repo
    file_path = hf_hub_download(
        repo_id="simon-donike/RS-SR-LTDF",
        filename=file_name,
        repo_type="model"  # or "dataset" if it's in a dataset repo
    )

    # Load the tensor
    obj = torch.load(file_path)
    return(obj)

import contextlib
import os
import sys
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


