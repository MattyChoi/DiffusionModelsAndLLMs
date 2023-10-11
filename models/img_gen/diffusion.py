import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

from models.variance_scheduler import linear_beta_schedule, cosine_beta_schedule


# helper function to fetch the variance variables
def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# diffusion model learns the mean and variance parameters for each time step for 
# the gaussian distributions of the backward diffusion process
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size=64,
        timesteps=300,
        beta_schedule = 'cosine',
    ):
        super().__init__()

        # model takes in a noisy image (image * mean_t + noise * beta_t) and timestep t and predicts noise in the image
        # common nn for this task is unet
        self.model = model
        
        # get number of timesteps to perform
        self.timesteps = timesteps

        # create the variances scheduler
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps=timesteps)
        else:
            betas = linear_beta_schedule(timesteps=timesteps)

        self.image_size = image_size
        self.betas = betas

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # used to calculate the mean of the forward diffusion process
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # variables used to calculate the mean and variance of the reverse diffusion process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    # sample of the reverse diffusion process
    # Without adding @torch.no_grad() we quickly run out of memory, 
    # because pytorch tacks all the previous images for gradient calculation
    @torch.no_grad()
    def p_sample(self, x_t, t, t_index):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)

        # use the model to predict the noise and calculate mean of the current reverse diffusion distribution
        pred_noise = self.model(x_t, t)
        posterior_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # get the variance of the current reverse diffusion distribution
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_t.shape)
        
        if t_index == 0:
            return posterior_mean
        
        noise = torch.randn_like(x_t)
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise 


    # run the reverse diffusion process in its entirety
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            tsfm_img = torch.clip((img.permute((0, 2, 3, 1)) + 1) / 2.0, 0, 1).cpu().numpy()
            imgs.append(tsfm_img)
        return imgs


    def sample(self, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, self.image_size, self.image_size))


    # sample of the forward diffusion process
    def q_sample(self, x_0, t):
        device = x_0.device
        noise = torch.randn_like(x_0)

        # coefficient of the means of the forward diffusino process
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # calculate the mean and the variance of the gaussian
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        var = sqrt_one_minus_alphas_cumprod_t.to(device)

        return (mean + var * noise).to(device), noise.to(device)
        

    def forward(self, x_0):
        '''
        x_0: images from the data distribution
        x_t: images after t timesteps of adding Gaussian noise
        '''
        t = torch.randint(0, self.timesteps, (x_0.size(0),), device=x_0.device).long()
        x_t, noise = self.q_sample(x_0, t)
        pred_noise = self.model(x_t, t)

        return pred_noise, noise