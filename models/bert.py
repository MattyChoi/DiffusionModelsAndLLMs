import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())


# BERT architecture
class BERT(nn.Module):
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
        return 0


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
        
        return 0
        

    def forward(self, x_0):
        '''
        x_0: images from the data distribution
        x_t: images after t timesteps of adding Gaussian noise
        '''
        t = torch.randint(0, self.timesteps, (x_0.size(0),), device=x_0.device).long()
        x_t, noise = self.q_sample(x_0, t)
        pred_noise = self.model(x_t, t)

        return pred_noise, noise