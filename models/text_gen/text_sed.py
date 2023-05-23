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
class TextSED(nn.Module):
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
        