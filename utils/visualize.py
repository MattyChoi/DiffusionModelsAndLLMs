import numpy as np
import torch
import torchvision.transforms as tsfm
import matplotlib.pyplot as plt

def show_tensor_image(image):
    reverse_transforms = tsfm.Compose([
        tsfm.Lambda(lambda t: (t + 1) / 2),
        tsfm.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        tsfm.Lambda(lambda t: t * 255.),
        tsfm.Lambda(lambda t: t.numpy().astype(np.uint8)),
        tsfm.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))