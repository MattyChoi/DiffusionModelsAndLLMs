import numpy as np
import matplotlib.pyplot as plt


def plot_sample_image(imgs, cols=4):
    b = len(imgs)
    
    plt.figure(figsize=(15,15))
    plt.axis('off')

    for i, img in enumerate(imgs):
        plt.subplot(b // cols + 1, cols, i + 1)
        plt.imshow(img)

    plt.savefig('sample.png')
    plt.show() 