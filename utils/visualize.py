import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

random_index = 53

def plot_sample_gif(imgs, timesteps=300):
    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        im = plt.imshow(imgs[i][random_index], cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()


def plot_sample_image(imgs, cols=4):
    b = len(imgs)
    
    plt.figure(figsize=(15,15))
    plt.axis('off')

    for i, img in enumerate(imgs):
        plt.subplot(b // cols + 1, cols, i + 1)
        plt.imshow(img)

    plt.savefig('sample.png')
    plt.show() 