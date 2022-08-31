import os
from torchvision.utils import save_image
import torch
from experiment import VAEXperiment
import models
from PIL import Image
import argparse
from torchvision import transforms
import imageio.v2 as iio
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_vae_reconstruction(model, original_img, dir):
    p = model.latent_dim
    recon = model.generate(original_img)
    save_image(recon, f'{dir}/recon.png', normalize=True, value_range=(-1,1))
    save_image(original_img, f'{dir}/original.png', normalize=True, value_range=(-1,1))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=200, sharey=True)
    recon = mpimg.imread(f'{dir}/recon.png')
    original_img = mpimg.imread(f'{dir}/original.png')
    os.remove(f'{dir}/recon.png')
    os.remove(f'{dir}/original.png')
    axs[0].imshow(original_img)
    axs[1].imshow(recon)
    axs[0].set_title('Original', size=15)
    axs[1].set_title('Reconstruction', size=15)
    for i in range(2):
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    fig.suptitle(f'Reconstruction for Latent Dimension {p}', size=25)
    plt.savefig(f"{dir}/{p}_dim_recon.png")
    
def plot_vae_sample(model, dir):
    p = model.latent_dim
    latent = torch.tensor([[0.5]*p])
    sample = model.decode(latent)
    save_image(sample, f'{dir}/sample_{p}.png', normalize=True, value_range=(-1,1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot VAE."
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--vae-model-dir", default="vaelogs/VanillaVAE/version_0/checkpoints/epoch=73-step=35445.ckpt", type=str)
    parser.add_argument("--vae-config-dir", default="vaelogs/VanillaVAE/version_0/config.yaml", type=str)
    parser.add_argument("--img-dir", default="./img1.png", type=str)


    args = parser.parse_args()

    dir = './latex_figures'
    if not os.path.exists(dir):
        os.mkdir(dir)
    output_dir = '/'.join((dir, 'vae_output'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    torch.manual_seed(args.seed)

    with open(args.vae_config_dir, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    experiment = VAEXperiment.load_from_checkpoint(
        args.vae_model_dir, 
        vae_model = models.vae_models[config['model_params']['name']](**config['model_params']),
        params = config['exp_params']
    )
    vae_model = experiment.model.to(args.device)
    vae_model.eval()

    image = (Image.open(args.img_dir)).resize((64, 64))
    image.save(''.join(('64by64_', args.img_dir[2:])))
    image = iio.imread(''.join(('64by64_', args.img_dir[2:])))
    trans = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
             )
    print(''.join(('64by64_', args.img_dir[2:])))    
    plot_vae_reconstruction(vae_model, trans(image).unsqueeze(0), output_dir)

    plot_vae_sample(vae_model, output_dir)