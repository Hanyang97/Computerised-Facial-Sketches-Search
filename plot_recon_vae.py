"Plot reconstructed images of VAE in different dimensions"
from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pyro

import mlflow
import mlflow.pytorch
from experiment_tools.pyro_tools import auto_seed

import os
from torchvision.utils import save_image
import matplotlib.image as mpimg

import scipy.stats as st
import argparse

from pyro.infer import SVI, Trace_ELBO
from oed.primitives import observation_sample, latent_sample, compute_design
import pyro.distributions as dist

from pyro.infer.autoguide.initialization import init_to_uniform

import pickle
import pandas as pd

import matplotlib.pyplot as plt
from experiment import VAEXperiment
import models
from torchvision.utils import save_image
import yaml
import imageio.v2 as iio
from torchvision import transforms

def plot_vaes(img_dir, vae_list):
    img_paths = [f'{img_dir}/{i}' for i in os.listdir(img_dir)]
    ncol = len(vae_list) + 1
    nrow = len(img_paths)
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow), dpi=200)
    tmp_img_dir = img_dir + '/temp.png'
    for r in range(nrow):
        axs[r, 0].imshow(mpimg.imread(img_paths[r]))
        image = iio.imread(img_paths[r])
        trans = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
                )
        image_vec = trans(image).unsqueeze(0)
        for c in range(1, ncol):
            vaemodel = vae_list[c-1]
            recon = vaemodel.generate(image_vec)
            save_image(recon, tmp_img_dir, normalize=True, value_range=(-1,1))
            img = mpimg.imread(tmp_img_dir)
            axs[r, c].imshow(img)
    for r in range(nrow):
        for c in range(ncol):
            axs[r, c].axes.xaxis.set_visible(False)
            axs[r, c].axes.yaxis.set_visible(False)
    axs[0, 0].set_title('Original', size=25)
    for c in range(1, ncol):
        p = vae_list[c-1].latent_dim
        axs[0, c].set_title(f'Dim {p}', size=25)
    os.remove(tmp_img_dir)
    fig.suptitle(f'Example Reconstructed Images', size=35, y=0.95)
    plt.savefig(f"{img_dir}/reconvae.png")

def load_vae(vaelog_dirs, ckpt_paths):
    vae_list = []
    for i, vaelog_dir in enumerate(vaelog_dirs):
        vaeconfig_path = vaelog_dir + '/config.yaml'
        with open(vaeconfig_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        experiment = VAEXperiment.load_from_checkpoint(
            ckpt_paths[i], 
            vae_model = models.vae_models[config['model_params']['name']](**config['model_params']),
            params = config['exp_params']
        )
        vae_model = experiment.model.to('cpu')
        vae_list.append(vae_model)
    return vae_list

if __name__ == '__main__':
    img_dir = './example_cfs'
    vaelog_versions = [0, 1, 2, 3, 4]
    vaelog_versions.reverse()
    vaelog_dirs = [f'./vaelogs/VanillaVAE/version_{i}' for i in vaelog_versions]
    ckpt_paths = [f'{vaelog_dir}/checkpoints/last.ckpt' for vaelog_dir in vaelog_dirs]
    ##### load and log vae model
    vae_list = load_vae(vaelog_dirs, ckpt_paths)
    plot_vaes(img_dir, vae_list)