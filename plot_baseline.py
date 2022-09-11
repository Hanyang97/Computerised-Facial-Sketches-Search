"Plot baseline of 5 dims"
# from itertools import product
# from tqdm import tqdm

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

import torch
# import torch.nn as nn
# import pyro

# import mlflow
# import mlflow.pytorch
# from experiment_tools.pyro_tools import auto_seed

import os
# from torchvision.utils import save_image
# import matplotlib.image as mpimg

# import scipy.stats as st
# import argparse

# from pyro.infer import SVI, Trace_ELBO
# from oed.primitives import observation_sample, latent_sample, compute_design
# import pyro.distributions as dist

# from pyro.infer.autoguide.initialization import init_to_uniform

# import pickle

# import matplotlib.pyplot as plt
# from experiment import VAEXperiment
# import models
# from torchvision.utils import save_image
# import yaml
# import imageio.v2 as iio
# from torchvision import transforms
import pandas as pd

def record_baseline(dir, p, rounds, num_exp):
    d = {'mean': [], 'std': []}
    target = torch.FloatTensor([0.5]*p)
    samples = torch.randn(num_exp, 9, p)
    for i in range(rounds):
        dist = torch.norm(samples - target, dim=-1)
        select_idx = torch.argmin(dist, dim=-1)
        select = samples[torch.arange(num_exp), select_idx]
        dists = dist[torch.arange(num_exp), select_idx]
        mean = torch.mean(dists)
        std = torch.std(dists)
        d['mean'].append(mean)
        d['std'].append(std)
        new_samples = torch.randn(num_exp, 8, p)
        samples = torch.cat([new_samples, select.unsqueeze(1)], dim=1)
    dat = pd.DataFrame(d)
    dat.to_csv(f'{dir}/baseline_{p}dim.csv')



if __name__ == '__main__':
    dir = './baseline'
    if not os.path.exists(dir):
        os.mkdir(dir)
    record_baseline(dir, 20, 60, 4000)