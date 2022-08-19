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




def run_policy(implicit_model, theta=None, verbose=True):
    """
    can specify either theta or index. If none specified it will sample.
    If both are specified theta is used and indices is ignored.
    """
    if theta is not None:
        # condition on thetas
        def model():
            with pyro.plate_stack("expand_theta_test", [theta.shape[0]]):
                # condition on theta
                return pyro.condition(implicit_model.model, data={"theta": theta})()

    else:
        model = implicit_model.model

    with torch.no_grad():
        trace = pyro.poutine.trace(model).get_trace()
        designs = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "design_sample"
        ]
        observations = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        ]
        latents = [
            node["value"].detach()
            for node in trace.nodes.values()
            if node.get("subtype") == "latent_sample"
        ]
        latents = torch.cat(latents, axis=-1)

    return designs, observations, latents


def plot_posterior_grid(T0, T1, pdf_post, mi_estimator, true_theta, limits, dir):
    vmin = 0
    vmax = np.max(pdf_post)
    levels = np.linspace(vmin, vmax, 6)

    fig, axs = plt.subplots(1, len(true_theta), figsize=(6*len(true_theta), 6), dpi=200, sharey=True)
    # fig = plt.figure(figsize=(6, 6), dpi=300)
    # ax = fig.add_subplot(111)
    

    for i, ax in enumerate(axs):
        CS_post = ax.contourf(
            T0, T1, pdf_post[i], cmap="BuGn", linewidths=1.5, levels=levels[:], zorder=10
        )
    
        ax.scatter(
            true_theta[i][0],
            true_theta[i][1],
            c="r",
            marker="x",
            s=200,
            zorder=20,
            label="Ground truth",
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.legend(loc="upper left")
        ax.set_xlabel(r"first dimension", size=10)
        ax.set_ylabel(r"second dimension", size=10)
        # ax.set_title("T=30 Example Posterior", size=10)
        ax.tick_params(labelsize=20)
        ax.grid(True, ls="--")
    fig.suptitle(f'T=30 Example {mi_estimator} Posterior', size=25)

    fig.colorbar(CS_post)

    plt.tight_layout()
    # plt.show()
    # return fig, ax
    plt.savefig(f"{dir}/2_dim_{mi_estimator}_posterior.png")


def get_posterior_logprob(critic, prior, mi_estimator, eval_latents, *design_obs_pairs):
    with torch.no_grad():
        foo = []
        for eval_latent in eval_latents:
            foo.append(critic(eval_latent.unsqueeze(dim=0), *design_obs_pairs)[0])
        foo = torch.tensor(foo)
    const = 0.0 if mi_estimator == "InfoNCE" else 1.0
    res = foo.squeeze(0) + prior.log_prob(eval_latents).sum(-1) + const
    # normalize
    return res - res.logsumexp(0)

def plot_vae_decode(vae_model, embedding_list, dir, mi_estimator, type='True_Embedding'):
    fig, axs = plt.subplots(1, len(embedding_list), figsize=(6*len(embedding_list), 6), dpi=200)
    for i, embedding in enumerate(embedding_list):
        recon = vae_model.decode(embedding.cpu())
        tmp_img_dir = os.path.join(dir, f"temp.png")
        save_image(recon, tmp_img_dir, normalize=True, value_range=(-1,1))
        img = mpimg.imread(tmp_img_dir)
        axs[i].imshow(img)
    os.remove(tmp_img_dir)
    fig.suptitle(f'T=30 Example {mi_estimator} {type}', size=25)
    plt.savefig(f"{dir}/2_dim_{mi_estimator}_{type}.png")



if __name__ == "__main__":
    dir = './latex_figures'
    if not os.path.exists(dir):
        os.mkdir(dir)

    device = "cpu"
    mi_estimator = 'NWJ'

    if mi_estimator == 'infoNCE':
        run_id = "3978677728c64ab1b2dbcbe76eff95ba"
    elif mi_estimator == 'NWJ':
        run_id = 'aeb8e31099c049bca676528bc328065a'
    with mlflow.start_run(run_id=run_id) as run:
        mi_estimator = mlflow.ActiveRun(run).data.params["mi_estimator"]
    artifact_path = f"mlruns/1/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    critic_location = f"{artifact_path}/critic"
    vae_location = f'{artifact_path}/vae_model'

    # load model and critic
    ho_model = mlflow.pytorch.load_model(model_location, map_location=device)
    critic_net = mlflow.pytorch.load_model(critic_location, map_location=device)
    vae_model = mlflow.pytorch.load_model(vae_location, map_location=device)
    vae_model.eval()

    p = ho_model.p * 1
    scale = ho_model.noise_scale * torch.tensor(
        1.0, dtype=torch.float32, device=device
    )
    mean = torch.zeros(p).to(device)
    prior = torch.distributions.multivariate_normal.MultivariateNormal(mean, scale*torch.eye(p))

    

    # prepare theta grid

    N_GRID = 300
    auto_seed(420)
    beta_lims = [-1, 2]
    gamma_lims = [-1, 2]
    t_beta = torch.linspace(*beta_lims, N_GRID).to(device)
    t_gamma = torch.linspace(*gamma_lims, N_GRID).to(device)
    T0, T1 = torch.meshgrid(t_beta, t_gamma)
    theta_grid = torch.tensor(list(product(t_beta, t_gamma))).to(device)

    true_thetas = torch.tensor(
        [[0.15, 0.15], [-0.2, 0.3], [1.1, 0.4]]
    ).to(device)

    plot_vae_decode(vae_model, true_thetas, dir, mi_estimator)



    foo_list = []
    true_theta_list = []
    map_list = []

    for i, tt in enumerate(true_thetas):
        true_theta = tt.unsqueeze(0)
        designs, observations, latents = run_policy(ho_model, true_theta)

        foo = get_posterior_logprob(
            critic_net,
            prior,
            mi_estimator,
            theta_grid,
            *zip(list(designs), list(observations)),
        )
        map_list.append(theta_grid[foo.argmax()])
        foo_list.append(foo.reshape(N_GRID, N_GRID).exp().cpu().numpy())
        true_theta_list.append(true_theta.squeeze(0).cpu().numpy())

    plot_posterior_grid(
        T0.cpu().numpy(),
        T1.cpu().numpy(),
        foo_list,
        mi_estimator, 
        true_theta=true_theta_list,
        limits=[beta_lims, gamma_lims],
        dir = dir
    )
    plot_vae_decode(vae_model, map_list, dir, mi_estimator, type='MAP')