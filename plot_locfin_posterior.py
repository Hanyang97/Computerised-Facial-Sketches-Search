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
import argparse



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

# version using for loop

# def get_posterior_logprob(critic, prior, mi_estimator, eval_latents, *design_obs_pairs):
#     with torch.no_grad():
#         foo = []
#         for eval_latent in eval_latents:
#             foo.append(critic(eval_latent.unsqueeze(dim=0), *design_obs_pairs)[0])
#         foo = torch.tensor(foo)
#     const = 0.0 if mi_estimator == "InfoNCE" else 1.0
#     res = foo.squeeze(0) + prior.log_prob(eval_latents).sum(-1) + const
#     # normalize
#     return res - res.logsumexp(0)

def get_posterior_logprob(critic, prior, mi_estimator, eval_latents, *design_obs_pairs):
    '''
    allow estimation of posterior at eval_latents at the same time
    '''
    ############# to change method in latent encoder network to allow parallel computing #####
    def temp(self, x, y=None):
        return x.flatten(-2)
    def new_prepare_input(self, x, y=None):
        return x
    import types
    critic.latent_encoder_network._prepare_input = types.MethodType(new_prepare_input, critic.latent_encoder_network)

    ##########################################################################
    with torch.no_grad():
        foo, _ = critic(eval_latents, *design_obs_pairs)
    ### change back ###################
    critic.latent_encoder_network._prepare_input = types.MethodType(temp, critic.latent_encoder_network)
    ###########################
    const = 0.0 if mi_estimator == "InfoNCE" else 1.0
    res = foo.squeeze(0) + prior.log_prob(eval_latents).sum(-1) + const
    # normalize
    return res - res.logsumexp(0)

def plot_vae_decode(vae_model, embedding_list, dir, mi_estimator, T, p, type='True_Embedding'):
    fig, axs = plt.subplots(1, len(embedding_list), figsize=(6*len(embedding_list), 6), dpi=200)
    for i, embedding in enumerate(embedding_list):
        recon = vae_model.decode(embedding.cpu())
        tmp_img_dir = os.path.join(dir, f"temp.png")
        save_image(recon, tmp_img_dir, normalize=True, value_range=(-1,1))
        img = mpimg.imread(tmp_img_dir)
        axs[i].imshow(img)
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    dist = (embedding_list[0] - embedding_list[-1]).norm(p=2).cpu().numpy()
    axs[0].set_title(f'Experiment {T}, Distance={dist:.2f}', size=15)
    axs[1].set_title('Target', size=15)
    os.remove(tmp_img_dir)
    fig.suptitle(f'{mi_estimator} {type} Using Critics', size=25)
    plt.savefig(f"{dir}/{p}_dim_{mi_estimator}_{type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MAP and Posterior Using Critics."
    )
    parser.add_argument("--experiment-id", default="1", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--run-id", default="7d9ff16d7267491a8c88bbcf5f06d3ed", type=str)
    parser.add_argument("--bin", default=50, type=int)
    parser.add_argument("--limits", nargs='+', default=[-1.5, 1.5])
    parser.add_argument("--T", default=-1, type=int)

    args = parser.parse_args()

    seed = auto_seed(args.seed)

    dir = './latex_figures'
    if not os.path.exists(dir):
        os.mkdir(dir)


    # if mi_estimator == 'infoNCE':
    #     run_id = "3978677728c64ab1b2dbcbe76eff95ba"
    # elif mi_estimator == 'NWJ':
    #     run_id = 'aeb8e31099c049bca676528bc328065a'

    with mlflow.start_run(run_id=args.run_id) as run:
        mi_estimator = mlflow.ActiveRun(run).data.params["mi_estimator"]
    artifact_path = f"mlruns/{args.experiment_id}/{args.run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    critic_location = f"{artifact_path}/critic"
    vae_location = f'{artifact_path}/vae_model'

    # load model and critic
    ho_model = mlflow.pytorch.load_model(model_location, map_location=args.device)
    critic_net = mlflow.pytorch.load_model(critic_location, map_location=args.device)
    vae_model = mlflow.pytorch.load_model(vae_location, map_location=args.device)
    vae_model.eval()

    ############# solely to correct previous model mistake, no need for future experiments #####
    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        # two norm squared
        self.norm = ho_model.norm
        # mlflow.log_param('norm', self.norm)
        sq_two_norm = (xi - theta).norm(p=self.norm, dim=-1).pow(2)
        # sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        # add a small number before taking inverse (determines max signal)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y
    import types
    ho_model.forward_map = types.MethodType(forward_map, ho_model)
    ##########################################################################

    p = ho_model.p * 1
    prior = ho_model.theta_prior

    if args.T == -1:
        args.T = ho_model.T

    output_dir = f'{dir}/{p}dim_{mi_estimator}_{args.T}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    

    # prepare theta grid

    lims = [float(limit) for limit in args.limits]
    # t_beta = torch.linspace(lims, args.bin).to(args.device)
    # t_gamma = torch.linspace(lims, args.bin).to(args.device)
    t_grids = [torch.linspace(*lims, args.bin).to(args.device)] * p
    # T0, T1 = torch.meshgrid(t_beta, t_gamma)

    theta_grid = torch.tensor(list(product(*t_grids))).to(args.device)

    true_thetas = torch.tensor(
        [[0.5]*p]
    ).to(args.device)



    temp, ho_model.T = ho_model.T, args.T

    for i, tt in enumerate(true_thetas):
        map_list = []
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
        # foo_list.append(foo.reshape(args.bin, args.bin).exp().cpu().numpy())
        map_list.append(true_theta.squeeze(0))
        plot_vae_decode(vae_model, map_list, output_dir, mi_estimator, args.T, p, type='MAP')
        

    # plot_posterior_grid(
    #     T0.cpu().numpy(),
    #     T1.cpu().numpy(),
    #     foo_list,
    #     mi_estimator, 
    #     true_theta=true_theta_list,
    #     limits=[beta_lims, gamma_lims],
    #     dir = dir
    # )
    # plot_vae_decode(vae_model, map_list, output_dir, mi_estimator, args.T, true_theta_list, type='MAP')

    ho_model.T = temp