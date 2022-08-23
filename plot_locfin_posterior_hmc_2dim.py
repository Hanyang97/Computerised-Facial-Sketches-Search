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

from pyro.infer.mcmc import MCMC, NUTS
import scipy.stats as st
import argparse

from pyro.infer import SVI, Trace_ELBO
from oed.primitives import observation_sample, latent_sample, compute_design
import pyro.distributions as dist


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


def plot_posterior_grid(limits, grid, pdf_post_list, designs, T_to_plot, mi_estimator, true_theta, dir, index, map_list):
    xx, yy = grid


    fig, axs = plt.subplots(1, len(T_to_plot), figsize=(6*len(T_to_plot), 6), dpi=200, sharey=True)

    for i, T in enumerate(T_to_plot):
        vmin = 0
        vmax = np.max(pdf_post_list[i])
        levels = np.linspace(vmin, vmax, 10)
        ax = axs[i]
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        theta_pos = ax.scatter(
            true_theta[0][0],
            true_theta[0][1],
            c="r",
            marker="x",
            s=200,
            zorder=20,
            label="Ground truth",
        )

        ax.scatter(
            map_list[i][0],
            map_list[i][1],
            c="g",
            marker="+",
            s=200,
            zorder=20,
            label="MAP",
        )

        designs_0, designs_1 = [], []
        for j, design in enumerate(designs[:T]):
            designs_0.append(design.squeeze()[0])
            designs_1.append(design.squeeze()[1])
        ax.scatter(
            designs_0,
            designs_1,
            color='k',
            marker="o",
            s=15,
            zorder=19,
            label="Design",
        )


        # Contourf plot
        cfset = ax.contourf(xx, yy, pdf_post_list[i], cmap='Blues',levels=levels[:], zorder=10)
        ## Or kernel density estimate plot instead of the contourf plot
        #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

        # Contour plot
        cset = ax.contour(xx, yy, pdf_post_list[i], colors='k')

        # Label plot
        ax.clabel(cset, inline=1, fontsize=15)
        ax.set_xlabel('first dimension', size=15)
        
        l = ax.legend(loc="upper left")
        l.set_zorder(21)
        ax.tick_params(labelsize=15)
        # ax.grid(True, ls="--")
        ax.set_title(f'Experiment {T}', size=15)
        fig.colorbar(cfset, ax=ax)
    
    axs[0].set_ylabel('second dimension', size=15)

    fig.suptitle(f'T={T_to_plot[-1]} {mi_estimator} Posterior', size=25)

    

    plt.tight_layout()
    plt.savefig(f"{dir}/2dim_{mi_estimator}_{T_to_plot[-1]}_{index}progress.png")



def plot_vae_decode(vae_model, embedding_list, dir, mi_estimator, T_list, index, type='True_Embedding'):
    fig, axs = plt.subplots(1, len(embedding_list), figsize=(6*len(embedding_list), 6), dpi=200)
    for i, embedding in enumerate(embedding_list):
        recon = vae_model.decode(embedding.cpu())
        tmp_img_dir = os.path.join(dir, f"temp.png")
        save_image(recon, tmp_img_dir, normalize=True, value_range=(-1,1))
        img = mpimg.imread(tmp_img_dir)
        axs[i].imshow(img)
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
        if i < len(T_list):
            dist = (embedding - embedding_list[-1]).norm(p=2).cpu().numpy()
            axs[i].set_title(f'Experiment {T_list[i]}, Distance={dist:.2f}', size=15)
        else:
            axs[i].set_title(f'Target', size=15)
    os.remove(tmp_img_dir)
    fig.suptitle(f'T={T_list[-1]} {mi_estimator} {type}', size=25)
    plt.savefig(f"{dir}/2_dim_{mi_estimator}_{type}_{index}_progress.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MAP and Posterior at dim 2."
    )
    parser.add_argument("--experiment-id", default="1", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--mi-estimator", default="PCE_Attention", type=str)
    parser.add_argument("--theta-index", default=0, type=int)
    args = parser.parse_args()

    seed = auto_seed(args.seed)

    dir = './latex_figures'
    if not os.path.exists(dir):
        os.mkdir(dir)
    ########## put your own runid here
    if args.mi_estimator == 'infoNCE':
        run_id = "3978677728c64ab1b2dbcbe76eff95ba"
    elif args.mi_estimator == 'NWJ':
        run_id = 'aeb8e31099c049bca676528bc328065a'
    elif args.mi_estimator == 'PCE':
        run_id = 'fd674fe1061a49dc8dbba2499a6b8233'
    elif args.mi_estimator == 'PCE_Attention':
        run_id = '85d37b20c50c4fb591e795ade881909c'
    #######################################
    with mlflow.start_run(run_id=run_id) as run:
        mi_estimator = mlflow.ActiveRun(run).data.params["mi_estimator"]
        design_arch = mlflow.ActiveRun(run).data.params["design_arch"]
    if mi_estimator == 'sPCE':
        mi_estimator = '_'.join((mi_estimator, design_arch))

    artifact_path = f"mlruns/{args.experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    vae_location = f'{artifact_path}/vae_model'

    # load model and critic
    ho_model = mlflow.pytorch.load_model(model_location, map_location=args.device)
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
    if p != 2:
        print('USE 2 DIMENSION!')

    ####### true theta ###############
    ####### currently can only run one true theta, easy to improve code
    theta_list = [
        torch.tensor([[-1, -0.3]]).to(args.device),
        torch.tensor([[0.5, -0.4]]).to(args.device),
        torch.tensor([[-0.2, -1]]).to(args.device)
    ]
    theta = theta_list[args.theta_index]
    map_list = []   # maximum a posterior
    f_list = []     # posterior pdf
    limits = (-1.5, 1.5)
    xmin, xmax = limits
    ymin, ymax = limits
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    # one experiment
    designs, observations, _ = run_policy(ho_model, theta)
    T_to_plot = [0, 5, 10, ho_model.T]


    for T in T_to_plot:
        data_dict = {}
        temp, ho_model.T = ho_model.T, T
        for t in range(ho_model.T):
            data_dict[f'xi{t+1}'] = designs[t].unsqueeze(0)
            data_dict[f'y{t+1}'] = observations[t].unsqueeze(0)

        def model(data_dict):
            with pyro.plate_stack("expand_theta_test", [theta.shape[0]]):
                # condition on theta
                return pyro.condition(ho_model.model, data=data_dict)()
    
        kernel = NUTS(model, target_accept_prob=0.9)        
        mcmc = MCMC(kernel, num_samples=1000*p, warmup_steps=200*p, num_chains=4)

        mcmc.run(data_dict)
        print(mcmc.summary())
        print(mcmc.diagnostics())

        posterior = mcmc.get_samples()['theta']
        posterior = posterior.squeeze().cpu().numpy()
        posterior_by_dim = [posterior[:, i] for i in range(p)]
        values = np.vstack(posterior_by_dim)
        kernel = st.gaussian_kde(values)
        pdf = kernel(positions)

        def map_model(data_dict):
            ########################################################################
            # Sample latent variables theta
            ########################################################################
            theta = latent_sample("theta", ho_model.theta_prior)
            y_outcomes = []
            xi_designs = []

            # T-steps experiment
            for t in range(ho_model.T):
                ####################################################################
                # Get a design xi; shape is [batch size x ho_model.n x ho_model.p]
                ####################################################################
                xi = compute_design(
                    f"xi{t + 1}", ho_model.design_net.lazy(*zip(xi_designs, y_outcomes)), obs=data_dict[f"xi{t + 1}"]
                )
                ####################################################################
                # Sample y at xi; shape is [batch size x 1]
                ####################################################################
                mean = ho_model.forward_map(xi, theta)
                sd = ho_model.noise_scale
                y = observation_sample(f"y{t + 1}", dist.Normal(mean, sd).to_event(1), obs=data_dict[f"y{t + 1}"])

                ####################################################################
                # Update history
                ####################################################################
                y_outcomes.append(y)
                xi_designs.append(xi)

            return theta, xi_designs, y_outcomes

        autoguide_map = pyro.infer.autoguide.AutoDelta(map_model)
    
        def train(model, guide, lr=0.005, n_steps=12001):
            pyro.clear_param_store()
            adam_params = {"lr": lr, 'betas': (0.95, 0.999)}
            adam = pyro.optim.Adam(adam_params)
            svi = SVI(model, guide, adam, loss=Trace_ELBO())

            for step in range(n_steps):
                loss = svi.step(data_dict)
                if step % 50 == 0:
                    print('[iter {}]  loss: {:.4f}'.format(step, loss))
        
        autoguide_map = pyro.infer.autoguide.AutoDelta(model)
        train(map_model, autoguide_map)

        MAP = autoguide_map.median(data_dict)["theta"]
        print("Our MAP estimate of the theta is {}".format(MAP.cpu()))
        MAP = MAP.squeeze().cpu()
        map_list.append(MAP)


        # map_list.append(torch.tensor(positions[:, pdf.argmax()], dtype=torch.float32, device=args.device).reshape(-1))
        f = np.reshape(pdf.T, xx.shape)
        f_list.append(f)

        ho_model.T = temp

    if p == 2:
        plot_posterior_grid(limits, (xx, yy), f_list, designs, T_to_plot, mi_estimator, theta, dir, args.theta_index, map_list)
        plot_vae_decode(vae_model, map_list+[true_theta for true_theta in theta], dir, mi_estimator, T_to_plot, args.theta_index, type='MAP')
