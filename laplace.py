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
from plotters import plot_trace

from pyro.infer import SVI, Trace_ELBO
from oed.primitives import observation_sample, latent_sample, compute_design
import pyro.distributions as dist

from pyro.infer.autoguide.initialization import init_to_uniform
import pandas as pd


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


def plot_posterior_grid(limits, grid, pdf_post_list, designs, T_to_plot, mi_estimator, true_theta, dir, index):
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

        designs_0, designs_1 = [], []
        for j, design in enumerate(designs[:T]):
            designs_0.append(design.squeeze()[0])
            designs_1.append(design.squeeze()[1])
        ax.scatter(
            designs_0,
            designs_1,
            color='k',
            marker="o",
            s=10,
            zorder=20,
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
        ax.set_ylabel('second dimension', size=15)
        ax.legend(loc="upper left")
        ax.tick_params(labelsize=15)
        ax.grid(True, ls="--")
        ax.set_title(f'Experiment {T}', size=15)
        fig.colorbar(cfset, ax=ax)

    fig.suptitle(f'T={T_to_plot[-1]} Example {mi_estimator} Posterior', size=25)

    

    plt.tight_layout()
    plt.savefig(f"{dir}/2dim_{mi_estimator}_{T_to_plot[-1]}_{index}progress.png")



def plot_vae_decode(vae_model, embedding_list, dir, mi_estimator, T_list, index, p, type='True_Embedding'):
    ncol = 4
    nrow = len(T_list) // ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 6*nrow), dpi=200)
    for i, embedding in enumerate(embedding_list):
        row, col = i // ncol, i % ncol
        recon = vae_model.decode(embedding.cpu())
        tmp_img_dir = os.path.join(dir, f"temp.png")
        save_image(recon, tmp_img_dir, normalize=True, value_range=(-1,1))
        img = mpimg.imread(tmp_img_dir)
        axs[row, col].imshow(img)
        if i < len(T_list):
            dist = (embedding - embedding_list[-1]).norm(p=2).cpu().numpy()
            axs[row, col].set_title(f'Experiment {T_list[i]}, Distance={dist:.2f}', size=15)
        else:
            axs[row, col].set_title(f'Target', size=15)
    for r in range(nrow):
        for c in range(ncol):
            axs[r, c].axes.xaxis.set_visible(False)
            axs[r, c].axes.yaxis.set_visible(False)
    for ax in axs[-1, len(embedding_list)%ncol:]:
        ax.remove()
    os.remove(tmp_img_dir)
    fig.suptitle(f'T={T_list[-1]} Example {mi_estimator} {type}', size=25)
    plt.savefig(f"{dir}/{p}_dim_{mi_estimator}_{type}_{index}_progress.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MAP and Posterior at Different T."
    )
    parser.add_argument("--experiment-id", default="1", type=str)
    # parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--run-id", default="7d9ff16d7267491a8c88bbcf5f06d3ed", type=str)
    parser.add_argument("--T-to-plot", default=None, type=int)
    parser.add_argument("--theta-index", default=0, type=int)
    # parser.add_argument("--T-step", default=10, type=int)
    parser.add_argument("--svi-lr", default=0.005, type=float)
    parser.add_argument("--svi-steps", default=401, type=int)

    args = parser.parse_args()

    seed = auto_seed(args.seed)

    dir = './laplace'
    if not os.path.exists(dir):
        os.mkdir(dir)

    with mlflow.start_run(run_id=args.run_id) as run:
        mi_estimator = mlflow.ActiveRun(run).data.params["mi_estimator"]
        design_arch = mlflow.ActiveRun(run).data.params["design_arch"]
    if mi_estimator == 'sPCE':
        mi_estimator = '_'.join((mi_estimator, design_arch))

    artifact_path = f"mlruns/{args.experiment_id}/{args.run_id}/artifacts"
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

    ####### true theta ###############
    ####### currently can only run one true theta, easy to improve code
    # theta_list = [torch.tensor([[0.5]*p]).to(args.device)]
    # theta_list = [torch.randn(1, p).to(args.device)]
    theta_list = [torch.tensor([[0.5]*p]).to(args.device)]
    theta = theta_list[args.theta_index]
    #########################################
    map_list = []   # maximum a posterior
    f_list = []     # posterior pdf
    # limits = tuple([float(x) for x in args.limits])
    # limits = (-1.5, 1.5)    ####### important for posterior, lim
    # grid = np.mgrid[[slice(limits[0],limits[1],args.bin*1j) for i in range(p)]]
    # positions = np.vstack([grid[i].ravel() for i in range(p)])
    # if args.T_to_plot is None:
    #     T_to_plot = [i for i in range(0, ho_model.T+1,args.T_step)]
    # else:
    #     T_to_plot = [i for i in range(0, args.T_to_plot+1, args.T_step)]
    

    ####
    T_to_plot = [args.T_to_plot]
    dist_list = []
    
    print(f'Run ID {args.run_id}; Dim {p}; {mi_estimator}; T={args.T_to_plot}')
    


    # one experiment
    temp, ho_model.T = ho_model.T, T_to_plot[-1]
    designs, observations, _ = run_policy(ho_model, theta)
    output = []
    output_path = os.path.join(dir, f'{p}_dim_{mi_estimator}_{ho_model.T}_laplace.csv')
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    # recon = vae_model.decode(theta.squeeze().cpu())
    # save_image(recon, os.path.join(output_path, f"target_{0}.png"), normalize=True, value_range=(-1,1))
    # run_xis = []
    # run_ys = []
    # # Print optimal designs, observations for given theta
    # for t in range(ho_model.T):
    #     xi = designs[t][0].detach().cpu().reshape(-1)
    #     run_xis.append(xi)
    #     y = observations[t][0].detach().cpu().item()
    #     run_ys.append(y)
    #     if t % 5 == 0:
    #         recon = vae_model.decode(xi)
    #         save_image(recon, os.path.join(output_path, f"target_{0}_recon_{t}.png"), normalize=True, value_range=(-1,1))

    # run_df = pd.DataFrame(torch.stack(run_xis).numpy())
    # run_df.columns = [f"xi_{i}" for i in range(ho_model.p)]
    # run_df["observations"] = run_ys
    # run_df["order"] = list(range(1, ho_model.T + 1))
    # run_df["run_id"] = 0 + 1
    # output.append(run_df)

    # plot_trace(0, ho_model.p, ho_model.T, run_df, theta.cpu().numpy(), norm=ho_model.norm, face_finding=True, face_folder=output_path)
    
    ho_model.T = temp

    for T in T_to_plot:
        data_dict = {}
        temp, ho_model.T = ho_model.T, T
        for t in range(ho_model.T):
            data_dict[f'xi{t+1}'] = designs[t].unsqueeze(0)
            data_dict[f'y{t+1}'] = observations[t].unsqueeze(0)
        
        def model(data_dict):
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

    
        def train(model, guide, lr=args.svi_lr, n_steps=args.svi_steps):
            pyro.clear_param_store()
            adam_params = {"lr": lr, 'betas': (0.95, 0.999)}
            adam = pyro.optim.Adam(adam_params)
            svi = SVI(model, guide, adam, loss=Trace_ELBO())

            for step in range(n_steps):
                loss = svi.step(data_dict)
                if step % 50 == 0:
                    print('[iter {}]  loss: {:.4f}'.format(step, loss))


        pyro.clear_param_store()
        autoguide_map = pyro.infer.autoguide.AutoLaplaceApproximation(model)
            
        train(model, autoguide_map)

        # q_list = [0.025, 0.05, 0.5, 0.95, 0.975]
        # quantiles = torch.stack(autoguide_map.laplace_approximation(data_dict).quantiles(q_list)['theta']).cpu().numpy()
        posterior = autoguide_map.laplace_approximation(data_dict).get_posterior(data_dict).sample_n(4000)

        d = {}
        for i in range(p):
            d[f'dim {i+1}'] = posterior[:,i]
        df = pd.DataFrame(d)
        df_stats = df.describe(percentiles=[0.025, 0.05, 0.5, 0.95, 0.975]).T
        df_stats.to_csv(output_path, float_format='%.3f')

        # print(torch.stack(autoguide_map.laplace_approximation(data_dict).quantiles(q_list)['theta']))
        # print(autoguide_map.laplace_approximation(data_dict).get_posterior(data_dict).log_prob(theta[0]).detach())
        
        # MAP = autoguide_map.get_posterior(data_dict).rsample().detach()
        # print(autoguide_map.laplace_approximation(data_dict).get_posterior(data_dict).log_prob(MAP).detach())
        # print("Our MAP estimate of the theta is {}".format(MAP.cpu()))
        # MAP = MAP.squeeze().cpu()
        # map_list.append(MAP)
        # dist_list.append((MAP - theta[0]).norm(p=2))

        # print(map_list)
        # print(dist_list)
        ho_model.T = temp

    # plot_vae_decode(vae_model, map_list+[true_theta for true_theta in theta], output_path, mi_estimator, T_to_plot, args.theta_index, p, type='MAP')

