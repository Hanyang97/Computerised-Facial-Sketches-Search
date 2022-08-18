
"""
# @Description:
    face finding continuous with larger number of experiments
"""
import argparse

import mlflow
import os
import torch

from experiment_tools.pyro_tools import auto_seed

def plot_n_experiments(T_deploy, ho_model, vae_model, img_path, n_trace=5, theta=None, verbose=True):
    temp, ho_model.T = ho_model.T, T_deploy
    # ho_model.model()
    ho_model.eval_vae(vae_model, img_path, n_trace, theta, verbose)
    ho_model.T = temp   # reset


def eval_deployment(experiment_id, run_id, T_deploy, seed, device, n_trace=5, theta=None, verbose=True):
    seed = auto_seed(seed)
    img_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{T_deploy}_{seed}"
    ho_model = mlflow.pytorch.load_model(
        f'mlruns/{experiment_id}/{run_id}/artifacts/model',
        map_location=device
    )
    vae_model = mlflow.pytorch.load_model(
        f'mlruns/{experiment_id}/{run_id}/artifacts/vae_model',
        map_location=device
    )
    vae_model.eval()
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    plot_n_experiments(T_deploy, ho_model, vae_model, img_path, n_trace, theta, verbose)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Finding Continuous Search with Different T at Deployment."
    )
    parser.add_argument("--experiment-id", default="10", type=str)
    # parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--run-id", default="7d9ff16d7267491a8c88bbcf5f06d3ed", type=str)
    parser.add_argument("--T-deploy", default=300, type=int)
    parser.add_argument("--n-trace", default=2, type=int)

    args = parser.parse_args()

    eval_deployment(
        experiment_id=args.experiment_id,
        run_id=args.run_id, 
        T_deploy=args.T_deploy, 
        seed=args.seed, 
        device=args.device,
        n_trace=args.n_trace,  
        verbose=True
    )
