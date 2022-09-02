# Computerised Facial Sketches Search Using (Implicit) Deep Adaptive Design
This code supports the Master's Dissertation 'Guided Computerised Facial Sketches Search Using Deep Adaptive Design' in Oxford Statistics Department 2022.

For DAD and iDAD, please refer to the following

```
@article{foster2021deep,
  title={Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design},
  author={Foster, Adam and Ivanova, Desi R and Malik, Ilyas and Rainforth, Tom},
  journal={arXiv preprint arXiv:2103.02438},
  year={2021}
}
```

```
@article{ivanova2021implicit,
  title={Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods},
  author={Ivanova, Desi R. and Foster, Adam and Kleinegesse, Steven and Gutmann, Michael and Rainforth, Tom},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

The VAE models are adapted from the PyTorch VAE Repository with some changes. We only used Vanilla VAEs

```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}
```

Some plotter functions are adapted from last year's dissertation

```
https://github.com/chris142857/dad_final
```


## Computing infrastructure requirements
We have tested this codebase on Linux (Ubuntu x86_64) with Python 3.8.
To train DAD and iDAD networks, we recommend the use of a GPU. We used one GeForce RTX 2080Ti GPU on a machine with 11 GiB of GPU memory.

## Installation
1. Ensure that Python and `venv` are installed.
1. Create and activate a new `venv` virtual environment as follows
```bash
python3 -m venv cfs_venv
source cfs_venv/bin/activate
```
1. Install the package requirements using `pip install -r requirements.txt`.

## Dataset
To download the dataset, use `sh get_dataset.sh`.

## MLFlow
We use `mlflow` to log metric and store network parameters. Each experiment run is stored in
a directory `mlruns` which will be created automatically. Each experiment is assigned a
numerical `<ID>` and each run gets a unique `<HASH>`. The DAD and iDAD networks will be saved in
`./mlruns/<ID>/<HASH>/artifacts`, which will be printed at the end of each training run. 
We can also check 5 examples of simulations in `./mlruns/<ID>/<HASH>/artifacts`.

## Tensorboard
To log VAEs, we use tensorboard.

## VAE
To train VAE, change the yaml in `./configs` folder. Note we only tested the `vae.yaml` file. 
If other vaes do not work, please refer to `https://github.com/AntixK/PyTorch-VAE`. 
You may need to check the `vae_run` file (I added latent_dim and kld_weight arguments) 
and `dataset.py` file (I added my own dataloader) if you use the default configs.
```bash
python3 vae_run.py \
    -c configs/vae.yaml \
    --latent-dim 50 \
    --kld-weight 0.00005
```

## Location Finding Experiment

To train a DAD network without attention, execute the command
```bash
python3 location_finding.py \
    --num-steps 10000 \
    --physical-dim 2 \
    --num-sources 1 \
    --lr 1e-5 \
    --num-experiments <10 or 30 or 50> \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator sPCE \
    --device cuda:0 \
    --design-arch sum \
    --mlflow-experiment-name locfin \
    --ckpt-path <vaelog model checkpoint path> \
    --vaeconfig-path <vaelog model config path>
```

To train a DAD network with attention, execute the command
```bash
python3 location_finding.py \
    --num-steps 10000 \
    --physical-dim 2 \
    --num-sources 1 \
    --lr 1e-5 \
    --num-experiments 50 \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator sPCE \
    --device cuda:0 \
    --mlflow-experiment-name locfin \
    --ckpt-path <vaelog model checkpoint path> \
    --vaeconfig-path <vaelog model config path>
```

To train an iDAD network with the InfoNCE bound, execute the command
```bash
python3 location_finding.py \
    --num-steps 20000 \
    --physical-dim 2 \
    --num-sources 1 \
    --lr 5e-5 \
    --num-experiments 50 \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator InfoNCE \
    --device cuda:0 \
    --mlflow-experiment-name locfin \
    --ckpt-path <vaelog model checkpoint path> \
    --vaeconfig-path <vaelog model config path>
```

To train an iDAD network with the NWJ bound, execute the command
```bash
python3 location_finding.py \
    --num-steps 50000 \
    --physical-dim 2 \
    --num-sources 1 \
    --lr 1e-5 \
    --num-experiments 50 \
    --encoding-dim 64 \
    --hidden-dim 512 \
    --mi-estimator NWJ \
    --device cuda:0 \
    --mlflow-experiment-name locfin \
    --ckpt-path <vaelog model checkpoint path> \
    --vaeconfig-path <vaelog model config path>
```

To evaluate all the resulting networks in the `experiment-id`, execute the command
```bash
python3 eval_sPCE.py --experiment-id <ID>
```

To plot the 2 dimensional search with HMC and MAP by pyro.svi, (need update run_id in `plot_locfin_posterior_hmc_2dim_pyrosvi.py`) execute the command
```
python3 plot_locfin_posterior_hmc_2dim_pyrosvi.py \
    --mi-estimator <infoNCE, NWJ, PCE or PCE_Attention> \
    --theta-index 2
```

To plot the 2 dimensional search with HMC and MAP by critics (grid), (need update run_id in `plot_locfin_posterior_2dim.py`) execute the command
```
python3 plot_locfin_posterior_2dim.py 
```

To plot MAP images for higher dimensions, execute the command
```bash
python3 plot_locfin_posterior_pyrosvi.py \
    --seed 1 \
    --run-id <ID> \
    --T-to-plot 80 \
    --T-step 20 \
    --svi-lr 0.005 \
    --svi-steps 6201
```

To run NUTS and evaluate posterior, execute the command
```bash
python3 mcmc.py \
    --seed 1 \
    --run-id <ID> \
    --T-to-plot 60 \
    --nsample 1000 \
    --nwarmup 400 \
    --nchain 4 \
    --plothist 1
```

To run Laplace Approximation and evaluate posterior, execute the command
```bash
python3 laplace.py \
    --seed 1 \
    --run-id <ID> \
    --T-to-plot 30 \
    --svi-lr 0.01 \
    --svi-steps 2801
```

For more commands details, you can refer to `./zizcommands`.
