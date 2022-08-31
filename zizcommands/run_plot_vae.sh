#!/bin/bash

# This script is a working example for using Slurm to run GPU tasks.
# All lines starting with "#SBATCH" are not arbitrary comments, Slurm reads and makes use of those parameters.

# We've commented the different options below based on what we understand them to do, 
# but we haven't read the manual closely to make sure it is what they do.

# run dos2unix x.sh to avoid sbatch error
# Usage: `sbatch /data/ziz/not-backed-up/haliu/idad/zizcommands/run_plot_vae.sh`

# These settings will give you plenty of emails, you probably want to change them or remove them.
# If you want to try receiving emails enter your email address below.
# SBATCH --mail-user=hanyang.liu@keble.ox.ac.uk
# SBATCH --mail-type=ALL

#SBATCH --job-name=ffceval

#SBATCH --output=/data/ziz/not-backed-up/haliu/idad/zizlogs/slurm-%j.o
#SBATCH --error=/data/ziz/not-backed-up/haliu/idad/zizlogs/slurm-%j.o


#SBATCH --cluster=swan
#SBATCH --partition=high-bigbayes-cpu
#NOTSBATCH --partition=standard-bigbayes-cpu

#NOTSBATCH --nodelist=ziz01.cpu.stats.ox.ac.uk


#SBATCH --cpus-per-task=1

#SBATCH --time=2-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=20G  # RAM (not vRAM), make sure this is enough for your job, otherwise your job will be killed on the spot

#SBATCH --ntasks=1


source /data/ziz/not-backed-up/haliu/idad/new_dad_fs_venv/bin/activate #activate your platform

# Launch your training.

cd /data/ziz/not-backed-up/haliu/idad #cd into your folder

# 2dim     --vae-model-dir ./vaelogs/VanillaVAE/version_4/checkpoints/epoch=87-step=42151.ckpt \
    # --vae-config-dir ./vaelogs/VanillaVAE/version_4/config.yaml \

# 10dim     --vae-model-dir vaelogs/VanillaVAE/version_2/checkpoints/epoch=79-step=38319.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_2/config.yaml \

# 10dim 0.00005     --vae-model-dir vaelogs/VanillaVAE/version_14/checkpoints/epoch=71-step=34487.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_14/config.yaml \

# 20dim     --vae-model-dir vaelogs/VanillaVAE/version_1/checkpoints/epoch=70-step=34008.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_1/config.yaml \

# 20dim 0.00005     --vae-model-dir vaelogs/VanillaVAE/version_13/checkpoints/epoch=71-step=34487.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_13/config.yaml \

# 50dim     --vae-model-dir vaelogs/VanillaVAE/version_0/checkpoints/epoch=73-step=35445.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_0/config.yaml \

# 128dim    --vae-model-dir vaelogs/VanillaVAE/version_6/checkpoints/epoch=67-step=32571.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_6/config.yaml \

# 50dim 0.00005     --vae-model-dir vaelogs/VanillaVAE/version_8/checkpoints/epoch=81-step=39277.ckpt \
    # --vae-config-dir vaelogs/VanillaVAE/version_8/config.yaml \

echo 'Job started.'
python3 plot_vae.py \
    --device cpu \
    --seed 1 \
    --vae-model-dir vaelogs/VanillaVAE/version_1/checkpoints/epoch=70-step=34008.ckpt \
    --vae-config-dir vaelogs/VanillaVAE/version_1/config.yaml \
    --img-dir ./img1.png
echo "Job completed." 