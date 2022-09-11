#!/bin/bash

# This script is a working example for using Slurm to run GPU tasks.
# All lines starting with "#SBATCH" are not arbitrary comments, Slurm reads and makes use of those parameters.

# We've commented the different options below based on what we understand them to do, 
# but we haven't read the manual closely to make sure it is what they do.

# run dos2unix x.sh to avoid sbatch error
# Usage: `sbatch /data/ziz/not-backed-up/haliu/idad/zizcommands/run_plot_recon_vae.sh`

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
#SBATCH --mem=50G  # RAM (not vRAM), make sure this is enough for your job, otherwise your job will be killed on the spot

#SBATCH --ntasks=1


source /data/ziz/not-backed-up/haliu/idad/new_dad_fs_venv/bin/activate #activate your platform

# Launch your training.

cd /data/ziz/not-backed-up/haliu/idad #cd into your folder

echo 'Job started.'
python3 plot_recon_vae.py
echo "Job completed." 