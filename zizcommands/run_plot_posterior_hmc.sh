#!/bin/bash

# This script is a working example for using Slurm to run GPU tasks.
# All lines starting with "#SBATCH" are not arbitrary comments, Slurm reads and makes use of those parameters.

# We've commented the different options below based on what we understand them to do, 
# but we haven't read the manual closely to make sure it is what they do.

# run dos2unix x.sh to avoid sbatch error
# Usage: `sbatch /data/ziz/not-backed-up/haliu/idad/zizcommands/run_plot_posterior_hmc.sh`

# These settings will give you plenty of emails, you probably want to change them or remove them.
# If you want to try receiving emails enter your email address below.
#SBATCH --mail-user=hanyang.liu@keble.ox.ac.uk
#SBATCH --mail-type=ALL

#SBATCH --job-name=ffceval

#SBATCH --output=/data/ziz/not-backed-up/haliu/idad/zizlogs/slurm-%j.o
#SBATCH --error=/data/ziz/not-backed-up/haliu/idad/zizlogs/slurm-%j.o


#SBATCH --cluster=swan
#SBATCH --partition=high-bigbayes-cpu
#NOTSBATCH --partition=standard-bigbayes-cpu

#NOTSBATCH --nodelist=ziz01.cpu.stats.ox.ac.uk


#SBATCH --cpus-per-task=2

#SBATCH --time=2-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=200G  # RAM (not vRAM), make sure this is enough for your job, otherwise your job will be killed on the spot

#SBATCH --ntasks=1


source /data/ziz/not-backed-up/haliu/idad/new_dad_fs_venv/bin/activate #activate your platform

# Launch your training.

cd /data/ziz/not-backed-up/haliu/idad #cd into your folder

# 2dim: 85d37b20c50c4fb591e795ade881909c
# 5dim: a4e6b9307b774fe1acc28e8e4b8d2b3c    T 80; bin 5; limits -0.5, 1.5; T-step 20
# 10dim: 0b32deb8ec8e489194dcda784f3899b9   T 80; bin 5; limits -0.5 1.5; T-step 20

echo 'Job started.'
python3 plot_locfin_posterior_hmc.py \
    --seed 2 \
    --run-id a4e6b9307b774fe1acc28e8e4b8d2b3c \
    --T-to-plot 60 \
    --bin 5 \
    --limits -0.5 1.5 \
    --T-step 15
echo "Job completed." 