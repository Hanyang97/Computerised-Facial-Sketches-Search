#!/bin/bash

# This script is a working example for using Slurm to run GPU tasks.
# All lines starting with "#SBATCH" are not arbitrary comments, Slurm reads and makes use of those parameters.

# We've commented the different options below based on what we understand them to do, 
# but we haven't read the manual closely to make sure it is what they do.

# run dos2unix x.sh to avoid sbatch error
# Usage: `sbatch /data/ziz/not-backed-up/haliu/idad/zizcommands/run_mcmc.sh`

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


#SBATCH --cpus-per-task=1

#SBATCH --time=2-00:00:00  # kills job if it runs for longer than this time, 14 days is the maximum
#SBATCH --mem=50G  # RAM (not vRAM), make sure this is enough for your job, otherwise your job will be killed on the spot

#SBATCH --ntasks=1


source /data/ziz/not-backed-up/haliu/idad/new_dad_fs_venv/bin/activate #activate your platform

# Launch your training.

cd /data/ziz/not-backed-up/haliu/idad #cd into your folder

# 2dim: 85d37b20c50c4fb591e795ade881909c    T30; 
# 5dim: a4e6b9307b774fe1acc28e8e4b8d2b3c    T 60; lr0.05; steps1201
# 10dim: 0b32deb8ec8e489194dcda784f3899b9   T 80; T-step 20; lr0.1; steps2801
# 20dim: 578fbd2180754a08a2cc7be19d09763f   T100; T-step 20; lr0.01; steps12801

# 20dim infoNCE: 973e8209a2284486bb99e158a0b19f1c
# 10dim infoNCE: abdd95880f584f7980258723b5c45452
# 5dim infoNCE: 0d47db9831d04f83968f25f77f799d97

# 5dim T10 0abc55b827e04c3b9fcf28883a5308c7

# 10dim NWJ T50: 5957f83ee6c842528b083d4ba189151d

# 2dim DAD sum: fd674fe1061a49dc8dbba2499a6b8233

# 2dim infonce: 3978677728c64ab1b2dbcbe76eff95ba
# 2dim nwj: aeb8e31099c049bca676528bc328065a

echo 'Job started.'
python3 mcmc.py \
    --seed 1 \
    --run-id a4e6b9307b774fe1acc28e8e4b8d2b3c \
    --T-to-plot 60 \
    --nsample 1000 \
    --nwarmup 400 \
    --nchain 4 \
    --plothist 1
echo "Job completed." 