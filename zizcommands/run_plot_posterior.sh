#!/bin/bash

# This script is a working example for using Slurm to run GPU tasks.
# All lines starting with "#SBATCH" are not arbitrary comments, Slurm reads and makes use of those parameters.

# We've commented the different options below based on what we understand them to do, 
# but we haven't read the manual closely to make sure it is what they do.

# run dos2unix x.sh to avoid sbatch error
# Usage: `sbatch /data/ziz/not-backed-up/haliu/idad/zizcommands/run_plot_posterior.sh`

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
#SBATCH --mem=250G  # RAM (not vRAM), make sure this is enough for your job, otherwise your job will be killed on the spot

#SBATCH --ntasks=1


source /data/ziz/not-backed-up/haliu/idad/new_dad_fs_venv/bin/activate #activate your platform

# Launch your training.

cd /data/ziz/not-backed-up/haliu/idad #cd into your folder

# 2dim infoNCE: bae74bbb51dc46739a02edee44d27970    100; -1.5, 1.5; 50
# 2dim NWJ: aeb8e31099c049bca676528bc328065a    100; -1.5, 1.5; 50  2 seconds
# 5dim infoNCE: 0d47db9831d04f83968f25f77f799d97    11; -1 1; 80
# 5dim NWJ: 6f9d74a484924b10a8787a47668405d1    21; -1 1; 80    18min
# 10dim infoNCE: abdd95880f584f7980258723b5c45452   7; -0.25, 1.25; 100     25min
# 10dim NWJ: 5957f83ee6c842528b083d4ba189151d   7; -0.25, 1.25; 100
# 20dim infoNCE: 973e8209a2284486bb99e158a0b19f1c   3; -0.25, 1.25; 120
# 20dim NWJ: 910d0f503bb34245bd0171c17a8197f0   

echo 'Job started.'
python3 plot_locfin_posterior.py \
    --seed 1 \
    --run-id 973e8209a2284486bb99e158a0b19f1c \
    --bin 3 \
    --limits -0.25 1.25 \
    --T 120
echo "Job completed." 