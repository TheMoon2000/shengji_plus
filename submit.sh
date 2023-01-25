#!/bin/bash

# the SBATCH directives must appear before any executable
# line in this script

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=4 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
##SBATCH --nodelist=steropes # if you need specific nodes
#SBATCH --exclude=blaze,freddie,steropes,como # nodes not yet on SLURM-only
#SBATCH -t 5-00:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /home/eecs/jiarui.shan/shengji-combos
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
#SBATCH -o exps/d95_ablation1_finetune2/%N.%j.out # STDOUT
##SBATCH -e slurm.%N.%j.err # STDERR
# if you want to get emails as your jobs run/fail
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiarui.shan@berkeley.edu # Where to send mail 

# print some info for context
pwd
hostname
date

echo starting job...

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
# source ~/.bashrc
# conda activate my_env

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
python3 -c "import torch; print('There are', torch.cuda.device_count(), 'GPU(s)')"
echo visible devices = $CUDA_VISIBLE_DEVICES

python TrainLoop.py --model-folder exps/d95_ablation1_finetune2 --discount 0.95 --epsilon 0.02 --games 500 --eval-size 300 --dynamic-kitty --oracle-duration 50000 --max-games 700000 --enable-combos

# print completion time
date