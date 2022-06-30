#!/bin/bash
#SBATCH --job-name=Rove
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3-00:00:00
#gpu specs
#SBATCH -p gpu --gres=gpu:a100
#Skipping many options! see man sbatch
# From here on, we can start our program

echo "Starting..." > /home/ngw861/out.txt
git log -1 --format="%H" > /home/ngw861/01_abbey_rove/git_commit.txt
source /home/ngw861/venvs/01_abbey_rove/bin/activate
python -m pip install -r /home/ngw861/01_abbey_rove/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
export GIT_PYTHON_REFRESH=quiet
python main.py --dataset=rove --suffix=multisim3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize --augmentation=rove

#gpu:a100
#python main.py --dataset=rove --suffix=npair --source_path=/home/ngw861/01_abbey_rove/data --bs=112 --samples_per_class=2 --loss=npair --batch_mining=npair --arch=resnet50_frozen --augmentation=rove
#python main.py --dataset=rove --suffix=multisim0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize --augmentation=rove
#python main.py --dataset=rove --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6 --augmentation=rove
#python main.py --dataset=rove --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize --augmentation=rove
#python main.py --dataset=rove --suffix=contra0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove
#python main.py --dataset=rove --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove
#python main.py --dataset=rove --suffix=lifted0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen --augmentation=rove
#python main.py --dataset=rove --suffix=proxy0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize --augmentation=rove
