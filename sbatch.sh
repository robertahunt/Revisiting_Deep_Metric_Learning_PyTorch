#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=Rove
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=6000M
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3-00:00:00
#gpu specs
#SBATCH -p gpu --gres=gpu:titanrtx:1
#Skipping many options! see man sbatch
# From here on, we can start our program

#[ -d /home ] && echo "/home directory exists."
#[ -d /home/ngw861 ] && echo "/home/ngw861 directory exists."
#[ -d /home/ngw861/venv ] && echo "/home/ngw861/venv directory exists."
#[ -d /home/ngw861/venv/01_abbey_rove/bin ] && echo "/home/ngw861/venv/01_abbey_rove directory exists."
#[ -d /home/ngw861/01_abbey_rove ] && echo "/home/ngw861/01_abbey_rove directory exists."
#echo "$(ls -l /home/ngw861/venv/01_abbey_rove/bin)"

#echo "Starting..." > /home/ngw861/out.txt
#git log -1 --format="%H" > /home/ngw861/01_abbey_rove/ python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#git_commit.txt
module load python/3.9.9
source /home/ngw861/venvs/01_abbey_rove/bin/activate
echo "$(nvidia-smi)"

#python3 -m pip install pip --upgrade
#python3 -m pip install -r /home/ngw861/01_abbey_rove/requirements_clust.txt -f https://download.pytorch.org/whl/torch_stable.html
#export GIT_PYTHON_REFRESH=quiet
#echo "$(which python3)"
#echo "$(whereis python3)"

##### RoveGenus #####

### PROXY ###
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=proxy4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize

### LIFTED ###
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=lifted0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=lifted1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=lifted2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=lifted3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=lifted4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen

### CONTRA ###
python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=contra0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=contra1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=contra2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=contra3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=contra4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize

### ARCFACE ###
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=arcface1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=arcface2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=arcface3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=arcface4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize


### MARGIN06 ###
#python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=margin06D1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=margin06D2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=margin06D3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=roveGenus  --augmentation=rove --use_tv_split --suffix=margin06D4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6


###  TRIPLET  ###
#python main.py --dataset=roveGenus --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0  --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=roveGenus --suffix=tripD1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1  --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=roveGenus --suffix=tripD2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2  --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=roveGenus --suffix=tripD3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3  --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=roveGenus --suffix=tripD4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4  --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split

### MULTISIM
#python3 main.py --dataset=roveGenus  --augmentation=rove --suffix=multisim0 --use_tv_split --source_path=/home/ngw861/01_abbey_rove/data --seed=0  --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --suffix=multisim1 --use_tv_split --source_path=/home/ngw861/01_abbey_rove/data --seed=1  --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --suffix=multisim2 --use_tv_split --source_path=/home/ngw861/01_abbey_rove/data --seed=2  --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --suffix=multisim3 --use_tv_split --source_path=/home/ngw861/01_abbey_rove/data --seed=3  --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python3 main.py --dataset=roveGenus  --augmentation=rove --suffix=multisim4 --use_tv_split --source_path=/home/ngw861/01_abbey_rove/data --seed=4  --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize

##### CUB200 BS 112 #####

### MULTISIM
#python main.py --dataset=cub200 --suffix=multisim0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=multisim1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=multisim2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=multisim3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=multisim4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize

###### CUB200 ######
### PROXY ###
#python main.py --dataset=cub200  --suffix=proxy0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=proxy1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=proxy2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=proxy3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=proxy4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize

### LIFTED ###
#python main.py --dataset=cub200 --suffix=lifted0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=cub200 --suffix=lifted1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=cub200 --suffix=lifted2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=cub200 --suffix=lifted3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=cub200 --suffix=lifted4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen

### CONTRA ###
#python main.py --dataset=cub200 --suffix=contra0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=contra1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=contra2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=contra3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=contra4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize

### ARCFACE ###
#python main.py --dataset=cub200  --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200  --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize

### MARGIN06 ###
#python main.py --dataset=cub200 --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=cub200 --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=cub200 --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=cub200 --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=cub200 --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6

###  TRIPLET 
#python main.py --dataset=cub200 --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=tripD1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=tripD2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=tripD3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=cub200 --suffix=tripD4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize



###### INAT ######

#python main.py --dataset=inaturalist_mini --suffix=multisim0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=inaturalist_mini --suffix=multisim1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=inaturalist_mini --suffix=multisim2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=inaturalist_mini --suffix=multisim3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=inaturalist_mini --suffix=multisim4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize




###### ROVE ######
### PROXY ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=proxy0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=proxy1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=proxy2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=proxy3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=proxy4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=proxynca --arch=resnet50_frozen_normalize

### LIFTED ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=lifted0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=lifted1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=lifted2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=lifted3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=lifted4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=lifted --batch_mining=lifted --arch=resnet50_frozen

### CONTRA ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=contra0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=contra1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=contra2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=contra3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=contra4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=contrastive --batch_mining=distance --arch=resnet50_frozen_normalize

### ARCFACE ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=arcface0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=arcface1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=arcface2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=arcface3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=arcface4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=arcface --arch=resnet50_frozen_normalize

### MARGIN06 ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=margin06D0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=margin06D1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=margin06D2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=margin06D3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=margin06D4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=margin --batch_mining=distance --arch=resnet50_frozen_normalize --loss_margin_beta=0.6

### MULTISIM ###
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=multisim0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=multisim1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=multisim2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=multisim3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize
#python main.py --dataset=rove  --augmentation=rove --use_tv_split --suffix=multisim4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2 --loss=multisimilarity --arch=resnet50_frozen_normalize

###  TRIPLET  ###
#python main.py --dataset=rove --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=rove --suffix=tripD1 --source_path=/home/ngw861/01_abbey_rove/data --seed=1 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=rove --suffix=tripD2 --source_path=/home/ngw861/01_abbey_rove/data --seed=2 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=rove --suffix=tripD3 --source_path=/home/ngw861/01_abbey_rove/data --seed=3 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
#python main.py --dataset=rove --suffix=tripD4 --source_path=/home/ngw861/01_abbey_rove/data --seed=4 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split


