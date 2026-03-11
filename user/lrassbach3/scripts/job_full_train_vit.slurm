#!/bin/bash
#SBATCH --job-name=vit_full_training_6k
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1    
#SBATCH --qos=inferno
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

#SBATCH --mail-type=ALL             # Email notifications
#SBATCH --mail-user=lrassbach3@gatech.edu

# Activate environment
source ~/ps-clef2026_img_ai4agri-0/.venv/bin/activate

# Run your script
python ~/ps-clef2026_img_ai4agri-0/lrassbach3/imageclef-ai4agri-2026/user/lrassbach3/AIForAgri_ViT_patches.py > ~/ps-clef2026_img_ai4agri-0/lrassbach3/imageclef-ai4agri-2026/user/lrassbach3/models/ViT/output/slurm_6k.out
