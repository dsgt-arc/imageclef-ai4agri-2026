#!/bin/bash
#SBATCH --job-name=vit_80ep_rtx6000_60hl
#SBATCH --account=paceship-clef2026_img_ai4agri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --constraint=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:8    
#SBATCH --qos=inferno
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

#SBATCH --mail-type=ALL             # Email notifications
#SBATCH --mail-user=lrassbach3@gatech.edu

# Activate environment
source ~/ps-clef2026_img_ai4agri-0/.venv/bin/activate

# set up tmp
export TMPDIR=~/scratch/lrassbach3_job
mkdir -p "$TMPDIR"

# Run your script
python ~/ps-clef2026_img_ai4agri-0/lrassbach3/imageclef-ai4agri-2026/user/lrassbach3/fetch_data.py

python ~/ps-clef2026_img_ai4agri-0/lrassbach3/imageclef-ai4agri-2026/user/lrassbach3/AIForAgri_ViT_patches.py
