#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=g40423
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --comment=openfold
#SBATCH --exclusive

cat /etc/hosts
source /fsx/home-zanussbaum/bio-chem-lm/env/bin/activate
jupyter notebook --ip=0.0.0.0  
