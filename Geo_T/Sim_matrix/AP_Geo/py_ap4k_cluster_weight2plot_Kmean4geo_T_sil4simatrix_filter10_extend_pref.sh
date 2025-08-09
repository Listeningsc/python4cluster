#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=hpib
#SBATCH --job-name=ap-gT/f10/e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4geo_T_sil4simatrix_filter10_extend_pref.py
