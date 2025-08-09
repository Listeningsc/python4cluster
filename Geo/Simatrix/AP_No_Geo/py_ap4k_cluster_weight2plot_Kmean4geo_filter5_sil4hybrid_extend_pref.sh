#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=pub
#SBATCH --job-name=geo_h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4geo_filter5_sil4hybrid_extend_pref.py
