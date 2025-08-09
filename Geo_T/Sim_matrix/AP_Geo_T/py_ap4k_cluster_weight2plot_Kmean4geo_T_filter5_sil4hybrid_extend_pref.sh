#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=pub
#SBATCH --job-name=p-geo_T/5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4geo_T_filter5_silhybrid_extend_pref.py
