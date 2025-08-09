#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=hpib
#SBATCH --job-name=p-geo/10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4geo_filter10_sil4simatrix_regular_pref.py
