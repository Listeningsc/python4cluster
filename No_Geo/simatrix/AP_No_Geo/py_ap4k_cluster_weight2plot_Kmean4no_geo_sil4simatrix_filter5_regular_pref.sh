#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=hpib
#SBATCH --job-name=ap_ng5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4no_geo_filter5_sil4simatrix_regular_pref.py
