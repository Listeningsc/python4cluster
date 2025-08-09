#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=hpib
#SBATCH --job-name=p-n/f10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4no_geo_filter10_sil4hybrid_regular_pref.py
