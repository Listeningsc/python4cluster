#!/bin/bash
#SBATCH --account=xiaofeng4
#SBATCH --partition=pub
#SBATCH --job-name=nogeo10r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7


python batch_parallel_ap4k_cluster_based_weight2plot_Kmean4no_geo_filter10_sil4hybrid_extend_pref_recalculate.py
