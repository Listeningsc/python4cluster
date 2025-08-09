import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import scipy.io
import bt_remove
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def remove_outliers(labels: np.ndarray,
                    coords_rad: np.ndarray,
                    pca_feats: np.ndarray,
                    alpha: float = 0.7,
                    coef: float = 3.0):
    """Hybrid‑distance (α·geo + (1−α)·feat) based outlier detection.

    Parameters
    ----------
    剔除异常点
    """
    new_labels = labels.copy()
    outliers = []

    # Pre‑compute full hybrid distance matrix once
    geo_dist = haversine_distances(coords_rad) * 6371  # km
    feat_dist = euclidean_distances(pca_feats)
    sim_matrix = - (alpha * geo_dist**2 + (1 - alpha) * feat_dist**2)
    hybrid_dist = -sim_matrix


    for lbl in np.unique(labels):
        if lbl < 0:
            continue
        idx = np.where(labels == lbl)[0]
        if idx.size < 2:
            continue
        # mean distance to all cluster members ≈ distance to centroid
        dists = hybrid_dist[np.ix_(idx, idx)].mean(axis=1)
        mu, sigma = dists.mean(), dists.std(ddof=0)
        thresh = mu + coef * sigma
        bad = idx[dists > thresh]
        new_labels[bad] = -1
        outliers.extend(bad)

    return new_labels, np.asarray(outliers, dtype=int)

def optimize_affinity_propagation(coords_rad, pca_features, alpha=0.7):
    geo_dist = haversine_distances(coords_rad) * 6371
    feat_dist = euclidean_distances(pca_features)
    hybrid_matrix = - (alpha * geo_dist + (1 - alpha) * feat_dist)
    # 对geo和feat的距离进行归一化
    geo_sim = -geo_dist**2
    geo_sim_norm = (geo_sim - geo_sim.min()) / (geo_sim.max() - geo_sim.min())
    feat_sim = -feat_dist ** 2
    feat_sim_norm = (feat_sim - feat_sim.min()) / (feat_sim.max() - feat_sim.min())

    sim_dist = alpha * -geo_sim + (1 - alpha) * -feat_sim

    percentiles = [25, 40, 50, 60, 75]
    dampings = [0.5, 0.6, 0.7, 0.8, 0.9]

    sil_matrix = np.full((len(percentiles), len(dampings)), np.nan)
    ch_matrix  = np.full_like(sil_matrix, np.nan)

    best_score = -1
    best_labels = None
    best_exemplars = None
    best_config = None

    for i, p in enumerate(percentiles):
        pref = np.percentile(hybrid_matrix, p)
        for j, d in enumerate(dampings):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    ap = AffinityPropagation(
                        affinity='precomputed',
                        preference=pref,
                        damping=d,
                        max_iter=200,
                        random_state=42
                    )
                    ap.fit(hybrid_matrix)
                labels = ap.labels_
                counts = np.bincount(labels)
                valid = np.where(counts >= 5)[0]
                if len(valid) < 2:
                    continue
                # 保留有效簇，其他标 -1
                filtered = np.full_like(labels, -1)
                remap = {old: new for new, old in enumerate(valid)}
                for idx, lab in enumerate(labels):
                    if lab in remap:
                        filtered[idx] = remap[lab]
                mask = filtered >= 0
                sil = silhouette_score(sim_dist[np.ix_(mask, mask)], filtered[mask], metric='precomputed')
                ch  = calinski_harabasz_score(pca_features[mask], filtered[mask])
                sil_matrix[i, j] = sil
                ch_matrix[i, j]  = ch
                if sil > best_score:
                    best_score = sil
                    best_labels = filtered.copy()
                    centers = ap.cluster_centers_indices_
                    best_exemplars = [c for c in centers if labels[c] in valid]
                    best_config = (p, d)
            except Exception:
                continue
    return {
        'best_score':    best_score,
        'best_labels':   best_labels,
        'best_exemplars':best_exemplars,
        'best_config':   best_config,
        'sil_matrix':    sil_matrix,
        'ch_matrix':     ch_matrix,
        'percentiles':   percentiles,
        'dampings':      dampings
    }


def plot_silhouette_heatmap(sil_matrix, percentiles, dampings, title, save_path):
    df = pd.DataFrame(sil_matrix, index=[f"P{p}" for p in percentiles],
                      columns=[f"D{d}" for d in dampings])
    plt.figure(figsize=(6,6))
    sns.heatmap(df, annot=True, fmt=".2f", cbar_kws={'label':'Silhouette Score'})
    plt.title(title)
    plt.xlabel("Damping")
    plt.ylabel("Preference Percentile")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ch_heatmap(ch_matrix, percentiles, dampings, title, save_path):
    df = pd.DataFrame(ch_matrix, index=[f"P{p}" for p in percentiles],
                      columns=[f"D{d}" for d in dampings])
    plt.figure(figsize=(6,6))
    sns.heatmap(df, annot=True, fmt=".0f", cbar_kws={'label':'Calinski–Harabasz Score'})
    plt.title(title)
    plt.xlabel("Damping")
    plt.ylabel("Preference Percentile")
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_year(year_info):
    year, data_dir, output_dir, bt_parent, sic_parent, K,feature_path = year_info
    try:
        # 文件路径
        csv_file = os.path.join(data_dir, f"{year}_multiep_arctic_maxd-based_epsg3413_gama_plus4merge.csv")
        mat_file = os.path.join(data_dir, f"{year}_bt_series_reprocessed_interpolant2epsg3413_sep2April-mid.mat")
        mat_idx  = os.path.join(data_dir, f"{year}_bt2tie_p.mat")

        feats_mat = os.path.join(feature_path,f"{year}_cluster_features.mat")
        # features中的分别是经纬度、海水系点，海水波动，海冰系点、海冰波动、亮温序列均值、亮温序列标准差、亮温序列偏度、亮温序列峰度
        features  = scipy.io.loadmat(feats_mat)['features']

        # 读取与清洗
        tie = pd.read_csv(csv_file, header=None).values
        feats_kcenter = features[:,0:10]
                # 特征及PCA
        coords_rad = np.radians(tie[:, [1,0]])
        # feats = tie[:,[2,6,7,9]]
        scaler   = StandardScaler()
        feats_scaled = scaler.fit_transform(feats_kcenter)
        pca      = PCA(n_components=0.9)
        pca_feats_kcenter= pca.fit_transform(feats_scaled)
        
        
        feats_ap = features[:,2:10]
        feats_ap_scaled = scaler.fit_transform(feats_ap)
        pca      = PCA(n_components=0.9)
        pca_feats_ap= pca.fit_transform(feats_ap_scaled)

        # 初始 KMeans 分组
        init_lbl = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(pca_feats_kcenter)

        # 准备累积
        percentiles = [25, 40, 50, 60, 75]
        dampings    = [0.5, 0.6, 0.7, 0.8, 0.9]
        sil_accum   = np.zeros((len(percentiles), len(dampings)))
        ch_accum    = np.zeros_like(sil_accum)
        count       = np.zeros_like(sil_accum)
        nan_any     = np.zeros_like(sil_accum, dtype=bool)

        group_results=[]
        out_dir = os.path.join(output_dir, str(year))
        os.makedirs(out_dir, exist_ok=True)

        # 分组AP优化
        for gid in range(K):
            mask = init_lbl == gid
            if mask.sum() < 5: continue
            res = optimize_affinity_propagation(coords_rad[mask], pca_feats_ap[mask])
            if res['best_labels'] is None: continue
            # 输出单组热力图
            # plot_silhouette_heatmap(res['sil_matrix'], res['percentiles'], res['dampings'],
            #                         f"{year}_G{gid}_silhouette", os.path.join(out_dir, f"{gid}_sil.png"))
            # plot_ch_heatmap(res['ch_matrix'], res['percentiles'], res['dampings'],
            #                 f"{year}_G{gid}_CH", os.path.join(out_dir, f"{gid}_ch.png"))
            # 累积指标
            sil_accum += np.nan_to_num(res['sil_matrix'])
            ch_accum  += np.nan_to_num(res['ch_matrix'])
            count     += ~np.isnan(res['sil_matrix'])
            nan_any   |= np.isnan(res['sil_matrix'])

            # 保存group结果
            idxs = np.where(mask)[0]
            group_results.append({'ids':idxs, 'labels':res['best_labels'], 'exemplars':res['best_exemplars']})

        # 合并全局标签并剔除小簇(<5)
        full_labels = np.full(len(tie), -1, int)
        offset = 0
        exemplars=[]
        for g in group_results:
            labs = np.unique(g['labels'][g['labels']>=0])
            for new, old in enumerate(labs):
                members = g['ids'][g['labels']==old]
                if len(members) < 5:
                    # 小簇标 -1
                    full_labels[members] = -1
                else:
                    full_labels[members] = offset + new
            for ex in g['exemplars']:
                # 仅记录有效示例点
                if full_labels[g['ids'][ex]] >= 0:
                    exemplars.append((g['ids'][ex], full_labels[g['ids'][ex]]))
            offset += len(labs)

        # full_labels, outliers = remove_outliers(full_labels, coords_rad, pca_feats)

        # 导出聚类中心和簇成员
        centers = [idx for idx, lbl in exemplars if full_labels[idx] >= 0]
        # Log cluster counts per year
        num_clusters = len(np.unique(full_labels[full_labels >= 0]))
        logging.info(f"{year}: {num_clusters} clusters(after outlier pruning)")

        pd.DataFrame(tie[centers]).to_csv(os.path.join(out_dir, f"{year}_AP_centroids.csv"), index=False, header=False)
        for lbl in np.unique(full_labels):
            if lbl < 0: continue
            members = np.where(full_labels==lbl)[0]
            pd.DataFrame(tie[members]).to_csv(os.path.join(out_dir, f"{year}_AP_cluster_{lbl}.csv"), index=False, header=False)
        
        # --- save outliers
        # if outliers.size:
        #     pd.DataFrame(tie[outliers]).to_csv(os.path.join(out_dir, f"{year}_AP_outliers.csv"),
        #                                        index=False, header=False)
        # 计算年度聚合指标（排除-1）
        sil_avg = sil_accum / np.where(count==0, 1, count)
        ch_avg  = ch_accum  / np.where(count==0, 1, count)
        sil_avg[count==0] = np.nan
        ch_avg[count==0] = np.nan

        # 若任一子组缺失则设NaN
        sil_avg[nan_any] = np.nan
        ch_avg[nan_any]  = np.nan

        # 保存年度聚合CSV和热力图
        pd.DataFrame(sil_avg, index=[f"P{p}" for p in percentiles], columns=[f"D{d}" for d in dampings])\
            .to_csv(os.path.join(out_dir, f"{year}_aggregated_sil.csv"))
        pd.DataFrame(ch_avg,  index=[f"P{p}" for p in percentiles], columns=[f"D{d}" for d in dampings])\
            .to_csv(os.path.join(out_dir, f"{year}_aggregated_ch.csv"))
        # plot_silhouette_heatmap(sil_avg, percentiles, dampings, f"{year} Aggregated Silhouette", 
        #                         os.path.join(out_dir, f"{year}_agg_sil.png"))
        # plot_ch_heatmap(ch_avg,        percentiles, dampings, f"{year} Aggregated CH",
        #                 os.path.join(out_dir, f"{year}_agg_ch.png"))
        
        # 7. 计算本年最优参数组合
        if np.all(np.isnan(sil_avg)):
            best_p, best_d, best_sil = np.nan, np.nan, np.nan
        else:
            idx_max = np.nanargmax(sil_avg)
            bi, bj = np.unravel_index(idx_max, sil_avg.shape)
            best_p = percentiles[bi]
            best_d = dampings[bj]
            best_sil = sil_avg[bi, bj]

        # 保存本年最优参数
        df_best = pd.DataFrame([{
            'percentile': best_p,
            'damping': best_d,
            'silhouette': best_sil
        }])
        df_best.to_csv(os.path.join(out_dir, f"{year}_best_params.csv"), index=False)

        return year, full_labels, sil_avg, ch_avg,(best_p, best_d, best_sil)
    except Exception as e:
        logging.error(f"{year} failed: {e}")
        return year, None, None, None


if __name__ == '__main__':
    start_time = time.time()
    father_path = r'/project/lijiaxing/data_source/bt_data/L1C2epsg3413'
    output_path = r'/scratch/lijiaxing/output/python_output/cluster/AP4Kcenter/grid_search/Geo/Hybrid_matrix/Another_pref/AP_No_Geo/sil4simatrix_f5_r_pref'
    feature_path = r'/project/lijiaxing/data_source/cluster/features'
    bt_parent   = r'/project/lijiaxing/data_source/bt_data/'
    sic_parent  = r'/project/lijiaxing/data_source/SIC'
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path,'clustering_log.txt'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    years = list(range(2010,2024))
    ks    = [9,9,9,9,8,9,9,9,8,9,9,9,9,9]
    args  = [(years[i], father_path, output_path, bt_parent, sic_parent, ks[i],feature_path) for i in range(len(years))]

    with Pool(processes=7) as pool:
        results = pool.map(process_year, args)

    years_done, full_labels_list, sil_avgs, ch_avgs, bests = zip(*results)
    df_bp = pd.DataFrame({
        'year': years_done,
        'best_percentile': [b[0] for b in bests],
        'best_damping':    [b[1] for b in bests],
        'best_silhouette':[b[2] for b in bests]
    })
    df_bp.to_csv(os.path.join(output_path, 'best_params_per_year.csv'), index=False)

    # 打印总耗时
    total = time.time() - start_time
    print(f"Total time: {total:.2f} seconds")
    logging.info(f"Total time: {total:.2f} seconds")
