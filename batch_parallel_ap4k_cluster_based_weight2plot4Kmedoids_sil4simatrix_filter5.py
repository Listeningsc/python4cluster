'''该代码基于原有kmedoids聚类结果，在每个子类中进行AP聚类，再最后根据AP聚类结果，得出全年整体的CH和轮廓系数，并绘制年度与跨年度热力图。'''
import os
import time
import numpy as np
import pandas as pd
import scipy.io
import bt_remove
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import Pool
from sklearn.cluster import AffinityPropagation
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
    sim_matrix = - (alpha * geo_dist**2 + (1 - alpha) * feat_dist**2)
    hybrid_dist = -sim_matrix
    
    percentiles = [25, 40, 50, 60, 75]
    dampings    = [0.5, 0.6, 0.7, 0.8, 0.9]
    sil_matrix = np.full((len(percentiles), len(dampings)), np.nan)
    ch_matrix  = np.full_like(sil_matrix, np.nan)

    best_score     = -1
    best_labels    = None
    best_exemplars = None
    best_config    = None

    for i, p in enumerate(percentiles):
        pref = np.percentile(sim_matrix, p)
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
                    ap.fit(sim_matrix)

                raw_labels = ap.labels_
                counts = np.bincount(raw_labels)
                valid = np.where(counts >= 5)[0]
                if len(valid) < 2:
                    continue
                filt_lbl = np.full_like(raw_labels, -1)
                for idx, lab in enumerate(raw_labels):
                    if lab in valid:
                        filt_lbl[idx] = np.where(valid == lab)[0][0]

                mask = filt_lbl >= 0
                # hybrid_sub = (
                #     geo_dist[np.ix_(mask, mask)] * alpha +
                #     feat_dist[np.ix_(mask, mask)] * (1-alpha)
                # )
                sil = silhouette_score(hybrid_dist[np.ix_(mask, mask)], filt_lbl[mask], metric='precomputed')
                ch  = calinski_harabasz_score(pca_features[mask], filt_lbl[mask])

                sil_matrix[i, j] = sil
                ch_matrix[i, j]  = ch

                if sil > best_score:
                    best_score     = sil
                    best_labels    = filt_lbl.copy()
                    best_exemplars = [c for c in ap.cluster_centers_indices_ if raw_labels[c] in valid]
                    best_config    = (p, d)
            except Exception:
                continue

    return {
        'best_score':     best_score,
        'best_labels':    best_labels,
        'best_exemplars': best_exemplars,
        'best_config':    best_config,
        'sil_matrix':     sil_matrix,
        'ch_matrix':      ch_matrix,
        'percentiles':    percentiles,
        'dampings':       dampings
    }


def plot_heatmap(matrix, percentiles, dampings, title, save_path, fmt=".2f", label='Score'):
    df = pd.DataFrame(matrix,
                      index=[f"P{p}" for p in percentiles],
                      columns=[f"D{d}" for d in dampings])
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=fmt, cbar_kws={'label': label})
    plt.title(title)
    plt.xlabel("Damping")
    plt.ylabel("Preference Percentile")
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_year(year_info):
    year, data_dir, output_path, bt_parent, sic_parent, cluster_dir = year_info
    year_out = os.path.join(output_path, str(year))
    os.makedirs(year_out, exist_ok=True)

    # 1. 读取 tie-point 与 BT
    csv_file = os.path.join(data_dir, f"{year}_multiep_arctic_maxd-based_epsg3413_gama_plus4merge.csv")
    mat_file = os.path.join(data_dir, f"{year}_bt_series_reprocessed_interpolant2epsg3413_sep2April-mid.mat")
    mat_inx_file = os.path.join(data_dir, f"{year}_bt2tie_p.mat")

    tie_content = np.array(pd.read_csv(csv_file, header=None))
    bt_series = scipy.io.loadmat(mat_file)['bt_series']
    bt2tie = scipy.io.loadmat(mat_inx_file)['bt2tie_inx']
    bt2tie_inx = (bt2tie > 0).flatten()
    tie_bt_series = bt_series[bt2tie_inx, :]

    # 2. 清洗与特征预处理
    sic_fmt    = '.hdf' if year==2010 else '.nc' if year==2011 else '.he5'
    rm_idx     = bt_remove.bt_remove_index(os.path.join(sic_parent, f"{year}-{year+1}"), bt_parent, sic_fmt)
    remove_arr = np.array(rm_idx, dtype=bool)
    #  True 表示剔除，所以要取反来保留对应列
    keep_mask  = ~remove_arr  

    # 4. 应用布尔索引剔除列
    cleaned_bt_series = tie_bt_series[:, keep_mask]

    df_bt = pd.DataFrame(cleaned_bt_series)
    df_bt = df_bt.bfill().ffill()
    df_bt = df_bt.fillna(df_bt.mean())
    if df_bt.isna().any().any():
        raise ValueError(f"{year}: NaNs remain in tie_bt_series after all filling attempts")
    tie2bt_series = df_bt.values

    # 特征工程
    means = np.mean(tie2bt_series[:, 2:], axis=1)
    stds = np.std(tie2bt_series[:, 2:], axis=1)
    skewnesses = pd.DataFrame(tie2bt_series[:, 2:]).skew(axis=1).values
    kurtoses = pd.DataFrame(tie2bt_series[:, 2:]).kurtosis(axis=1).values
    other_features = np.column_stack([
        means, stds, skewnesses, kurtoses,
        tie_content[:, [2, 6, 7, 9]]
    ])
    
    scaler     = StandardScaler()
    pca_feats  = PCA(n_components=0.9).fit_transform(scaler.fit_transform(other_features))
    coords_rad = np.radians(tie_content[:, [1,0]])

    # 3. 初始化年度累加
    percentiles = [25, 40, 50, 60, 75]
    dampings    = [0.5, 0.6, 0.7, 0.8, 0.9]
    sil_accum = np.zeros((len(percentiles), len(dampings)))
    ch_accum  = np.zeros_like(sil_accum)
    count     = np.zeros_like(sil_accum)
    nan_any     = np.zeros_like(sil_accum, dtype=bool)

    # 4. AP细化 K-medoids 子簇
    km_dir = os.path.join(cluster_dir, str(year))
    prefix = f"{year}_kmedoids_best_cluster_"
    group_results = []

    for fn in os.listdir(km_dir):
        if not (fn.startswith(prefix) and fn.endswith('.csv')):
            continue
        df_cl = pd.read_csv(os.path.join(km_dir, fn), header=None).values
        ids = np.array([np.argmin(haversine_distances(np.radians([[lat, lon]]), coords_rad).flatten())
                        for lon, lat in df_cl[:, :2]], int)

        res = optimize_affinity_propagation(coords_rad[ids], pca_feats[ids])
        if res['best_labels'] is None:
            continue
        cid = int(fn.replace(prefix, '').replace('.csv',''))

        # 保存子簇热力图
        plot_heatmap(res['sil_matrix'], res['percentiles'], res['dampings'],
                     f"{year}_Cluster{cid}_Silhouette", 
                     os.path.join(year_out, f"cluster{cid}_sil.png"), fmt=".2f", label='Silhouette')
        plot_heatmap(res['ch_matrix'], res['percentiles'], res['dampings'],
                     f"{year}_Cluster{cid}_CH",
                     os.path.join(year_out, f"cluster{cid}_ch.png"), fmt=".0f", label='CH')

        sil_accum += np.nan_to_num(res['sil_matrix'])
        ch_accum  += np.nan_to_num(res['ch_matrix'])
        count     += ~np.isnan(res['sil_matrix'])
        nan_any   |= np.isnan(res['sil_matrix'])
        group_results.append({'ids':ids, 'labels':res['best_labels'], 'exemplars':res['best_exemplars']})

    # 5. 合并 & 导出示例点与簇成员
    full_labels = np.full(len(tie_content), -1, int)
    offset = 0
    exemplars = []
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

    full_labels, outliers = remove_outliers(full_labels,coords_rad, pca_feats)

    # 导出聚类中心和簇成员
    centers = [idx for idx, lbl in exemplars if full_labels[idx] >= 0]
    # Log cluster counts per year
    num_clusters = len(np.unique(full_labels[full_labels >= 0]))
    logging.info(f"{year}: {num_clusters} clusters(after outlier pruning)")

    pd.DataFrame(tie_content[centers]).to_csv(
        os.path.join(year_out, f"{year}_AP_centroids.csv"), index=False, header=False)
    for lbl in np.unique(full_labels):
        if lbl<0: continue
        members = np.where(full_labels==lbl)[0]
        pd.DataFrame(tie_content[members]).to_csv(
            os.path.join(year_out, f"{year}_AP_cluster_{lbl}.csv"), index=False, header=False)
        
    # --- save outliers
    if outliers.size:
        pd.DataFrame(tie_content[outliers]).to_csv(os.path.join(year_out, f"{year}_AP_outliers.csv"),
                                            index=False, header=False)

    # 6. 年度聚合指标 & 单年度热力图输出
    sil_avg = sil_accum / np.where(count>0, count, 1)
    ch_avg  = ch_accum  / np.where(count>0, count, 1)
    sil_avg[count==0] = np.nan
    ch_avg[count==0]  = np.nan

    # 若任一子组缺失则设NaN
    sil_avg[nan_any] = np.nan
    ch_avg[nan_any]  = np.nan

    pd.DataFrame(sil_avg, index=[f"P{p}" for p in percentiles], columns=[f"D{d}" for d in dampings]) \
        .to_csv(os.path.join(year_out, f"{year}_aggregated_sil.csv"))
    pd.DataFrame(ch_avg,  index=[f"P{p}" for p in percentiles], columns=[f"D{d}" for d in dampings]) \
        .to_csv(os.path.join(year_out, f"{year}_aggregated_ch.csv"))

    plot_heatmap(sil_avg, percentiles, dampings, f"{year}_Aggregated_Silhouette",
                 os.path.join(year_out, f"{year}_agg_sil.png"), fmt=".2f", label='Silhouette')
    plot_heatmap(ch_avg, percentiles, dampings, f"{year}_Aggregated_CH",
                 os.path.join(year_out, f"{year}_agg_ch.png"), fmt=".0f", label='CH')
    
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
    df_best.to_csv(os.path.join(year_out, f"{year}_best_params.csv"), index=False)

    return year, sil_avg, ch_avg,(best_p, best_d, best_sil)

if __name__ == '__main__':
    start_time = time.time()
    father_path     = r'/project/lijiaxing/data_source/bt_data/L1C2epsg3413'
    data_dir        = father_path.replace('Harmonic_Output', 'processed')
    output_path     = r'/project/lijiaxing/output/python_output/cluster/AP4Kmedoids'
    bt_parent_path  = r'/project/lijiaxing/data_source/bt_data'
    sic_parent_path = r'/project/lijiaxing/data_source/SIC'
    cluster_dir     = r'/project/lijiaxing/data_source/cluster/kmedoids'
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, 'ap_refine.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    years = list(range(2010,2024))
    args  = [(yr, father_path, output_path, bt_parent_path, sic_parent_path, cluster_dir) for yr in years]
    with Pool(processes=7) as pool:
        results = pool.map(process_year, args)

    # Summary
    years_done, sils, chs, bests = zip(*results)

    df_bp = pd.DataFrame({
        'year': years_done,
        'best_percentile': [b[0] for b in bests],
        'best_damping':    [b[1] for b in bests],
        'best_silhouette':[b[2] for b in bests]
    })
    df_bp.to_csv(os.path.join(output_path, 'best_params_per_year.csv'), index=False)

    print("Summary:")
    for y, sil in zip(years_done, sils):
        logging.info(f"{y}: silhouette grid computed")
        print(f"{y}: silhouette and CH computed")

    total = time.time() - start_time
    print(f"Total time: {total:.2f} seconds")
    logging.info(f"Total time: {total:.2f} seconds")
