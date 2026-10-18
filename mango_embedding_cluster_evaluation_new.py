# Mango disease embedding, clustering, and cluster-aware evaluation
# Edit the paths in CONFIGURATION before running.
# Dependencies: numpy pandas matplotlib scikit-learn umap-learn hdbscan joblib

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, adjusted_rand_score,
    normalized_mutual_info_score, silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------- CONFIGURATION -------------------------
ROOT = Path('.')
EMBEDDINGS_DIR = ROOT / 'Saved_Embeddings'
OUTPUT_DIR = ROOT / 'Embedding_Cluster_Evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLUSTERS = 12                 # Must match the K used in your chosen K-means analysis
ATYPICAL_PERCENTILE = 95        # Threshold is calibrated on validation only
MIN_CLUSTER_SIZE = 20           # HDBSCAN sensitivity parameters; report these
MIN_SAMPLES = 5
MAX_SILHOUETTE_SAMPLES = 5000
N_REPRESENTATIVES = 5

# Change filenames here if yours differ. Frozen ViT filenames match prior context.
EXPERIMENTS = {
    'Frozen_ViT': {
        'train': EMBEDDINGS_DIR / 'train_vit_embeddings.npz',
        'validation': EMBEDDINGS_DIR / 'validation_vit_embeddings.npz',
        'test': EMBEDDINGS_DIR / 'test_vit_embeddings.npz',
    },
    'Fine_tuned_ViT': {
        'train': EMBEDDINGS_DIR / 'FT_train_fine_vit_embeddings.npz',
        'validation': EMBEDDINGS_DIR / 'FT_validation_fine_vit_embeddings.npz',
        'test': EMBEDDINGS_DIR / 'test_fine_vit_embeddings.npz',
        # Old names, can be changed back to "train_fine...etc. if embeddings run again
    },
    'ResNet50': {
        'train': EMBEDDINGS_DIR / 'train_resnet50_embeddings.npz',
        'validation': EMBEDDINGS_DIR / 'validation_resnet50_embeddings.npz',
        'test': EMBEDDINGS_DIR / 'test_resnet50_embeddings.npz',
    },
    'CAE': {
        'train': EMBEDDINGS_DIR / 'train_cae_embeddings.npz',
        'validation': EMBEDDINGS_DIR / 'validation_cae_embeddings.npz',
        'test': EMBEDDINGS_DIR / 'test_cae_embeddings.npz',
    },
}

# ------------------------- DATA LOADING -------------------------
def get_key(npz, names, required=False):
    for name in names:
        if name in npz.files:
            return name
    if required:
        raise KeyError(f'Missing one of {names}. Available NPZ keys: {npz.files}')
    return None


def load_split(path, split, experiment):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing {experiment}/{split}: {path}. Edit EXPERIMENTS.')
    with np.load(path, allow_pickle=True) as z:
        X = np.asarray(z[get_key(z, ['embeddings', 'embedding', 'X'], True)], dtype=np.float32)
        n = len(X)
        def get(names, fallback):
            k = get_key(z, names)
            if k is None:
                return np.repeat(fallback, n)
            a = np.asarray(z[k])
            if len(a) != n:
                raise ValueError(f'{path}: {k} length does not match embeddings')
            return a
        df = pd.DataFrame({
            'path': get(['path', 'paths', 'image_path'], '').astype(str),
            'class_id_original': get(['class_id', 'labels', 'label'], -1),
            'Class': get(['class_name', 'Class', 'class'], 'Unknown').astype(str),
            'Category': get(['category', 'Category'], 'Unknown').astype(str),
            'Plant_Part': get(['plant_part', 'Plant_Part', 'plantPart'], 'Unknown').astype(str),
            'split': split,
            'experiment': experiment,
        })
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f'{path}: embeddings must be a finite 2-D array; shape={X.shape}')
    return X, df


def load_experiment(name, paths):
    loaded = {s: load_split(paths[s], s, name) for s in ['train', 'validation', 'test']}
    dims = {loaded[s][0].shape[1] for s in loaded}
    if len(dims) != 1:
        raise ValueError(f'{name}: embedding dimensions differ: {dims}')
    X_all = np.vstack([loaded[s][0] for s in ['train', 'validation', 'test']])
    meta = pd.concat([loaded[s][1] for s in ['train', 'validation', 'test']], ignore_index=True)
    if (meta['Class'] == 'Unknown').all():
        raise ValueError(f'{name}: NPZ must include class_name or Class')
    nonempty = meta.loc[meta.path.ne(''), 'path']
    if nonempty.duplicated().any():
        print(f'[WARNING] {name}: duplicate image paths across splits: {int(nonempty.duplicated().sum())}')
    return loaded, X_all, meta


def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    keep = labels != -1
    X, labels = X[keep], labels[keep]
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return np.nan
    if len(X) > MAX_SILHOUETTE_SAMPLES:
        rng = np.random.default_rng(SEED)
        ix = rng.choice(len(X), MAX_SILHOUETTE_SAMPLES, replace=False)
        X, labels = X[ix], labels[ix]
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return np.nan


def purity(y, clusters):
    d = pd.DataFrame({'y': pd.Series(y).astype(str), 'cluster': clusters})
    d = d[d.cluster != -1]
    if d.empty:
        return np.nan
    return d.groupby('cluster').y.apply(lambda x: x.value_counts().max()).sum() / len(d)


def ext_metrics(y, clusters):
    y = pd.Series(y).astype(str).to_numpy()
    c = np.asarray(clusters)
    keep = c != -1
    return {
        'ARI_non_noise': adjusted_rand_score(y[keep], c[keep]) if keep.sum() else np.nan,
        'NMI_non_noise': normalized_mutual_info_score(y[keep], c[keep]) if keep.sum() else np.nan,
        'ARI_noise_as_cluster': adjusted_rand_score(y, c),
        'NMI_noise_as_cluster': normalized_mutual_info_score(y, c),
    }


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

# ------------------------- CLASSIFICATION PROBE -------------------------
def linear_probe(name, loaded, outdir):
    """Auxiliary representation comparison, not the original neural classifier."""
    Xtr, mtr = loaded['train']; Xv, mv = loaded['validation']; Xte, mte = loaded['test']
    ytr = mtr.Class.astype(str).to_numpy()
    probe = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED)
    probe.fit(Xtr, ytr)  # raw embeddings: do not use the clustering scaler here
    rows = []
    for split, X, m in [('validation', Xv, mv), ('test', Xte, mte)]:
        y = m.Class.astype(str).to_numpy()
        pred = probe.predict(X)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
        _, _, wf1, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
        rows.append({'experiment': name, 'evaluation': 'linear_probe', 'split': split,
                     'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
                     'macro_precision': p, 'macro_recall': r, 'macro_F1': f1, 'weighted_F1': wf1})
        pd.DataFrame({'path': m.path, 'split': split, 'true_class': y,
                      'predicted_class': pred, 'correct': y == pred}).to_csv(
            outdir / f'{name}_{split}_linear_probe_predictions.csv', index=False)
        pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T.to_csv(
            outdir / f'{name}_{split}_linear_probe_class_report.csv')
        labels = sorted(set(y) | set(pred))
        cm = confusion_matrix(y, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(8, len(labels)*.4), max(6, len(labels)*.35)))
        im = ax.imshow(cm, aspect='auto'); fig.colorbar(im, ax=ax)
        ax.set(title=f'{name} {split}: linear-probe confusion matrix', xlabel='Predicted', ylabel='True')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        savefig(outdir / f'{name}_{split}_linear_probe_confusion_matrix.png')
    return rows

# ------------------------- CLUSTERING COMPARISON -------------------------
def clustering_analysis(name, X_all, meta, outdir):
    # Fit scaler on TRAIN ONLY, then apply it consistently to all splits.
    train_mask = meta.split.eq('train').to_numpy()
    scaler = StandardScaler().fit(X_all[train_mask])
    Xs = scaler.transform(X_all)
    joblib.dump(scaler, outdir / f'{name}_scaler_trainfit.joblib')

    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20)
    km_labels = km.fit_predict(Xs)  # descriptive all-split clustering; labels never used in fitting
    agg_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(Xs)

    # Agglomerative dendrogram: use a reproducible sample of TRAIN embeddings.
    # A full dendrogram for ~25k images is expensive and unreadable, so use a
    # truncated view of the last 30 merges from at most 1,000 training images.
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(SEED)
    dendrogram_n = min(1000, len(train_indices))
    dendrogram_indices = np.sort(rng.choice(train_indices, size=dendrogram_n, replace=False))
    X_dendrogram = Xs[dendrogram_indices]
    if len(X_dendrogram) >= 2:
        Z = linkage(X_dendrogram, method='ward', metric='euclidean')
        fig, ax = plt.subplots(figsize=(15, 7))
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,
            show_leaf_counts=True,
            show_contracted=True,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
        )
        ax.set_title(
            f'{name}: Agglomerative dendrogram (Ward linkage; {dendrogram_n} sampled training images)'
        )
        ax.set_xlabel('Merged groups (parenthetical values show leaf counts)')
        ax.set_ylabel('Ward linkage distance')
        savefig(outdir / f'{name}_agglomerative_dendrogram.png')
        pd.DataFrame({
            'sampled_row_index': dendrogram_indices,
            'path': meta.iloc[dendrogram_indices].path.to_numpy(),
            'Class': meta.iloc[dendrogram_indices].Class.to_numpy(),
            'Category': meta.iloc[dendrogram_indices].Category.to_numpy(),
            'Plant_Part': meta.iloc[dendrogram_indices].Plant_Part.to_numpy(),
        }).to_csv(outdir / f'{name}_agglomerative_dendrogram_sample.csv', index=False)
    try:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                              metric='euclidean', prediction_data=True)
        hdb_labels = hdb.fit_predict(Xs)
        hdb_prob = hdb.probabilities_
    except ImportError:
        print('[WARNING] hdbscan not installed; HDBSCAN results will be skipped.')
        hdb_labels = np.full(len(meta), -1); hdb_prob = np.full(len(meta), np.nan)

    assignments = meta.copy()
    assignments['KMeans_cluster'] = km_labels
    assignments['Agglomerative_cluster'] = agg_labels
    assignments['HDBSCAN_cluster'] = hdb_labels
    assignments['HDBSCAN_probability'] = hdb_prob
    assignments.to_csv(outdir / f'{name}_all_splits_cluster_assignments.csv', index=False)

    metric_rows = []
    algs = {'KMeans': km_labels, 'Agglomerative': agg_labels, 'HDBSCAN': hdb_labels}
    for alg, cl in algs.items():
        nonnoise = cl != -1
        row = {'experiment': name, 'algorithm': alg,
               'n_clusters_excluding_noise': len(set(cl[nonnoise])),
               'noise_rate': float((cl == -1).mean()) if alg == 'HDBSCAN' else 0.0,
               'silhouette_non_noise': safe_silhouette(Xs, cl),
               'class_purity': purity(meta.Class, cl)}
        for col in ['Class', 'Category', 'Plant_Part']:
            for key, val in ext_metrics(meta[col], cl).items(): row[f'{col}_{key}'] = val
        if alg != 'HDBSCAN':
            try: row['davies_bouldin'] = davies_bouldin_score(Xs, cl)
            except Exception: row['davies_bouldin'] = np.nan
            try: row['calinski_harabasz'] = calinski_harabasz_score(Xs, cl)
            except Exception: row['calinski_harabasz'] = np.nan
        else:
            row['davies_bouldin'] = np.nan; row['calinski_harabasz'] = np.nan
        metric_rows.append(row)

        sizes = pd.Series(cl).value_counts().sort_index()
        sizes.rename_axis('cluster').reset_index(name='n_images').to_csv(outdir / f'{name}_{alg}_cluster_sizes.csv', index=False)
        fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(sizes.index.astype(str), sizes.values)
        ax.set(title=f'{name}: {alg} cluster sizes', xlabel='Cluster (-1 = HDBSCAN noise)', ylabel='Images')
        savefig(outdir / f'{name}_{alg}_cluster_sizes.png')

        for col in ['Class', 'Category', 'Plant_Part']:
            ct = pd.crosstab(pd.Series(cl, name='cluster'), meta[col])
            ct.to_csv(outdir / f'{name}_{alg}_cluster_by_{col}.csv')
            comp = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            fig, ax = plt.subplots(figsize=(max(9, comp.shape[1]*.45), max(5, comp.shape[0]*.4)))
            im = ax.imshow(comp.to_numpy(), aspect='auto', vmin=0, vmax=1); fig.colorbar(im, ax=ax)
            ax.set(title=f'{name}: {alg} composition by {col}', xlabel=col, ylabel='Cluster')
            ax.set_xticks(range(comp.shape[1])); ax.set_xticklabels(comp.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(comp.shape[0])); ax.set_yticklabels(comp.index.astype(str), fontsize=8)
            savefig(outdir / f'{name}_{alg}_composition_{col}.png')

        # Interpretation summary: cluster size, dominant class/category/plant part and purity.
        temp = pd.DataFrame({'cluster': cl, 'Class': meta.Class.to_numpy(),
                             'Category': meta.Category.to_numpy(), 'Plant_Part': meta.Plant_Part.to_numpy()})
        summary = []
        for cid, g in temp.groupby('cluster'):
            summary.append({'cluster': cid, 'size': len(g),
                            'dominant_class': g.Class.mode().iloc[0],
                            'class_purity': g.Class.value_counts(normalize=True).max(),
                            'dominant_category': g.Category.mode().iloc[0],
                            'dominant_plant_part': g.Plant_Part.mode().iloc[0]})
        pd.DataFrame(summary).sort_values('size', ascending=False).to_csv(
            outdir / f'{name}_{alg}_cluster_interpretation.csv', index=False)

    # K-means k sensitivity: inertia and silhouette. Metrics use scaled embeddings, not UMAP.
    ks = range(2, min(20, len(Xs)-1) + 1)
    sweep = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        lab = model.fit_predict(Xs)
        sweep.append({'k': k, 'inertia': model.inertia_, 'silhouette': safe_silhouette(Xs, lab)})
    sweep = pd.DataFrame(sweep); sweep.to_csv(outdir / f'{name}_kmeans_k_sensitivity.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax1.plot(sweep.k, sweep.inertia, marker='o')
    ax1.set(xlabel='K', ylabel='Inertia', title=f'{name}: K-means k sensitivity')
    ax2 = ax1.twinx(); ax2.plot(sweep.k, sweep.silhouette, marker='s'); ax2.set_ylabel('Silhouette')
    savefig(outdir / f'{name}_kmeans_elbow_silhouette.png')

    # Representative examples nearest K-means centroids in scaled embedding space.
    reps = []
    for c in range(N_CLUSTERS):
        ix = np.where(km_labels == c)[0]
        if not len(ix): continue
        ds = np.linalg.norm(Xs[ix] - km.cluster_centers_[c], axis=1)
        for rank, j in enumerate(ix[np.argsort(ds)[:N_REPRESENTATIVES]], start=1):
            reps.append({'cluster': c, 'rank': rank, 'distance_to_centroid': float(np.linalg.norm(Xs[j]-km.cluster_centers_[c])),
                         'path': meta.iloc[j].path, 'Class': meta.iloc[j].Class,
                         'Category': meta.iloc[j].Category, 'Plant_Part': meta.iloc[j].Plant_Part,
                         'split': meta.iloc[j].split})
    pd.DataFrame(reps).to_csv(outdir / f'{name}_kmeans_representative_images.csv', index=False)

    # UMAP is visualisation only; fit on all embeddings for a shared exploratory view.
    try:
        import umap
        coords = umap.UMAP(n_neighbors=15, min_dist=.1, n_components=2, random_state=SEED).fit_transform(Xs)
        um = assignments.copy(); um['umap_x'] = coords[:, 0]; um['umap_y'] = coords[:, 1]
        um.to_csv(outdir / f'{name}_umap_coordinates.csv', index=False)
        for col in ['Class', 'Category', 'Plant_Part', 'KMeans_cluster', 'Agglomerative_cluster', 'HDBSCAN_cluster']:
            labs = um[col].astype(str)
            fig, ax = plt.subplots(figsize=(9, 7))
            for lab in labs.unique():
                ix = labs.eq(lab).to_numpy(); ax.scatter(coords[ix, 0], coords[ix, 1], s=5, alpha=.65, label=lab)
            ax.set(title=f'{name}: UMAP by {col}', xlabel='UMAP 1', ylabel='UMAP 2')
            if labs.nunique() <= 25: ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, markerscale=2)
            savefig(outdir / f'{name}_umap_by_{col}.png')
    except ImportError:
        print('[WARNING] umap-learn not installed; skipping UMAP.')
    return metric_rows

# ------------------------- CLUSTER-AWARE INFERENCE -------------------------
def cluster_aware_inference(name, loaded, outdir):
    """Prototype-based demonstration. Replace proxy predictions with original-model predictions for final claims."""
    Xtr, mtr = loaded['train']; Xv, mv = loaded['validation']; Xte, mte = loaded['test']
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xv), scaler.transform(Xte)
    joblib.dump(scaler, outdir / f'{name}_inference_scaler_trainfit.joblib')
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20).fit(Xtr_s)
    ctr, cv, cte = km.labels_, km.predict(Xv_s), km.predict(Xte_s)
    ytr = mtr.Class.astype(str).to_numpy()

    # Class-specific prototype means; only training embeddings are used.
    prototypes = {}
    proto_rows = []
    for c in np.unique(ctr):
        for cls in np.unique(ytr):
            ix = (ctr == c) & (ytr == cls)
            if ix.sum() >= 3:
                prototypes[(int(c), str(cls))] = Xtr_s[ix].mean(axis=0)
                proto_rows.append({'cluster': int(c), 'Class': str(cls), 'n_train': int(ix.sum())})
    if not prototypes: raise RuntimeError('No class-specific prototypes. Check labels and N_CLUSTERS.')
    pd.DataFrame(proto_rows).to_csv(outdir / f'{name}_class_specific_prototypes.csv', index=False)

    def score(X, meta, assigned_cluster):
        # A nearest-prototype class is used as a transparent proxy because this
        # script does not assume the architecture/checkpoint of the original net.
        # For the final thesis, substitute actual neural-model predictions here.
        preds, dists, proto_clusters = [], [], []
        for x in X:
            candidates = [(np.linalg.norm(x-mu), cls, c) for (c, cls), mu in prototypes.items()]
            d, cls, c = min(candidates, key=lambda q: q[0])
            preds.append(cls); dists.append(float(d)); proto_clusters.append(c)
        return meta.Class.astype(str).to_numpy(), np.asarray(preds), np.asarray(dists), np.asarray(proto_clusters)

    yv, pv, dv, pvcv = score(Xv_s, mv, cv)
    yt, pt, dt, pvct = score(Xte_s, mte, cte)
    delta = float(np.percentile(dv, ATYPICAL_PERCENTILE))

    metrics = []
    for split, y, pred, dist in [('validation', yv, pv, dv), ('test', yt, pt, dt)]:
        flags = dist > delta; accepted = ~flags
        metrics.append({'experiment': name, 'split': split, 'delta': delta,
                        'threshold_percentile': ATYPICAL_PERCENTILE, 'n': len(y),
                        'flagged_n': int(flags.sum()), 'flag_rate': float(flags.mean()),
                        'coverage_unflagged': float(accepted.mean()),
                        'accuracy_all_proxy': accuracy_score(y, pred),
                        'accuracy_unflagged_proxy': accuracy_score(y[accepted], pred[accepted]) if accepted.any() else np.nan,
                        'error_rate_unflagged_proxy': 1-accuracy_score(y[accepted], pred[accepted]) if accepted.any() else np.nan,
                        'median_distance': float(np.median(dist)), 'p90_distance': float(np.percentile(dist, 90)),
                        'p95_distance': float(np.percentile(dist, 95)), 'p99_distance': float(np.percentile(dist, 99))})
        meta = mv if split == 'validation' else mte
        assigned = cv if split == 'validation' else cte
        pc = pvcv if split == 'validation' else pvct
        out = meta.copy(); out['cluster_id'] = assigned; out['prototype_cluster_id'] = pc
        out['true_class'] = y; out['predicted_class_proxy'] = pred; out['correct_proxy'] = y == pred
        out['prototype_distance'] = dist; out['threshold_delta'] = delta; out['atypical_flag'] = flags
        out.to_csv(outdir / f'{name}_{split}_cluster_aware_inference.csv', index=False)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(dist, bins=40, alpha=.8)
        ax.axvline(delta, linestyle='--', linewidth=2, label=f'delta={delta:.3f}')
        ax.set(title=f'{name} {split}: distance threshold', xlabel='Distance to nearest class-specific prototype', ylabel='Images')
        ax.legend(); savefig(outdir / f'{name}_{split}_distance_distribution.png')

    # Risk-coverage curve on test: retain the lowest-distance fraction.
    order = np.argsort(dt); risk = []
    for coverage in range(10, 101, 5):
        n = max(1, int(np.ceil(len(order)*coverage/100))); ix = order[:n]
        acc = accuracy_score(yt[ix], pt[ix])
        risk.append({'experiment': name, 'target_coverage_pct': coverage, 'actual_coverage': n/len(order),
                     'accepted_accuracy_proxy': acc, 'accepted_error_rate_proxy': 1-acc,
                     'distance_cutoff': float(dt[ix[-1]])})
    risk = pd.DataFrame(risk); risk.to_csv(outdir / f'{name}_test_risk_coverage.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(risk.actual_coverage, risk.accepted_error_rate_proxy, marker='o')
    ax.set(title=f'{name}: test risk–coverage (prototype proxy)', xlabel='Coverage accepted', ylabel='Error among accepted')
    savefig(outdir / f'{name}_test_risk_coverage.png')
    return metrics

# ------------------------- RUN ALL FOUR EXPERIMENTS -------------------------
all_probe, all_clusters, all_inference, errors = [], [], [], []
for name, paths in EXPERIMENTS.items():
    print(f'\n===== {name} =====')
    try:
        loaded, X_all, meta = load_experiment(name, paths)
        outdir = OUTPUT_DIR / name; outdir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'validation', 'test']:
            print(f'{split}: {loaded[split][0].shape}; classes={loaded[split][1].Class.nunique()}')
        all_probe.extend(linear_probe(name, loaded, outdir))
        all_clusters.extend(clustering_analysis(name, X_all, meta, outdir))
        all_inference.extend(cluster_aware_inference(name, loaded, outdir))
    except Exception as e:
        print(f'[ERROR] {name}: {type(e).__name__}: {e}')
        errors.append({'experiment': name, 'error_type': type(e).__name__, 'error': str(e)})

if all_probe: pd.DataFrame(all_probe).to_csv(OUTPUT_DIR/'ALL_experiments_linear_probe_metrics.csv', index=False)
if all_clusters: pd.DataFrame(all_clusters).to_csv(OUTPUT_DIR/'ALL_experiments_clustering_metrics.csv', index=False)
if all_inference: pd.DataFrame(all_inference).to_csv(OUTPUT_DIR/'ALL_experiments_cluster_aware_metrics.csv', index=False)
if errors: pd.DataFrame(errors).to_csv(OUTPUT_DIR/'ERRORS_to_fix.csv', index=False)
print(f'\nFinished. Results saved under: {OUTPUT_DIR.resolve()}')
print('Note: linear-probe and prototype-proxy predictions are not your original neural classifier results.')
print('No anomaly AUROC/AP is calculated because ground-truth atypical labels are not assumed.')
