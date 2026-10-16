"""
Programmer: Khai Kieu
Project 2: Dry Bean clustering (K-means and Gaussian Mixture, k = 7)

Four configurations:
  KMeans / GMM  x  standardized features / standardized + PCA.
The Class column is never used. Run:  python dry_bean_clustering.py
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
K = 7
PCA_VAR = 0.95            # keep enough components for >= 95% explained variance
STAB_SEEDS = range(10)    # seeds used for the stability check
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BEAN_XLSX", os.path.join(HERE, "Dry_Bean_Dataset.xlsx"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "outputs"))
LAST_NAME = os.environ.get("LAST_NAME", "LastName")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data prep
df = pd.read_excel(DATA, sheet_name="Dry_Beans_Dataset")
ids = df["sample_id"].to_numpy()
X_raw = df.drop(columns=["sample_id", "Class"])      # Class excluded entirely
feats = list(X_raw.columns)
prep = {
    "n_samples": len(df),
    "n_features": len(feats),
    "missing_values": int(X_raw.isna().sum().sum()),
    "duplicate_feature_rows": int(X_raw.duplicated().sum()),
    "rows_with_any_abs_z_gt_4": int(((X_raw - X_raw.mean()) / X_raw.std()).abs().gt(4).any(axis=1).sum()),
}
print("Preparation:", prep)

# Duplicates and unusual rows are kept: every sample needs an assignment and
# the extreme values look like genuine large/small beans, not errors.
X = StandardScaler().fit_transform(X_raw)             # unsupervised, no labels

pca_full = PCA(random_state=SEED).fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n_pc = int(np.searchsorted(cum, PCA_VAR) + 1)
pca_info = {"n_components": n_pc, "cumulative_explained_variance": [round(float(c), 4) for c in cum[:n_pc + 2]]}
print("PCA:", pca_info)
Z = PCA(n_components=n_pc, random_state=SEED).fit_transform(X)
V2 = PCA(n_components=2, random_state=SEED).fit_transform(X)   # visualization only


def fit_predict(method, data, seed):
    if method == "KMeans":
        return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(data)
    return GaussianMixture(n_components=K, covariance_type="full", n_init=3,
                           random_state=seed).fit(data).predict(data)


# --------------------------------------------------------------- experiments
spaces = {"Standardized": X, f"Standardized+PCA({n_pc})": Z}
rows, labels = [], {}
for method, (sname, S) in itertools.product(["KMeans", "GMM"], spaces.items()):
    lab = fit_predict(method, S, SEED)
    runs = [fit_predict(method, S, s) for s in STAB_SEEDS]
    stab = np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)])
    rows.append({"method": method, "space": sname,
                 "silhouette": round(silhouette_score(S, lab), 4),
                 "silhouette_in_standardized_space": round(silhouette_score(X, lab), 4),
                 "stability_ARI_10_seeds": round(float(stab), 4),
                 "cluster_sizes": sorted(np.bincount(lab).tolist(), reverse=True)})
    labels[(method, sname)] = lab
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/clustering_comparison.csv", index=False)
print(res.to_string(index=False))

# agreement between configurations (how much PCA changes the partition)
keys = list(labels)
agree = pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in keys] for a in keys],
                     index=[" / ".join(k) for k in keys], columns=[" / ".join(k) for k in keys]).round(3)
agree.to_csv(f"{OUT}/config_agreement_ARI.csv")
print(agree.to_string())

# ------------------------------------------------- selection and final output
# Selected configuration per method: higher silhouette, ties broken by stability.
sel = {}
for m in ["KMeans", "GMM"]:
    sub = res[res.method == m].sort_values(["silhouette_in_standardized_space", "stability_ARI_10_seeds"], ascending=False)
    sel[m] = sub.iloc[0]["space"]
final_method = res[res.apply(lambda r: r.space == sel[r.method], axis=1)] \
    .sort_values(["silhouette_in_standardized_space", "stability_ARI_10_seeds"], ascending=False).iloc[0]["method"]
final_space = sel[final_method]
final = labels[(final_method, final_space)]
print("Selected per method:", sel, "| FINAL:", final_method, final_space)

pd.DataFrame({"sample_id": ids, "Class_id": final}).to_csv(
    f"{OUT}/{LAST_NAME}_DryBeanClusters.csv", index=False)

# ------------------------------------------------------------ visualizations
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, m in zip(ax, ["KMeans", "GMM"]):
    lab = labels[(m, sel[m])]
    a.scatter(V2[:, 0], V2[:, 1], c=lab, cmap="tab10", s=4)
    a.set_title(f"{m} on {sel[m]}")
    a.set_xlabel("PC1 (visualization only)")
    a.set_ylabel("PC2 (visualization only)")
plt.tight_layout()
plt.savefig(f"{OUT}/pca2d_selected_configs.png", dpi=150)
plt.close()

# profile of the final configuration in original feature units
prof = X_raw.assign(cluster=final).groupby("cluster").agg("mean")
prof.insert(0, "n", pd.Series(final).value_counts().sort_index())
prof.round(4).to_csv(f"{OUT}/final_cluster_profile_means.csv")
zprof = pd.DataFrame(X, columns=feats).assign(cluster=final).groupby("cluster").mean()
fig, a = plt.subplots(figsize=(10, 4))
im = a.imshow(zprof.T.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
a.set_xticks(range(K)); a.set_xticklabels([f"C{i}\n(n={int((final == i).sum())})" for i in range(K)])
a.set_yticks(range(len(feats))); a.set_yticklabels(feats, fontsize=7)
plt.colorbar(im, label="mean z-score")
a.set_title(f"Final clusters: {final_method} on {final_space}")
plt.tight_layout()
plt.savefig(f"{OUT}/final_cluster_profile_heatmap.png", dpi=150)
plt.close()

fig, a = plt.subplots(figsize=(5, 3.5))
a.plot(range(1, len(cum) + 1), cum, marker="o")
a.axhline(PCA_VAR, ls="--", c="gray"); a.axvline(n_pc, ls="--", c="gray")
a.set_xlabel("Number of components"); a.set_ylabel("Cumulative explained variance")
plt.tight_layout()
plt.savefig(f"{OUT}/pca_cumulative_variance.png", dpi=150)
plt.close()

with open(f"{OUT}/summary.json", "w") as f:
    json.dump({"prep": prep, "pca": pca_info, "selected": sel,
               "final": [str(final_method), str(final_space)]}, f, indent=2)