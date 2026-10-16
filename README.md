Dependencies (versions used)

Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, openpyxl 3.1.5

Install: python -m pip install scikit-learn pandas numpy matplotlib openpyxl scipy

Run

Put Dry_Bean_Dataset.xlsx in the same folder as dry_bean_clustering.py, then:

python dry_bean_clustering.py

Optional: set LAST_NAME (PowerShell: $env:LAST_NAME="Kieu") to name the output CSV <LAST_NAME>_DryBeanClusters.csv. Paths can be overridden with BEAN_XLSX and OUT_DIR.

Runtime: about a minute on a laptop.

Random seeds

42 for the reported results; seeds 0-9 for the stability check.

Expected outputs (in outputs/)
<LAST_NAME>_DryBeanClusters.csv (sample_id, Class_id; 13,611 rows)
clustering_comparison.csv, config_agreement_ARI.csv
pca2d_selected_configs.png, final_cluster_profile_heatmap.png, pca_cumulative_variance.png
final_cluster_profile_means.csv, summary.json

The Class column is never read by any step.
