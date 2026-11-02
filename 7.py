# CSL7011 MACHINE LEARNING LAB - EXPERIMENT 7
# Title: Graph-based clustering (Spectral Clustering) and DBSCAN comparison
# K-means and K-medoids are NOT used.
# Dataset: Generated two-moon data, which forms curved clusters.

# 1. AUTOMATIC INSTALLATION
import sys
import subprocess
import importlib.util

packages = {"numpy": "numpy", "matplotlib": "matplotlib",
            "scikit-learn": "sklearn"}

for package, module in packages.items():
    if importlib.util.find_spec(module) is None:
        print("Installing", package)
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 2. IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

# 3. CREATE A DATASET WITH CURVED CLUSTERS
# This data is generated in code, so no CSV download is needed.
X, original_labels = make_moons(n_samples=300, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

print("Dataset: generated two-moon points")
print("Number of points:", X.shape[0])
print("Number of features:", X.shape[1])

# 4. GRAPH-BASED CLUSTERING USING SPECTRAL CLUSTERING
# It builds a nearest-neighbour similarity graph, then clusters the graph representation.
spectral = SpectralClustering(
    n_clusters=2,
    affinity="nearest_neighbors",
    n_neighbors=10,
    random_state=42
)
spectral_labels = spectral.fit_predict(X)

# 5. DBSCAN AS AN ADDITIONAL DENSITY-BASED CLUSTERING METHOD
# eps is neighbourhood distance; min_samples is the density requirement.
dbscan = DBSCAN(eps=0.25, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# 6. COUNT CLUSTERS AND NOISE
def cluster_summary(labels):
    cluster_ids = set(labels)
    number_of_clusters = len(cluster_ids - {-1})
    number_of_noise = int(np.sum(labels == -1))
    return number_of_clusters, number_of_noise

spectral_count, spectral_noise = cluster_summary(spectral_labels)
dbscan_count, dbscan_noise = cluster_summary(dbscan_labels)

print("\n--- Spectral Clustering ---")
print("Cluster labels:", sorted(set(spectral_labels)))
print("Number of clusters:", spectral_count)

print("\n--- DBSCAN ---")
print("Cluster labels (label -1 means noise):", sorted(set(dbscan_labels)))
print("Number of clusters:", dbscan_count)
print("Number of noise points:", dbscan_noise)

# 7. SILHOUETTE SCORE (only calculated when at least two clusters exist)
if len(set(spectral_labels)) > 1:
    print("Spectral silhouette score:",
          round(silhouette_score(X, spectral_labels), 4))

# For DBSCAN, calculate score on non-noise points only if two clusters remain.
non_noise = dbscan_labels != -1
dbscan_non_noise_labels = dbscan_labels[non_noise]
if len(set(dbscan_non_noise_labels)) >= 2:
    print("DBSCAN silhouette score (excluding noise):",
          round(silhouette_score(X[non_noise], dbscan_non_noise_labels), 4))
else:
    print("DBSCAN silhouette score not calculated: fewer than two non-noise clusters.")

# 8. VISUALIZE DATA AND CLUSTERING RESULTS
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=original_labels)
axes[0].set_title("Generated Data")

axes[1].scatter(X[:, 0], X[:, 1], c=spectral_labels)
axes[1].set_title("Spectral Clustering (Graph-based)")

axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels)
axes[2].set_title("DBSCAN (Noise = -1)")

plt.tight_layout()
plt.show()

# # EXPERIMENT 7: GRAPH-BASED CLUSTERING

# import sys
# import subprocess
# import importlib.util

# # AUTOMATIC INSTALLATION
# packages = {
#     "numpy": "numpy",
#     "matplotlib": "matplotlib",
#     "scikit-learn": "sklearn"
# }

# for package, module in packages.items():
#     if importlib.util.find_spec(module) is None:
#         subprocess.check_call(
#             [sys.executable, "-m", "pip", "install", package]
#         )

# # IMPORT LIBRARIES
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_moons
# from sklearn.cluster import SpectralClustering, DBSCAN
# from sklearn.metrics import silhouette_score

# # GENERATE SAMPLE DATA
# X, actual_labels = make_moons(
#     n_samples=300, noise=0.08, random_state=42
# )

# # SPECTRAL CLUSTERING
# spectral = SpectralClustering(
#     n_clusters=2,
#     affinity="nearest_neighbors",
#     n_neighbors=10,
#     random_state=42
# )

# spectral_labels = spectral.fit_predict(X)

# # DBSCAN
# dbscan = DBSCAN(eps=0.2, min_samples=5)
# dbscan_labels = dbscan.fit_predict(X)

# # DISPLAY RESULTS
# print("SPECTRAL CLUSTER LABELS:", set(spectral_labels))
# print("DBSCAN CLUSTER LABELS:", set(dbscan_labels))
# print("DBSCAN NOISE POINTS:", sum(dbscan_labels == -1))

# if len(set(spectral_labels)) > 1:
#     print(
#         "SPECTRAL SILHOUETTE SCORE:",
#         silhouette_score(X, spectral_labels)
#     )

# # VISUALIZATION
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# axes[0].scatter(X[:, 0], X[:, 1], c=actual_labels)
# axes[0].set_title("Original Data")

# axes[1].scatter(X[:, 0], X[:, 1], c=spectral_labels)
# axes[1].set_title("Spectral Clustering")

# axes[2].scatter(X[:, 0], X[:, 1], c=dbscan_labels)
# axes[2].set_title("DBSCAN")

# plt.tight_layout()
# plt.show()